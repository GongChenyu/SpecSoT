# coding=utf-8
"""
分布式Prefill管理器

该模块封装了分布式Prefill阶段的核心逻辑：
1. Base Model 的层级分布式计算（每层异步同步cache）
2. Eagle Layer 的计算（全部在最后一个rank执行）
3. Draft Tokens 打包传输（最高优先级）
4. Eagle Input Hidden States 传输（特定层：2, num_layers//2, num_layers-3）

分布式流程：
- Prefill阶段：各rank计算各自负责的层
  - 异步发送base model cache
  - 收集并发送eagle input hidden states
- Eagle Layer：只在最后一个rank执行
- Draft Tokens：最后一个rank生成后广播给所有rank

通信优先级和时机：
1. DRAFT_TOKENS: 最高优先级，只在最后一个chunk的最后一个stage同步一次
2. HIDDEN: 第二优先级，阻塞主线程，每个chunk的每个stage结束才传输
3. EAGLE_INPUT_HIDDEN: 第三优先级，异步传输，在指定layer传输，eagle layer计算前阻塞等待
4. BASE_CACHE: 第四优先级，异步传输，每个chunk的每个layer都同步
5. EAGLE_CACHE: 第五优先级，异步传输，每个chunk的expand_root后增量同步

日志记录说明：
- DEBUG: 详细的层级处理信息、cache同步细节
- INFO: 重要阶段开始/结束、关键通信事件
- WARNING: 超时、异常情况
- ERROR: 错误信息
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessorList

from .distributed_config import DistributedConfig
from .comm_manager import create_zmq_comm_manager, ZMQCommManagerBase, MessageType, Message
from ..kv_cache import KVCache


def setup_prefill_logger(rank: int, log_dir: str = None) -> logging.Logger:
    """
    设置分布式Prefill模块的日志记录器
    
    Args:
        rank: 当前进程rank
        log_dir: 日志目录
        
    Returns:
        配置好的logger
    """
    logger_name = f"DistPrefill-Rank{rank}"
    logger = logging.getLogger(logger_name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # 日志格式
    log_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"prefill_rank{rank}.log")
        # log_file = os.path.join(log_dir, f"prefill_rank{rank}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_tensor_info(tensor: torch.Tensor) -> str:
    """获取tensor的详细信息字符串"""
    if tensor is None:
        return "None"
    
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype).replace('torch.', '')
    device = str(tensor.device)
    numel = tensor.numel()
    size_bytes = tensor.element_size() * numel
    size_mb = size_bytes / (1024 * 1024)
    
    return f"shape={shape}, dtype={dtype}, device={device}, size={size_mb:.3f}MB"


class DistributedPrefillManager:
    """
    分布式Prefill管理器
    
    负责协调分布式Prefill阶段的计算和通信
    
    Attributes:
        config: 分布式配置
        comm: ZMQ通信管理器
        model: SpecSoT模型引用
        logger: 日志记录器
        timing_stats: 时间统计
    """
    
    def __init__(
        self,
        config: DistributedConfig,
        model: nn.Module,
        device: str = "cuda",
    ):
        """
        初始化分布式Prefill管理器
        
        Args:
            config: 分布式配置
            model: SpecSoT模型
            device: 设备
        """
        self.config = config
        self.model = model
        self.device = device
        
        # 设置日志（带文件输出）
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_dir, 'logs')
        self.logger = setup_prefill_logger(config.rank, log_dir)
        
        # 通信管理器
        self.comm: Optional[ZMQCommManagerBase] = None
        
        # 时间统计
        self.timing_stats = {
            'prefill_start': 0,
            'prefill_compute_end': 0,
            'prefill_end': 0,
            'cache_sync_time': 0,
            'hidden_send_time': 0,
            'hidden_recv_time': 0,
            'eagle_input_hidden_time': 0,
            'draft_tokens_time': 0,
        }
        
        # Cache接收状态
        self.cache_received_indicator = None
        self.num_chunks = 0
        
        # Eagle Cache接收状态 [num_chunks] - 每个chunk同步一次
        self.eagle_cache_received_indicator = None
        
        # Eagle Layer stable_kv 同步状态
        self.eagle_kv_received = False
        
        # Draft Tree 接收状态
        self.draft_tree_received = False
        
        # Prefill阶段标记 - 控制eagle cache是否同步
        self._is_prefill_phase = False
        
        # 初始化通信
        if config.enabled:
            self._init_communication()
    
    def _init_communication(self):
        """初始化ZMQ通信"""
        self.logger.info(f"[INIT] 初始化ZMQ通信管理器 | mode={self.config.comm_mode}, base_port={self.config.base_port}")
        
        self.comm = create_zmq_comm_manager(
            rank=self.config.rank,
            world_size=self.config.world_size,
            base_port=self.config.base_port,
            mode=self.config.comm_mode,
            device=self.device,
            node_addresses=self.config.node_addresses
        )
        
        # 等待其他节点
        self.logger.info(f"[INIT] 等待其他节点就绪 ({self.config.startup_delay}秒)...")
        time.sleep(self.config.startup_delay)
        
        # 启动通信
        self.comm.start()
        self.logger.info("[INIT] 通信管理器已启动")
    
    def cleanup(self):
        """清理资源"""
        if self.comm is not None:
            # 打印通信统计信息
            stats = self.comm.get_stats()
            self.logger.info(f"[CLEANUP] 通信统计: {stats}")
            self.comm.stop()
            self.comm = None
        self.logger.info("[CLEANUP] 分布式Prefill管理器已清理")
    
    # =========================================================================
    # 分布式Prefill核心方法
    # =========================================================================
    
    def prefill_single_distributed(
        self,
        input_ids: torch.Tensor,
        past_key_values: List,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        分布式Prefill阶段（单序列模式，支持多chunk）
        
        流程：
        1. 将输入序列切分为多个chunks
        2. 对每个chunk:
           - 第一个rank：执行embedding
           - 各rank：计算各自负责的层，异步发送cache
           - 发送hidden states给下一个rank
        3. 最后一个chunk处理完后:
           - 最后一个rank：执行norm、lm_head、eagle layer生成draft tree
           - 广播draft tokens给所有rank
        4. 所有rank等待cache同步完成
        
        Args:
            input_ids: 输入token IDs [1, seq_len]
            past_key_values: Base Model的KV Cache
            logits_processor: logits处理器
            
        Returns:
            - 最后一个rank: (draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token)
            - 其他rank: None（无需返回值，只需等待同步完成）
        """
        self.timing_stats['prefill_start'] = time.time()
        
        # 设置prefill阶段标记
        self._is_prefill_phase = True
        
        base_model = self.model.base_model
        eagle_layer = self.model.eagle_layer
        num_layers = base_model.config.num_hidden_layers
        start_layer, end_layer = self.config.get_layer_range(num_layers)
        total_seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        self.logger.info("=" * 60)
        self.logger.info("[PREFILL] 开始分布式 Prefill 阶段")
        self.logger.info("=" * 60)
        self.logger.info(f"  Rank: {self.config.rank}/{self.config.world_size-1}")
        self.logger.info(f"  负责层范围: [{start_layer}, {end_layer})")
        self.logger.info(f"  输入序列长度: {total_seq_length}")
        self.logger.info(f"  模型总层数: {num_layers}")
        
        # 初始化cache接收状态
        self._init_cache_tracking(num_layers, total_seq_length)
        
        # 切分chunks
        chunks = self._split_into_chunks(input_ids)
        self.logger.info(f"  Chunk数量: {len(chunks)} (chunk_size={self.config.chunk_size})")
        
        # 确定需要传输给eagle layer的hidden state层
        eagle_input_layers = self._get_eagle_input_layers(num_layers)
        self.logger.info(f"  Eagle Input Layers: {eagle_input_layers}")
        eagle_hidden_states = {}  # 收集eagle input hidden states
        
        # =====================================================================
        # 多Chunk循环处理
        # =====================================================================
        for chunk_idx, chunk in enumerate(chunks):
            is_first_chunk = (chunk_idx == 0)
            is_last_chunk = (chunk_idx == len(chunks) - 1)
            chunk_seq_length = chunk.shape[1]
            
            self._current_chunk_idx = chunk_idx
            chunk_start_time = time.time()
            
            self.logger.info(f"\n[CHUNK {chunk_idx+1}/{len(chunks)}] 开始处理 | seq_len={chunk_seq_length}, first={is_first_chunk}, last={is_last_chunk}")
            
            # =================================================================
            # Phase 1: Embedding / 接收hidden states
            # =================================================================
            phase1_start = time.time()
            
            if self.config.is_first_rank():
                # 第一个rank执行embedding
                self.logger.info(f"  [PHASE 1] 执行 Embedding...")
                hidden_states = base_model.model.embed_tokens(chunk)
                self.logger.info(f"  [PHASE 1] Embedding 完成 | {get_tensor_info(hidden_states)}")
            else:
                # 接收上一个rank的hidden states（阻塞）
                prev_rank = self.config.get_prev_rank()
                self.logger.info(f"  [PHASE 1] 等待接收 Hidden States (from Rank {prev_rank})...")
                result = self.comm.recv_hidden(
                    src_rank=prev_rank, 
                    timeout=60.0
                )
                if result is None:
                    self.logger.error(f"  [PHASE 1][TIMEOUT] 接收hidden states超时 (chunk={chunk_idx})")
                    raise RuntimeError(f"Rank {self.config.rank}: 接收hidden states超时 (chunk={chunk_idx})")
                hidden_states, recv_chunk_idx = result
                self.logger.info(f"  [PHASE 1][RECV] Hidden States | {get_tensor_info(hidden_states)} | chunk_idx={recv_chunk_idx}")
            
            phase1_time = (time.time() - phase1_start) * 1000
            self.logger.debug(f"  [PHASE 1] 完成 | 耗时: {phase1_time:.2f}ms")
            
            # =================================================================
            # Phase 2: Decoder Layers (各rank计算各自负责的层)
            # =================================================================
            phase2_start = time.time()
            self.logger.info(f"  [PHASE 2] 开始计算 Decoder Layers [{start_layer}, {end_layer})")
            
            # 计算past_key_values_length
            past_key_values_length = 0
            if past_key_values[start_layer][0].current_length > 0:
                past_key_values_length = past_key_values[start_layer][0].current_length.item()
            
            seq_length_with_past = chunk_seq_length + past_key_values_length
            self.logger.debug(f"    past_key_values_length={past_key_values_length}, seq_with_past={seq_length_with_past}")
            
            # 准备position ids
            position_ids = torch.arange(
                past_key_values_length,
                chunk_seq_length + past_key_values_length,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            
            # 准备attention mask
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=self.device,
            )
            
            attention_mask = base_model.model._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, chunk_seq_length),
                hidden_states,
                past_key_values_length,
            )
            
            # 获取position embeddings (Qwen3需要)
            if hasattr(base_model.model, 'rotary_emb'):
                position_embeddings = base_model.model.rotary_emb(hidden_states, position_ids)
            else:
                position_embeddings = None
            
            # 清空当前chunk的eagle hidden states
            eagle_hidden_states.clear()
            
            # 计算当前rank负责的层
            layers_computed = 0
            layers_cache_sent = 0
            eagle_hidden_sent = 0
            
            for layer_idx in range(start_layer, end_layer):
                layer_start = time.time()
                decoder_layer = base_model.model.layers[layer_idx]
                
                # Forward with KV Cache
                if position_embeddings is not None:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values[layer_idx],
                        output_attentions=False,
                        use_cache=True,
                        position_embeddings=position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values[layer_idx],
                        output_attentions=False,
                        use_cache=True,
                    )
                
                hidden_states = layer_outputs[0]
                layers_computed += 1
                
                # 收集eagle input hidden states（每个chunk都要收集）
                if layer_idx in eagle_input_layers:
                    if self.config.is_last_rank():
                        # 最后一个rank直接保存
                        eagle_hidden_states[layer_idx] = hidden_states.clone()
                        self.logger.debug(f"    [EAGLE] 保存 hidden state layer={layer_idx}, chunk={chunk_idx}")
                    else:
                        # 非最后rank需要发送给最后一个rank，同时传输chunk_idx
                        self.comm.send_eagle_input_hidden(
                            hidden=hidden_states.clone(),
                            layer_idx=layer_idx,
                            dst_rank=self.config.world_size - 1,
                            chunk_idx=chunk_idx
                        )
                        eagle_hidden_sent += 1
                        self.logger.debug(f"    [SEND][EAGLE] hidden state layer={layer_idx}, chunk={chunk_idx} -> Rank {self.config.world_size - 1}")
                
                # 获取新增的KV并发送给其他rank
                current_len = past_key_values[layer_idx][0].current_length.item()
                new_key = past_key_values[layer_idx][0].data[:, :, current_len - chunk_seq_length:current_len, :]
                new_value = past_key_values[layer_idx][1].data[:, :, current_len - chunk_seq_length:current_len, :]
                
                cache_size_mb = (new_key.numel() + new_value.numel()) * new_key.element_size() / 1024 / 1024
                
                # 异步发送base cache给其他rank
                for other_rank in range(self.config.world_size):
                    if other_rank != self.config.rank:
                        self.comm.send_base_cache_async(
                            kv_cache=(new_key.clone(), new_value.clone()),
                            dst_rank=other_rank,
                            layer_idx=layer_idx,
                            chunk_idx=chunk_idx
                        )
                        layers_cache_sent += 1
                        self.logger.debug(f"    [SEND][CACHE] layer={layer_idx} -> Rank {other_rank} | size={cache_size_mb:.2f}MB")
                
                # 标记本层cache已发送
                self.cache_received_indicator[chunk_idx, layer_idx] = 1
            
            phase2_time = (time.time() - phase2_start) * 1000
            self.logger.info(f"  [PHASE 2] 完成 | layers={layers_computed}, cache_sent={layers_cache_sent}, eagle_hidden_sent={eagle_hidden_sent} | 耗时: {phase2_time:.2f}ms")
            
            # =================================================================
            # Phase 3: 发送hidden states给下一个rank / 执行Eagle Layer
            # =================================================================
            phase3_start = time.time()
            
            if not self.config.is_last_rank():
                # 非最后rank：发送hidden states给下一个rank
                next_rank = self.config.get_next_rank()
                self.logger.info(f"  [PHASE 3][SEND] Hidden States -> Rank {next_rank}")
                self.comm.send_hidden(
                    hidden_states, 
                    dst_rank=next_rank, 
                    chunk_idx=chunk_idx
                )
                self.logger.debug(f"    Hidden States 发送完成 | {get_tensor_info(hidden_states)}")
            else:
                # 最后一个rank：执行Eagle Layer计算（Pipeline的一部分）
                self.logger.info(f"  [PHASE 3] 执行 Eagle Layer 计算...")
                
                # Norm
                hidden_states_normed = base_model.model.norm(hidden_states)
                self.logger.debug(f"    Norm 完成 | {get_tensor_info(hidden_states_normed)}")
                
                # LM Head
                orig = base_model.lm_head(hidden_states_normed)
                self.logger.debug(f"    LM Head 完成 | {get_tensor_info(orig)}")
                
                # 接收其他rank发送的eagle input hidden states（阻塞等待收集完毕）
                self.logger.info(f"  [PHASE 3][RECV] 等待 Eagle Input Hidden States (chunk={chunk_idx})...")
                recv_start = time.time()
                received_eagle_hidden = self.comm.recv_all_eagle_input_hidden(
                    eagle_input_layers=eagle_input_layers,
                    my_start_layer=start_layer,
                    my_end_layer=end_layer,
                    chunk_idx=chunk_idx,
                    timeout=60.0
                )
                recv_time = (time.time() - recv_start) * 1000
                self.logger.info(f"  [PHASE 3][RECV] 收到 {len(received_eagle_hidden)} 个 Eagle Hidden (chunk={chunk_idx}) | 耗时: {recv_time:.2f}ms")
                
                # 合并本地和接收的eagle hidden states
                eagle_hidden_states.update(received_eagle_hidden)
                self.logger.debug(f"    合并后 Eagle Hidden States 层: {sorted(eagle_hidden_states.keys())}")

                # print(f"chunk={chunk_idx} layer=1 eagle_hidden_states[1][0, :5, :5]: {eagle_hidden_states[1][0, :5, :5]}")
                # print(f"chunk={chunk_idx} layer=17 eagle_hidden_states[17][0, :5, :5]: {eagle_hidden_states[17][0, :5, :5]}")
                
                # 准备Eagle Layer的hidden states
                if self.model.use_eagle3:
                    hidden_states_for_eagle = self._cat_eagle_hidden_states(
                        eagle_hidden_states, num_layers
                    )
                    self.logger.debug(f"    EAGLE3 Hidden States | {get_tensor_info(hidden_states_for_eagle)}")
                else:
                    hidden_states_for_eagle = hidden_states_normed
                
                # 只在最后一个chunk时采样token，补齐input_ids
                if is_last_chunk:
                    # Sample first token
                    if logits_processor is not None:
                        logits = orig[:, -1]
                        logits = logits_processor(input_ids, logits)
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        token = torch.multinomial(probabilities, 1)
                    else:
                        token = torch.argmax(orig[:, -1])
                        token = token[None, None]
                    
                    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                    self.logger.info(f"  [PHASE 3] First Token 采样: {token.item()}")

                # Eagle Layer生成Draft Tree（使用分布式Prefill专用函数）
                # 非最后chunk只更新stable_kv，最后chunk才生成完整的draft tree
                start_idx = chunk_idx * self.config.chunk_size + 1
                end_idx = total_seq_length + 1 if is_last_chunk else start_idx + chunk_seq_length
                input_ids_this_chunk = input_ids[:, start_idx:end_idx]
                
                self.logger.debug(f"    Draft Tree 生成: input_ids range=[{start_idx}, {end_idx})")
                # original_input_len: 原始完整input_ids的长度（包含first token和采样token）
                # 对于最后一个chunk: total_seq_length + 1 (因为添加了采样的token)
                # 与单机版 generate_draft_tree 中的 len_posi = input_ids.shape[1] 对齐
                original_len = total_seq_length + 1 if is_last_chunk else end_idx
                tree_result, incremental_kv = eagle_layer.generate_draft_tree_dist_prefill(
                    hidden_states_for_eagle, 
                    input_ids_this_chunk,  # 使用当前chunk的input_ids
                    is_last_chunk=is_last_chunk,
                    chunk_idx=chunk_idx,
                    original_input_len=original_len,
                )
                
                # 显式发送Eagle增量KV（每个chunk都执行，只有最后rank会真正发送）
                if self.config.is_last_rank() and incremental_kv is not None:
                    key, value = incremental_kv
                    kv_size_mb = (key.numel() + value.numel()) * key.element_size() / 1024 / 1024
                    self.logger.info(f"  [PHASE 3][BROADCAST] Eagle Stable KV | chunk={chunk_idx}, size={kv_size_mb:.2f}MB")
                    self.comm.broadcast_eagle_stable_kv(
                        incremental_kv=incremental_kv, 
                        chunk_idx=chunk_idx,
                    )
                
                # 只在最后一个chunk时广播结果和采样token
                if is_last_chunk:
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = tree_result
                    
                    # 保存最终结果
                    final_orig = orig
                    final_hidden_states = hidden_states_normed
                    final_token = token
                    final_draft_tokens = draft_tokens
                    final_retrieve_indices = retrieve_indices
                    final_tree_mask = tree_mask
                    final_tree_position_ids = tree_position_ids
                    
                    # 广播Draft Tokens结果 (最高优先级)
                    self.logger.info(f"  [PHASE 3][BROADCAST] Draft Tokens | shape={draft_tokens.shape}")
                    self.comm.send_draft_tokens(
                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
                        dst_rank=-1  # 广播
                    )
                    self.logger.info(f"  [PHASE 3] Draft Tokens 广播完成")
            
            phase3_time = (time.time() - phase3_start) * 1000
            self.logger.debug(f"  [PHASE 3] 完成 | 耗时: {phase3_time:.2f}ms")
            
            # 处理接收到的cache（非阻塞）
            self._process_received_caches_nonblocking(past_key_values, num_layers, chunk_idx)
            
            chunk_time = (time.time() - chunk_start_time) * 1000
            self.logger.info(f"[CHUNK {chunk_idx+1}/{len(chunks)}] 完成 | 总耗时: {chunk_time:.2f}ms")
        
        # =====================================================================
        # 所有chunks处理完成后的后续处理
        # =====================================================================
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[POST-CHUNK] 所有 Chunks 处理完成，开始同步阶段")
        self.logger.info("=" * 60)
        
        if not self.config.is_last_rank():
            # -----------------------------------------------------------------
            # 非最后rank：等待接收Draft Tree结果
            # -----------------------------------------------------------------
            self.logger.info(f"[RECV] 等待 Draft Tokens (from Rank {self.config.world_size - 1})...")
            recv_start = time.time()
            result = self.comm.recv_draft_tokens(
                src_rank=self.config.world_size - 1,
                timeout=60.0
            )
            if result is None:
                self.logger.error("[TIMEOUT] 接收 draft tokens 超时")
                raise RuntimeError("接收draft tokens超时")
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
            recv_time = (time.time() - recv_start) * 1000
            self.logger.info(f"[RECV] Draft Tokens 接收完成 | shape={draft_tokens.shape} | 耗时: {recv_time:.2f}ms")
            
            # 等待所有cache同步（base cache和eagle cache）
            self.logger.info("[SYNC] 等待所有 Cache 同步...")
            sync_start = time.time()
            eagle_layer = self.model.eagle_layer
            start_layer, end_layer = self.config.get_layer_range(num_layers)
            self.comm.wait_for_all_caches(
                past_key_values=past_key_values,
                eagle_layer=eagle_layer,
                num_layers=num_layers,
                num_chunks=self.num_chunks,
                cache_received_indicator=self.cache_received_indicator,
                eagle_cache_received_indicator=self.eagle_cache_received_indicator,
                start_layer=start_layer,
                end_layer=end_layer,
                is_last_rank=self.config.is_last_rank(),
                timeout=60.0
            )
            sync_time = (time.time() - sync_start) * 1000
            self.logger.info(f"[SYNC_OK] Rank {self.config.rank} Cache 同步完成 | 耗时: {sync_time:.2f}ms")
            
            # 清除prefill阶段标记
            self._is_prefill_phase = False
            
            self.timing_stats['prefill_compute_end'] = time.time()
            self.timing_stats['prefill_end'] = time.time()
            
            prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
            self.logger.info("=" * 60)
            self.logger.info(f"[PREFILL] 分布式 Prefill 完成 | 总耗时: {prefill_time:.3f}s")
            self.logger.info("=" * 60)
            
            # 非最后rank不需要返回值
            return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, None, None, None
        
        # -----------------------------------------------------------------
        # 最后一个rank：等待cache同步并返回结果
        # -----------------------------------------------------------------
        # 等待所有base model cache同步
        self.logger.info("[SYNC] 最后一个 Rank 等待所有 Cache 同步...")
        sync_start = time.time()
        eagle_layer = self.model.eagle_layer
        start_layer, end_layer = self.config.get_layer_range(num_layers)
        self.comm.wait_for_all_caches(
            past_key_values=past_key_values,
            eagle_layer=eagle_layer,
            num_layers=num_layers,
            num_chunks=self.num_chunks,
            cache_received_indicator=self.cache_received_indicator,
            eagle_cache_received_indicator=self.eagle_cache_received_indicator,
            start_layer=start_layer,
            end_layer=end_layer,
            is_last_rank=self.config.is_last_rank(),
            timeout=60.0
        )
        sync_time = (time.time() - sync_start) * 1000
        self.logger.info(f"[SYNC_OK] Rank {self.config.rank} Cache 同步完成 | 耗时: {sync_time:.2f}ms")
        
        # 清除prefill阶段标记
        self._is_prefill_phase = False
        
        self.timing_stats['prefill_compute_end'] = time.time()
        self.timing_stats['prefill_end'] = time.time()
        
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        self.logger.info("=" * 60)
        self.logger.info(f"[PREFILL] 分布式 Prefill 完成 | 总耗时: {prefill_time:.3f}s")
        self.logger.info("=" * 60)
        
        return (
            final_draft_tokens,
            final_retrieve_indices,
            final_tree_mask,
            final_tree_position_ids,
            final_orig,
            final_hidden_states,
            final_token,
        )
    
    def _split_into_chunks(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        将输入序列切分为多个chunks
        
        Args:
            input_ids: 输入token IDs [1, seq_len]
            
        Returns:
            chunks列表
        """
        chunk_size = self.config.chunk_size
        seq_length = input_ids.shape[1]
        
        chunks = []
        for start_idx in range(0, seq_length, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_length)
            chunk = input_ids[:, start_idx:end_idx]
            chunks.append(chunk)
        
        self.logger.debug(f"[SPLIT] 输入序列切分为 {len(chunks)} 个 chunks | seq_len={seq_length}, chunk_size={chunk_size}")
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"    Chunk {i}: shape={chunk.shape}")
        
        return chunks
    
    def _process_received_caches_nonblocking(
        self,
        past_key_values: List,
        num_layers: int,
        current_chunk_idx: int
    ):
        """
        非阻塞方式处理接收到的cache
        
        Args:
            past_key_values: KV Cache列表
            num_layers: 模型层数
            current_chunk_idx: 当前chunk索引
        """
        start_layer, end_layer = self.config.get_layer_range(num_layers)
        received_count = 0
        
        # 尝试接收base cache（非阻塞）
        while True:
            result = self.comm.get_received_base_cache(timeout=0.001)
            if result is None:
                break
            
            kv_cache, src_rank, layer_idx, chunk_idx = result
            key, value = kv_cache
            
            # 只处理非本rank负责的层
            if not (start_layer <= layer_idx < end_layer):
                if self.cache_received_indicator[chunk_idx, layer_idx] == 0:
                    past_key_values[layer_idx][0].cat(key)
                    past_key_values[layer_idx][1].cat(value)
                    self.cache_received_indicator[chunk_idx, layer_idx] = 1
                    received_count += 1
                    cache_size_mb = (key.numel() + value.numel()) * key.element_size() / 1024 / 1024
                    self.logger.debug(f"  [RECV][CACHE] layer={layer_idx}, chunk={chunk_idx} from Rank {src_rank} | size={cache_size_mb:.2f}MB")
        
        if received_count > 0:
            self.logger.debug(f"  [CACHE] 非阻塞接收 {received_count} 个 Base Cache")
    
    def _get_eagle_input_layers(self, num_layers: int) -> List[int]:
        """
        获取需要收集的eagle input hidden states层索引
        
        根据modeling_qwen的逻辑：
        - layer_idx == 2
        - layer_idx == num_layers // 2
        - layer_idx == num_layers - 3
        """
        layers = [2 - 1, num_layers // 2 - 1, num_layers - 3 - 1]
        self.logger.debug(f"[EAGLE] Input Layers 配置: {layers} (总层数={num_layers})")
        return layers
    
    def _cat_eagle_hidden_states(
        self,
        eagle_hidden_states: Dict[int, torch.Tensor],
        num_layers: int
    ) -> torch.Tensor:
        """
        从收集的hidden states准备Eagle Layer的输入
        
        Eagle3需要拼接层2、num_layers//2、num_layers-3的hidden states
        
        Args:
            eagle_hidden_states: 收集的hidden states字典
            num_layers: 模型层数
            
        Returns:
            拼接后的hidden states
            
        Raises:
            RuntimeError: 如果缺少必要的hidden state
        """
        ea_device = self.model.eagle_layer.lm_head.weight.device
        
        eagle_input_layers = self._get_eagle_input_layers(num_layers)
        
        collected = []
        for layer_idx in eagle_input_layers:
            if layer_idx not in eagle_hidden_states:
                self.logger.error(f"[EAGLE] 缺少 layer {layer_idx} 的 hidden state")
                raise RuntimeError(
                    f"Eagle input hidden state for layer {layer_idx} not found. "
                    f"Expected layers: {eagle_input_layers}, "
                    f"Available layers: {list(eagle_hidden_states.keys())}"
                )
            h = eagle_hidden_states[layer_idx]
            if h.device != ea_device:
                self.logger.debug(f"    [EAGLE] 移动 layer {layer_idx} hidden state 到设备 {ea_device}")
                h = h.to(ea_device)
            collected.append(h)
            self.logger.debug(f"    [EAGLE] 收集 layer {layer_idx} | {get_tensor_info(h)}")
        
        # 拼接成Eagle3所需的格式
        hidden_states = torch.cat(collected, dim=-1)
        self.logger.debug(f"  [EAGLE] 拼接后 hidden_states | {get_tensor_info(hidden_states)}")
        return hidden_states

    # =========================================================================
    # Cache同步方法
    # =========================================================================
    
    def _init_cache_tracking(self, num_layers: int, seq_length: int):
        """
        初始化cache接收状态追踪
        
        Args:
            num_layers: 模型层数
            seq_length: 输入序列长度
        """
        # 根据chunk_size和seq_length计算chunk数量
        chunk_size = self.config.chunk_size
        self.num_chunks = (seq_length + chunk_size - 1) // chunk_size  # 向上取整
        
        # Base Cache接收状态 [num_chunks, num_layers]
        self.cache_received_indicator = torch.zeros(
            (self.num_chunks, num_layers),
            dtype=torch.int8,
            device='cpu'
        )
        
        # Eagle Cache接收状态 [num_chunks] - 每个chunk同步一次
        self.eagle_cache_received_indicator = torch.zeros(
            self.num_chunks,
            dtype=torch.int8,
            device='cpu'
        )
        
        self.logger.info(f"[INIT] Cache 追踪初始化: chunks={self.num_chunks}, layers={num_layers}, seq_len={seq_length}")

    # =========================================================================
    # 统计信息
    # =========================================================================
    
    def get_timing_stats(self) -> dict:
        """获取时间统计"""
        stats = {
            'prefill_time': self.timing_stats['prefill_end'] - self.timing_stats['prefill_start'],
            'compute_time': self.timing_stats['prefill_compute_end'] - self.timing_stats['prefill_start'],
            'cache_sync_time': self.timing_stats['cache_sync_time'],
        }
        self.logger.debug(f"[STATS] 时间统计: {stats}")
        return stats
    
    def get_comm_stats(self) -> dict:
        """获取通信统计"""
        if self.comm is not None:
            stats = self.comm.get_stats()
            self.logger.debug(f"[STATS] 通信统计: {stats}")
            return stats
        return {}
