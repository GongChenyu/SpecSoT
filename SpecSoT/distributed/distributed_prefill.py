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
"""

import os
import time
import logging
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessorList

from .distributed_config import DistributedConfig
from .comm_manager import create_zmq_comm_manager, ZMQCommManagerBase, MessageType, Message
from ..kv_cache import KVCache


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
        
        # 设置日志
        self.logger = logging.getLogger(f"DistPrefill-Rank{config.rank}")
        self.logger.setLevel(logging.INFO)
        
        # 通信管理器
        self.comm: Optional[ZMQCommManagerBase] = None
        
        # 时间统计
        self.timing_stats = {
            'prefill_start': 0,
            'prefill_compute_end': 0,
            'prefill_end': 0,
            'cache_sync_time': 0,
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
        self.logger.info(f"初始化ZMQ通信管理器 (mode={self.config.comm_mode})")
        
        self.comm = create_zmq_comm_manager(
            rank=self.config.rank,
            world_size=self.config.world_size,
            base_port=self.config.base_port,
            mode=self.config.comm_mode,
            device=self.device,
            node_addresses=self.config.node_addresses
        )
        
        # 等待其他节点
        self.logger.info(f"等待其他节点就绪 ({self.config.startup_delay}秒)...")
        time.sleep(self.config.startup_delay)
        
        # 启动通信
        self.comm.start()
        self.logger.info("通信管理器已启动")
    
    def cleanup(self):
        """清理资源"""
        if self.comm is not None:
            self.comm.stop()
            self.comm = None
        self.logger.info("分布式Prefill管理器已清理")
    
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
        
        self.logger.info(f"开始分布式Prefill, 负责层: [{start_layer}, {end_layer}), seq_len={total_seq_length}")
        
        # 初始化cache接收状态
        self._init_cache_tracking(num_layers, total_seq_length)
        
        # 切分chunks
        chunks = self._split_into_chunks(input_ids)
        self.logger.info(f"切分为 {len(chunks)} 个chunks (chunk_size={self.config.chunk_size})")
        
        # 确定需要传输给eagle layer的hidden state层
        eagle_input_layers = self._get_eagle_input_layers(num_layers)
        eagle_hidden_states = {}  # 收集eagle input hidden states
        
        # =====================================================================
        # 多Chunk循环处理
        # =====================================================================
        for chunk_idx, chunk in enumerate(chunks):
            is_first_chunk = (chunk_idx == 0)
            is_last_chunk = (chunk_idx == len(chunks) - 1)
            chunk_seq_length = chunk.shape[1]
            
            self._current_chunk_idx = chunk_idx
            self.logger.debug(f"处理 chunk {chunk_idx+1}/{len(chunks)}, seq_len={chunk_seq_length}")
            
            # =================================================================
            # Phase 1: Embedding / 接收hidden states
            # =================================================================
            if self.config.is_first_rank():
                # 第一个rank执行embedding
                hidden_states = base_model.model.embed_tokens(chunk)
                self.logger.debug(f"Rank 0: embedding完成, shape={hidden_states.shape}")
            else:
                # 接收上一个rank的hidden states（阻塞）
                result = self.comm.recv_hidden(
                    src_rank=self.config.get_prev_rank(), 
                    timeout=60.0
                )
                if result is None:
                    raise RuntimeError(f"Rank {self.config.rank}: 接收hidden states超时 (chunk={chunk_idx})")
                hidden_states, recv_chunk_idx = result
                self.logger.debug(f"接收hidden states, shape={hidden_states.shape}, chunk={recv_chunk_idx}")
            
            # =================================================================
            # Phase 2: Decoder Layers (各rank计算各自负责的层)
            # =================================================================
            # 计算past_key_values_length
            past_key_values_length = 0
            if past_key_values[0][0].current_length > 0:
                past_key_values_length = past_key_values[0][0].current_length.item()
            
            seq_length_with_past = chunk_seq_length + past_key_values_length
            
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
            for layer_idx in range(start_layer, end_layer):
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
                
                # 收集eagle input hidden states（每个chunk都要收集）
                if layer_idx in eagle_input_layers:
                    if self.config.is_last_rank():
                        # 最后一个rank直接保存
                        eagle_hidden_states[layer_idx] = hidden_states.clone()
                    else:
                        # 非最后rank需要发送给最后一个rank
                        self.comm.send_eagle_input_hidden(
                            hidden=hidden_states.clone(),
                            layer_idx=layer_idx,
                            dst_rank=self.config.world_size - 1
                        )
                        self.logger.debug(f"发送eagle input hidden (layer={layer_idx}, chunk={chunk_idx}) 到最后一个rank")
                
                # 获取新增的KV并发送给其他rank
                current_len = past_key_values[layer_idx][0].current_length.item()
                new_key = past_key_values[layer_idx][0].data[:, :, current_len - chunk_seq_length:current_len, :]
                new_value = past_key_values[layer_idx][1].data[:, :, current_len - chunk_seq_length:current_len, :]
                
                # 异步发送base cache给其他rank
                for other_rank in range(self.config.world_size):
                    if other_rank != self.config.rank:
                        self.comm.send_base_cache_async(
                            kv_cache=(new_key.clone(), new_value.clone()),
                            dst_rank=other_rank,
                            layer_idx=layer_idx,
                            chunk_idx=chunk_idx
                        )
                
                # 标记本层cache已发送
                self.cache_received_indicator[chunk_idx, layer_idx] = 1
            
            # =================================================================
            # Phase 3: 发送hidden states给下一个rank / 执行Eagle Layer
            # =================================================================
            if not self.config.is_last_rank():
                # 非最后rank：发送hidden states给下一个rank
                self.comm.send_hidden(
                    hidden_states, 
                    dst_rank=self.config.get_next_rank(), 
                    chunk_idx=chunk_idx
                )
                self.logger.debug(f"发送hidden states到 Rank {self.config.get_next_rank()} (chunk={chunk_idx})")
            else:
                # 最后一个rank：执行Eagle Layer计算（Pipeline的一部分）
                
                # Norm
                hidden_states_normed = base_model.model.norm(hidden_states)
                
                # LM Head
                orig = base_model.lm_head(hidden_states_normed)
                
                # 接收其他rank发送的eagle input hidden states（阻塞等待收集完毕）
                received_eagle_hidden = self.comm.recv_all_eagle_input_hidden(
                    eagle_input_layers=eagle_input_layers,
                    my_start_layer=start_layer,
                    my_end_layer=end_layer,
                    timeout=60.0
                )
                # 合并本地和接收的eagle hidden states
                eagle_hidden_states.update(received_eagle_hidden)
                
                # 准备Eagle Layer的hidden states
                if self.model.use_eagle3:
                    hidden_states_for_eagle = self._cat_eagle_hidden_states(
                        eagle_hidden_states, num_layers
                    )
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
                    self.logger.info(f"First token: {token.item()}")

                # Eagle Layer生成Draft Tree（使用分布式Prefill专用函数）
                # 非最后chunk只更新stable_kv，最后chunk才生成完整的draft tree
                start_idx = chunk_idx * self.config.chunk_size + 1
                end_idx = total_seq_length + 1 if is_last_chunk else start_idx + chunk_seq_length
                input_ids_this_chunk = input_ids[:, start_idx:end_idx]
                tree_result, incremental_kv = eagle_layer.generate_draft_tree_dist_prefill(
                    hidden_states_for_eagle, 
                    input_ids_this_chunk,  # 使用当前chunk的input_ids
                    is_last_chunk=is_last_chunk,
                    chunk_idx=chunk_idx
                )
                
                # 显式发送Eagle增量KV（每个chunk都执行，只有最后rank会真正发送）
                if self.config.is_last_rank() and incremental_kv is not None:
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
                    self.comm.send_draft_tokens(
                        draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
                        dst_rank=-1  # 广播
                    )
                    self.logger.info(f"Draft Tokens已广播")
            
            # 处理接收到的cache（非阻塞）
            self._process_received_caches_nonblocking(past_key_values, num_layers, chunk_idx)
            
            self.logger.debug(f"chunk {chunk_idx+1}/{len(chunks)} 处理完成")
        
        # =====================================================================
        # 所有chunks处理完成后的后续处理
        # =====================================================================
        
        if not self.config.is_last_rank():
            # -----------------------------------------------------------------
            # 非最后rank：等待接收Draft Tree结果
            # -----------------------------------------------------------------
            result = self.comm.recv_draft_tokens(
                src_rank=self.config.world_size - 1,
                timeout=6000.0
            )
            if result is None:
                raise RuntimeError("接收draft tokens超时")
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
            
            # 等待所有cache同步（base cache和eagle cache）
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
            
            # 清除prefill阶段标记
            self._is_prefill_phase = False
            
            self.timing_stats['prefill_compute_end'] = time.time()
            self.timing_stats['prefill_end'] = time.time()
            
            prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
            self.logger.info(f"分布式Prefill完成, 耗时: {prefill_time:.3f}s")
            
            # 非最后rank不需要返回值
            return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, None, None, None
        
        # -----------------------------------------------------------------
        # 最后一个rank：等待cache同步并返回结果
        # -----------------------------------------------------------------
        # 等待所有base model cache同步
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
        
        # 清除prefill阶段标记
        self._is_prefill_phase = False
        
        self.timing_stats['prefill_compute_end'] = time.time()
        self.timing_stats['prefill_end'] = time.time()
        
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        self.logger.info(f"分布式Prefill完成, 耗时: {prefill_time:.3f}s")
        
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
                    self.logger.debug(f"接收base cache: layer={layer_idx}, chunk={chunk_idx}")
    
    def _get_eagle_input_layers(self, num_layers: int) -> List[int]:
        """
        获取需要收集的eagle input hidden states层索引
        
        根据modeling_qwen的逻辑：
        - layer_idx == 2
        - layer_idx == num_layers // 2
        - layer_idx == num_layers - 3
        """
        return [2, num_layers // 2, num_layers - 3]
    
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
                raise RuntimeError(
                    f"Eagle input hidden state for layer {layer_idx} not found. "
                    f"Expected layers: {eagle_input_layers}, "
                    f"Available layers: {list(eagle_hidden_states.keys())}"
                )
            h = eagle_hidden_states[layer_idx]
            if h.device != ea_device:
                h = h.to(ea_device)
            collected.append(h)
        
        # 拼接成Eagle3所需的格式
        hidden_states = torch.cat(collected, dim=-1)
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
        
        self.logger.debug(f"初始化cache追踪: {self.num_chunks} chunks, {num_layers} layers")

    # =========================================================================
    # 统计信息
    # =========================================================================
    
    def get_timing_stats(self) -> dict:
        """获取时间统计"""
        return {
            'prefill_time': self.timing_stats['prefill_end'] - self.timing_stats['prefill_start'],
            'compute_time': self.timing_stats['prefill_compute_end'] - self.timing_stats['prefill_start'],
            'cache_sync_time': self.timing_stats['cache_sync_time'],
        }
    
    def get_comm_stats(self) -> dict:
        """获取通信统计"""
        if self.comm is not None:
            return self.comm.get_stats()
        return {}
