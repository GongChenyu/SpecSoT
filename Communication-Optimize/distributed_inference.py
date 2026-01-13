"""
SP+PP分布式推理主脚本
支持三台设备的Sequence Parallel + Pipeline Parallel推理
Prefill阶段：SP(chunk_size=128) + PP(模型层均分)
Decode阶段：全量冗余计算
"""

import os
import time
import argparse
import torch
import torch.distributed as dist
from typing import List, Optional, Tuple
from transformers import AutoTokenizer
from modeling_qwen3_kv_distributed import Qwen3ForCausalLMDistributed
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DistributedInferenceEngine:
    def __init__(
        self,
        model_path: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        chunk_size: int = 128,
        sync_strategy: str = "pairwise",  # "pairwise" or "ring"
        device_mode: str = "single_node",  # "single_node" or "multi_node"
        backend: str = "auto",  # "nccl", "gloo", or "auto"
    ):
        """
        初始化分布式推理引擎
        
        Args:
            model_path: 模型路径
            rank: 当前设备rank (0, 1, 2)
            world_size: 总设备数 (3)
            master_addr: 主节点地址
            master_port: 主节点端口
            chunk_size: SP的chunk大小
            sync_strategy: cache同步策略 ("pairwise"或"ring")
            device_mode: 设备模式 ("single_node"单机多卡 或 "multi_node"多机单卡)
            backend: 通信后端 ("nccl", "gloo", 或 "auto"自动选择)
        """
        self.rank = rank
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.sync_strategy = sync_strategy
        self.model_path = model_path
        self.device_mode = device_mode
        self.backend_preference = backend

        # 初始化logger
        self.logger = self._setup_logger()
        
        # 设置本地设备
        self.local_device = self._get_local_device()
        
        # 初始化分布式环境
        self._init_distributed(master_addr, master_port)
        
        # 加载模型和tokenizer
        self.logger.info(f"初始化设备 Rank {rank}/{world_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._load_model()
        
        # 时间测量点
        self.timing_stats = {
            'prefill_start': 0,
            'prefill_end': 0,
            'cache_sync_end': 0,
            'decode_start': 0,
            'decode_end': 0
        }
        
        # 🆕 Cache接收状态追踪矩阵: [num_chunks, num_pp_stages]
        # 初始化时不知道chunk数量，在prefill阶段动态创建
        self.cache_received_indicator = None
        self.num_chunks = 0
        self.num_pp_stages = world_size  # PP stage数量等于world_size
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger(f"Rank-{self.rank}")
        logger.setLevel(logging.INFO)
        return logger
    
    def _get_local_device(self) -> int:
        """
        根据设备模式获取本地CUDA设备ID
        """
        # 检查是否设置了 CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            # 如果环境变量限制了可见显卡（例如 Bash 脚本中设置了），
            # 那么无论物理ID是多少，当前进程内只能看到被映射为 0 的设备。
            # 这种情况下，Rank 1 的进程看到的也是 cuda:0
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            self.logger.info(f"[Rank {self.rank}] 检测到 CUDA_VISIBLE_DEVICES={visible_devices}，使用逻辑设备 cuda:0")
            return 0
            
        if self.device_mode == 'multi_node':
            # 多机单卡模式：每台机器使用GPU 0
            self.logger.info(f"[Rank {self.rank}] 多机单卡模式，使用物理 GPU 0（逻辑 cuda:0）")
            return 0
        else:  # single_node
            # 如果没有设置 CUDA_VISIBLE_DEVICES（比如直接 python script.py 启动），
            # 那么进程能看到所有卡，此时需要用 rank 来指定具体用哪张卡。
            self.logger.info(f"[Rank {self.rank}] 单机多卡模式(无环境隔离)，使用GPU {self.rank}")
            return self.rank
        
    def _init_distributed(self, master_addr: str, master_port: str):
        """初始化分布式环境"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        # 选择通信后端
        if self.backend_preference == 'auto':
            # 自动选择
            # 单机多卡：使用nccl（更快，支持CUDA tensors直接通信）
            # 多机环境（尤其是无线连接的Jetson）：使用gloo（更稳定，支持TCP，但只能发送CPU tensors）
            if self.device_mode == 'single_node':
                backend = 'nccl'
                self.logger.info(f"自动选择：单机多卡环境，使用NCCL后端")
            else:
                backend = 'gloo'
                self.logger.info(f"自动选择：多机环境，使用Gloo后端（适合TCP/无线网络）")
        else:
            # 使用用户指定的后端
            backend = self.backend_preference
            self.logger.info(f"使用指定后端: {backend}")
        
        # 保存实际使用的backend
        self.backend = backend
        
        # 初始化进程组
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=self.rank,
            world_size=self.world_size
        )
        
        # 设置当前设备（使用检测到的本地设备）
        torch.cuda.set_device(self.local_device)
        
    def _load_model(self):
        """加载模型（全量模型）"""
        self.logger.info(f"加载模型: {self.model_path}")
        self.logger.info(f"注意：所有设备都加载完整模型，Prefill阶段选择对应层计算")
        
        model = Qwen3ForCausalLMDistributed.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{self.local_device}",
            rank=self.rank,
            world_size=self.world_size,
            sync_strategy=self.sync_strategy,
            backend=self.backend
        )
        model.eval()        
        self.logger.info(f"模型加载完成，共{model.config.num_hidden_layers}层")
        
        # 🆕 初始化预分配的KV Cache（连续内存）
        max_seq_length = 2200  # 可以根据需要调整
        model.model.initialize_kv_cache(max_length=max_seq_length, batch_size=1)
        self.logger.info(f"KV Cache 预分配完成（最大长度: {max_seq_length}）")
        
        return model
        
    def _split_prompt_chunks(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """将prompt按chunk_size切分"""
        seq_len = input_ids.shape[1]
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = input_ids[:, i:i+self.chunk_size]
            chunks.append(chunk)
        return chunks
        
    def _get_layer_range(self, num_layers: int) -> Tuple[int, int]:
        """获取当前rank负责的层范围"""
        layers_per_device = num_layers // self.world_size
        start_layer = self.rank * layers_per_device
        end_layer = start_layer + layers_per_device if self.rank < self.world_size - 1 else num_layers
        return start_layer, end_layer
        
    def prefill_phase(self, prompt: str) -> Tuple[torch.Tensor, List]:
        """
        Prefill阶段：使用SP+PP
        
        Returns:
            last_hidden_state: 最后一层的隐藏状态
            kv_caches: 所有层的KV cache
        """
        self.logger.info("=" * 60)
        self.logger.info("开始 Prefill 阶段 (SP+PP)")
        self.timing_stats['prefill_start'] = time.time()
        
        # 🆕 重置KV Cache缓冲区
        self.model.model.reset_kv_cache()
        self.logger.info("KV Cache 已重置")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(f"cuda:{self.local_device}")
        input_ids = inputs['input_ids']
        
        self.logger.info(f"Prompt长度: {input_ids.shape[1]} tokens")
        
        # 切分chunks
        chunks = self._split_prompt_chunks(input_ids)
        self.logger.info(f"切分为 {len(chunks)} 个chunks (chunk_size={self.chunk_size})")
        
        # 🆕 初始化Cache接收状态追踪矩阵: [num_chunks × num_layers]
        self.num_chunks = len(chunks)
        num_layers = self.model.config.num_hidden_layers
        self.cache_received_indicator = torch.zeros(
            (self.num_chunks, num_layers),
            dtype=torch.int8,
            device='cpu'
        )
        self.logger.info(f"初始化Cache接收状态矩阵: [{self.num_chunks} chunks × {num_layers} layers]")
        
        # 获取当前rank负责的层范围
        start_layer, end_layer = self._get_layer_range(num_layers)
        self.logger.info(f"负责层范围: [{start_layer}, {end_layer})")
        
        # 设置模型的PP范围
        self.model.set_pipeline_range(start_layer, end_layer)
        
        # 🆕 使用预分配的KV Cache（已在_load_model中初始化）
        # 获取模型的KV cache对象
        kv_cache_list = self.model.model.past_key_values
        self.logger.info(f"使用预分配的KV Cache，共{len(kv_cache_list)}层")
        
        # 逐chunk、逐层处理
        last_hidden = None
        
        for chunk_idx, chunk in enumerate(chunks):
            self.logger.info(f"处理 chunk {chunk_idx+1}/{len(chunks)}")
            
            # 第一个rank从embedding开始
            if self.rank == 0:
                hidden_states = self.model.model.embed_tokens(chunk)
                self.logger.debug(f"  Rank 0: 生成embedding, shape={hidden_states.shape}")
                batch_size, seq_length = chunk.shape
            else:
                # 其他rank稍后在层循环中接收hidden states
                hidden_states = None
                batch_size, seq_length = 1, chunk.shape[1]   # 这里不够鲁棒
            
            # 计算past_key_values_length
            past_key_values_length = 0
            if kv_cache_list[0][0].current_length > 0:
                past_key_values_length = kv_cache_list[0][0].current_length.item()
            
            seq_length_with_past = seq_length + past_key_values_length
            
            # 🆕 遍历所有36层：当前rank计算自己负责的层，所有rank参与cache同步
            for layer_idx in range(num_layers):
                # 🔄 在当前rank的第一层之前，接收上一个rank的hidden states
                if layer_idx == start_layer and self.rank > 0:
                    hidden_states = self._receive_hidden_states()
                    batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
                    seq_length_with_past = seq_length + past_key_values_length
                    self.logger.debug(f"  Rank {self.rank}: 在layer {layer_idx}前接收hidden states, shape={hidden_states.shape}")
                
                # 判断当前rank是否负责这一层
                if start_layer <= layer_idx < end_layer:
                    # 我负责这一层，进行计算
                    self.logger.debug(f"  计算层 {layer_idx}")
                    
                    # 准备attention mask和position ids（在第一层时）
                    if layer_idx == start_layer:
                        position_ids = torch.arange(
                            past_key_values_length,
                            seq_length + past_key_values_length,
                            dtype=torch.long,
                            device=f"cuda:{self.local_device}",
                        ).unsqueeze(0)
                        
                        attention_mask = torch.ones(
                            (batch_size, seq_length_with_past),
                            dtype=torch.bool,
                            device=f"cuda:{self.local_device}",
                        )
                        
                        attention_mask = self.model.model._prepare_decoder_attention_mask(
                            attention_mask,
                            (batch_size, seq_length),
                            hidden_states,
                            past_key_values_length,
                        )
                        
                        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
                    
                    # 获取该层的past_key_value（使用KVCache对象）
                    layer_past_kv = kv_cache_list[layer_idx] if past_key_values_length > 0 else None
                    
                    # 单层forward
                    layer_output = self.model.forward_single_layer(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        past_key_value=layer_past_kv,
                        use_cache=True
                    )
                    
                    hidden_states = layer_output['hidden_states']
                    new_kv_cache = layer_output['past_key_value']
                else:
                    # 我不负责这一层，跳过计算
                    new_kv_cache = None
                
                # 🆕 所有rank参与该层cache的广播同步，并使用KVCache.append_sequence()追加
                self._sync_and_append_cache(layer_idx, new_kv_cache, chunk_idx, kv_cache_list)
                
                # 🆕 标记该层cache已接收
                self.cache_received_indicator[chunk_idx, layer_idx] = 1
                
                # 🔄 在当前rank的最后一层之后，发送hidden states给下一个rank
                if layer_idx == end_layer - 1 and self.rank < self.world_size - 1:
                    self._send_hidden_states(hidden_states)
                    self.logger.debug(f"  Rank {self.rank}: 在layer {layer_idx}后发送hidden states到 Rank {self.rank+1}")
            
            # 当前chunk在当前rank的所有层计算完成
            # 最后一个rank过norm
            if self.rank == self.world_size - 1:
                hidden_states = self.model.model.norm(hidden_states)
                self.logger.debug(f"  Rank {self.world_size-1}: 应用norm")
            
            last_hidden = hidden_states  # 保存最后的hidden_states
            
            self.logger.info(f"  chunk {chunk_idx+1} 处理完成")
        
        self.timing_stats['prefill_end'] = time.time()
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        self.logger.info(f"Prefill 完成，耗时: {prefill_time:.3f}s")
        
        # 🆕 打印Cache接收状态矩阵
        self._log_cache_received_status()
        
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception as e:
            self.logger.error(f"同步失败: {e}")
            # 刷新日志
            for handler in self.logger.handlers:
                handler.flush()
            raise
        
        self.timing_stats['cache_sync_end'] = self.timing_stats['prefill_end']
        self.logger.info(f"Cache 已在计算过程中逐层同步完成")
        
        # 返回所有层的cache（decode阶段需要完整的36层cache）
        # 直接返回KVCache对象列表
        self.logger.info(f"Prefill完成，返回{len(kv_cache_list)}层KVCache对象")
        if kv_cache_list[0][0].current_length > 0:
            cache_len = kv_cache_list[0][0].current_length.item()
            self.logger.info(f"Cache示例 - Layer 0: current_length={cache_len}, shape={kv_cache_list[0][0].shape}")
        
        return last_hidden, kv_cache_list
        
    def decode_phase(self, last_hidden: torch.Tensor, kv_caches: List, max_new_tokens: int = 100) -> str:
        """
        Decode阶段：所有设备进行相同的全量计算
        
        Args:
            kv_caches: Prefill阶段生成的KV cache
            max_new_tokens: 最大生成token数
            
        Returns:
            generated_text: 生成的文本
        """
        # 等待prefill和cache同步都完成
        try:
            dist.barrier()
        except Exception as e:
            self.logger.error(f"Decode 阶段同步失败: {e}")
            for handler in self.logger.handlers:
                handler.flush()
            raise
        
        self.logger.info("=" * 60)
        self.logger.info("开始 Decode 阶段 (全量冗余计算)")
        self.timing_stats['decode_start'] = time.time()
        
        # 重置模型为全量模式
        self.model.set_full_model_mode()
        
        # 获取上一个token (从prefill的最后输出)
        # 从kv_cache推断当前位置（使用KVCache的current_length）
        current_position = kv_caches[0][0].current_length.item()
        self.logger.info(f"Decode开始时的cache长度: {current_position}")
        
        # 生成初始token - 只有最后一个rank计算，然后广播给所有rank
        if self.rank == self.world_size - 1:
            # 最后一个设备有完整的hidden states，从last_hidden生成第一个token
            # 需要过lm_head得到logits
            with torch.no_grad():
                logits = self.model.lm_head(last_hidden[:, -1:, :])  # [batch, 1, vocab_size]
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [batch, 1]
        else:
            next_token_id = None
        
        # 广播初始token到所有rank
        next_token_id = self._broadcast_next_token(next_token_id)
        self.logger.debug(f"初始token: {next_token_id.item()}")
        
        generated_tokens = [next_token_id.item()]
        
        # 自回归生成
        for step in range(max_new_tokens):
            # 所有设备执行相同的forward
            # 将KVCache对象转换为模型期望的格式（tuple of tensors）
            past_kv_tuples = []
            for layer_kv in kv_caches:
                key_cache = layer_kv[0].get_data()
                value_cache = layer_kv[1].get_data()
                past_kv_tuples.append((key_cache, value_cache))
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=past_kv_tuples,
                    use_cache=True
                )
            
            # 更新KVCache：将模型输出的新cache追加到KVCache对象中
            new_past_kv = outputs.past_key_values
            for layer_idx in range(len(kv_caches)):
                new_key, new_value = new_past_kv[layer_idx]
                # 只追加新生成的部分（最后一个token的cache）
                new_key_slice = new_key[:, :, -1:, :]  # [batch, heads, 1, dim]
                new_value_slice = new_value[:, :, -1:, :]
                kv_caches[layer_idx][0].cat(new_key_slice, dim=2)
                kv_caches[layer_idx][1].cat(new_value_slice, dim=2)
            logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            
            # 简单的greedy decoding
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]
            
            generated_tokens.append(next_token_id.item())
            
            # 每5步打印cache长度以调试
            if (step + 1) % 5 == 0:
                cache_len = kv_caches[0][0].current_length.item()
                self.logger.info(f"Step {step+1}: cache长度={cache_len}, 最新token={next_token_id.item()}")
            
            # 检查是否生成了结束token
            if next_token_id.item() == self.tokenizer.eos_token_id:
                self.logger.info(f"遇到EOS token，停止生成 (step {step+1})")
                break
                
            if (step + 1) % 10 == 0:
                self.logger.info(f"已生成 {step+1} tokens")
        
        self.timing_stats['decode_end'] = time.time()
        decode_time = self.timing_stats['decode_end'] - self.timing_stats['decode_start']
        self.logger.info(f"Decode 完成，耗时: {decode_time:.3f}s")
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
        
    def _send_hidden_states(self, hidden_states: torch.Tensor):
        """发送hidden states到下一个设备"""
        # Gloo后端需要在CPU上通信
        if self.backend == 'gloo':
            # 先发送shape信息（在CPU上）
            shape = torch.tensor(hidden_states.shape, dtype=torch.long, device='cpu')
            dist.send(shape, dst=self.rank + 1)
            # 将数据移到CPU再发送
            hidden_states_cpu = hidden_states.contiguous().cpu()
            dist.send(hidden_states_cpu, dst=self.rank + 1)
        else:
            # NCCL后端可以直接在GPU上通信
            shape = torch.tensor(list(hidden_states.shape), dtype=torch.long, device=f"cuda:{self.local_device}")
            dist.send(shape, dst=self.rank + 1)
            dist.send(hidden_states.contiguous(), dst=self.rank + 1)
        
    def _receive_hidden_states(self) -> torch.Tensor:
        """从上一个设备接收hidden states"""
        # Gloo后端需要在CPU上通信
        if self.backend == 'gloo':
            # 先接收shape信息（在CPU上）
            shape = torch.zeros(3, dtype=torch.long, device='cpu')
            dist.recv(shape, src=self.rank - 1)
            self.logger.debug(f"  接收到shape: {shape.tolist()}")
            
            # 在CPU上创建tensor并接收数据
            hidden_states_cpu = torch.zeros(
                tuple(shape.tolist()),
                dtype=torch.float16
            )
            dist.recv(hidden_states_cpu, src=self.rank - 1)
            # 移到GPU
            hidden_states = hidden_states_cpu.to(f"cuda:{self.local_device}")
            self.logger.debug(f"  成功接收hidden states并移到GPU")
        else:
            # NCCL后端可以直接在GPU上通信
            shape = torch.zeros(3, dtype=torch.long, device=f"cuda:{self.local_device}")
            dist.recv(shape, src=self.rank - 1)
            self.logger.debug(f"  接收到shape: {shape.tolist()}")
            
            hidden_states = torch.zeros(
                tuple(shape.tolist()),
                dtype=torch.float16,
                device=f"cuda:{self.local_device}"
            )
            dist.recv(hidden_states, src=self.rank - 1)
            self.logger.debug(f"  成功接收hidden states")
        
        return hidden_states
    
    def _sync_and_append_cache(self, layer_idx: int, local_cache, chunk_idx: int, kv_cache_list):
        """
        同步并追加cache到KVCache对象
        
        Args:
            layer_idx: 层索引
            local_cache: 当前rank计算的cache (KVCache对象或None)
            chunk_idx: chunk索引
            kv_cache_list: KVCache对象列表
        """
        # 确定哪个rank负责这一层
        num_layers = self.model.config.num_hidden_layers
        layers_per_rank = num_layers // self.world_size
        owner_rank = layer_idx // layers_per_rank
        if owner_rank >= self.world_size:
            owner_rank = self.world_size - 1
        
        if self.rank == owner_rank:
            # 我是owner，广播cache
            if local_cache is None:
                raise RuntimeError(f"Rank {self.rank} 负责 layer {layer_idx}，但 local_cache 为 None！")
            
            # 从KVCache对象或tuple中提取key和value
            if isinstance(local_cache, (list, tuple)) and len(local_cache) == 2:
                if hasattr(local_cache[0], 'get_data'):   # 这里的判断始终是False
                    # KVCache对象
                    key = local_cache[0].get_data()
                    value = local_cache[1].get_data()
                else:
                    # tuple
                    key, value = local_cache
            else:
                raise TypeError(f"Unexpected local_cache type: {type(local_cache)}")
            
            # 广播shape
            shape_info = torch.tensor(key.shape, dtype=torch.long, device='cpu')
            # shape_info = torch.tensor([key.shape[2]], dtype=torch.long, device='cpu')
            for dest_rank in range(self.world_size):
                if dest_rank != self.rank:
                    dist.send(shape_info, dst=dest_rank)
            
            # 广播cache数据
            for dest_rank in range(self.world_size):
                if dest_rank != self.rank:
                    if self.backend == 'gloo':
                        dist.send(key.cpu().contiguous(), dst=dest_rank)
                        dist.send(value.cpu().contiguous(), dst=dest_rank)
                    else:
                        dist.send(key.contiguous(), dst=dest_rank)
                        dist.send(value.contiguous(), dst=dest_rank)
            
            # 追加到自己的KVCache
            kv_cache_list[layer_idx][0].cat(key)
            kv_cache_list[layer_idx][1].cat(value)
            
        else:
            # 我不是owner，接收cache
            shape_info = torch.zeros(4, dtype=torch.long, device='cpu')
            dist.recv(shape_info, src=owner_rank)
            kv_shape = tuple(shape_info.tolist())
            
            if self.backend == 'gloo':
                recv_key = torch.zeros(kv_shape, dtype=torch.float16, device='cpu')
                recv_value = torch.zeros(kv_shape, dtype=torch.float16, device='cpu')
                dist.recv(recv_key, src=owner_rank)
                dist.recv(recv_value, src=owner_rank)
                recv_key = recv_key.to(f"cuda:{self.local_device}")
                recv_value = recv_value.to(f"cuda:{self.local_device}")
            else:
                recv_key = torch.zeros(kv_shape, dtype=torch.float16, device=f"cuda:{self.local_device}")
                recv_value = torch.zeros(kv_shape, dtype=torch.float16, device=f"cuda:{self.local_device}")
                dist.recv(recv_key, src=owner_rank)
                dist.recv(recv_value, src=owner_rank)
            
            # 追加到KVCache    
            kv_cache_list[layer_idx][0].cat(recv_key)
            kv_cache_list[layer_idx][1].cat(recv_value)
    
    def _broadcast_layer_cache(self, layer_idx: int, local_cache: Tuple[torch.Tensor, torch.Tensor], chunk_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        广播某一层的cache到所有rank
        
        逻辑：
        - 确定哪个rank负责该layer（owner_rank）
        - owner_rank将local_cache广播给所有其他rank
        - 所有rank返回相同的cache
        
        Args:
            layer_idx: 层索引
            local_cache: 当前rank计算的该层cache (key, value)，非owner rank传入dummy
            chunk_idx: 当前chunk索引
            
        Returns:
            synced_cache: 同步后的cache（所有rank相同）
        """
        # 确定哪个rank负责这一层
        num_layers = self.model.config.num_hidden_layers
        layers_per_rank = num_layers // self.world_size
        owner_rank = layer_idx // layers_per_rank
        if owner_rank >= self.world_size:
            owner_rank = self.world_size - 1
        
        if self.rank == owner_rank:
            # 我是owner，先广播shape，再广播cache数据
            if local_cache is None:
                raise RuntimeError(f"Rank {self.rank} 负责 layer {layer_idx}，但 local_cache 为 None！")
            
            key, value = local_cache
            
            # 广播shape信息到所有其他rank
            shape_info = torch.tensor(key.shape, dtype=torch.long, device='cpu')
            for dest_rank in range(self.world_size):
                if dest_rank != self.rank:
                    dist.send(shape_info, dst=dest_rank)
            
            self.logger.debug(f"    Layer {layer_idx} Chunk {chunk_idx}: 广播cache (shape={key.shape})")
            
            # 广播cache数据
            for dest_rank in range(self.world_size):
                if dest_rank != self.rank:
                    if self.backend == 'gloo':
                        dist.send(key.cpu().contiguous(), dst=dest_rank)
                        dist.send(value.cpu().contiguous(), dst=dest_rank)
                    else:
                        dist.send(key.contiguous(), dst=dest_rank)
                        dist.send(value.contiguous(), dst=dest_rank)
            
            return (key, value)
        else:
            # 我不是owner，先接收shape，再接收cache数据
            # 接收shape信息
            shape_info = torch.zeros(4, dtype=torch.long, device='cpu')
            dist.recv(shape_info, src=owner_rank)
            
            # 准备接收缓冲区
            # batch_size = shape_info[1].item()
            # seq_len = shape_info[2].item()
            # num_heads = shape_info[3].item()
            # head_dim = shape_info[4].item()
            shape_info = tuple(shape_info.tolist())
            if self.backend == 'gloo':
                recv_key = torch.zeros(shape_info, dtype=torch.float16, device='cpu')
                recv_value = torch.zeros(shape_info, dtype=torch.float16, device='cpu')
                dist.recv(recv_key, src=owner_rank)
                dist.recv(recv_value, src=owner_rank)
                # 移到GPU
                recv_key = recv_key.to(f"cuda:{self.local_device}")
                recv_value = recv_value.to(f"cuda:{self.local_device}")
            else:
                recv_key = torch.zeros(shape_info, dtype=torch.float16, device=f"cuda:{self.local_device}")
                recv_value = torch.zeros(shape_info, dtype=torch.float16, device=f"cuda:{self.local_device}")
                # recv_key = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=f"cuda:{self.local_device}")
                # recv_value = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=f"cuda:{self.local_device}")
                dist.recv(recv_key, src=owner_rank)
                dist.recv(recv_value, src=owner_rank)
            
            self.logger.debug(f"    Layer {layer_idx} Chunk {chunk_idx}: 从Rank {owner_rank}接收cache (shape={recv_key.shape})")
            return (recv_key, recv_value)
    
    def _broadcast_next_token(self, token_id: torch.Tensor = None) -> torch.Tensor:
        """
        广播next token到所有rank，确保decode阶段所有设备使用相同的token
        
        Args:
            token_id: 当前rank生成的token_id，shape=[batch, 1]
                     只有最后一个rank需要传入有效值
        
        Returns:
            synced_token_id: 同步后的token，所有rank相同
        """
        if self.rank == self.world_size - 1:
            # 最后一个rank负责生成token并广播
            if token_id is None:
                raise RuntimeError("最后一个rank的token_id不能为None")
            # 确保在CPU上以便gloo后端使用
            if self.backend == 'gloo':
                token_cpu = token_id.cpu().contiguous()
                for dest_rank in range(self.world_size - 1):
                    dist.send(token_cpu, dst=dest_rank)
                return token_id
            else:
                # NCCL可以直接在GPU上广播
                token_gpu = token_id.contiguous()
                for dest_rank in range(self.world_size - 1):
                    dist.send(token_gpu, dst=dest_rank)
                return token_id
        else:
            # 其他rank接收token
            if self.backend == 'gloo':
                token_cpu = torch.zeros((1, 1), dtype=torch.long, device='cpu')
                dist.recv(token_cpu, src=self.world_size - 1)
                token_gpu = token_cpu.to(f"cuda:{self.local_device}")
                return token_gpu
            else:
                token_gpu = torch.zeros((1, 1), dtype=torch.long, device=f"cuda:{self.local_device}")
                dist.recv(token_gpu, src=self.world_size - 1)
                return token_gpu
        
    def print_timing_stats(self):
        """打印时间统计"""
        self.logger.info("=" * 60)
        self.logger.info("时间统计:")
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        cache_sync_time = self.timing_stats['cache_sync_end'] - self.timing_stats['prefill_end']
        decode_time = self.timing_stats['decode_end'] - self.timing_stats['decode_start']
        total_time = self.timing_stats['decode_end'] - self.timing_stats['prefill_start']
        
        self.logger.info(f"  Prefill 时间:      {prefill_time:.3f}s")
        self.logger.info(f"  Cache 同步时间:    {cache_sync_time:.3f}s")
        self.logger.info(f"  Decode 时间:       {decode_time:.3f}s")
        self.logger.info(f"  总时间:           {total_time:.3f}s")
        self.logger.info("=" * 60)
        
    def run_inference(self, prompt: str, max_new_tokens: int = 100) -> str:
        """运行完整的推理流程"""
        try:
            # Prefill阶段
            last_hidden, kv_caches = self.prefill_phase(prompt)
            
            # Decode阶段
            generated_text = self.decode_phase(last_hidden, kv_caches, max_new_tokens)
            
            # 打印统计信息
            self.print_timing_stats()
            
            if self.rank == 0:
                self.logger.info("=" * 60)
                self.logger.info("生成结果:")
                self.logger.info(generated_text)
                self.logger.info("=" * 60)
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"推理过程出错: {e}", exc_info=True)
            # 刷新日志缓冲区，确保错误信息被写入文件
            for handler in self.logger.handlers:
                handler.flush()
            # 通知其他 rank 发生错误（使用超时避免挂起）
            try:
                # 尝试一个快速的 barrier，如果其他 rank 也在错误处理中会快速失败
                dist.barrier()
            except Exception:
                pass  # 忽略 barrier 失败
            raise
        finally:
            self.cleanup()
    
    def _log_cache_received_status(self):
        """打印Cache接收状态矩阵（用于调试）"""
        if self.cache_received_indicator is None:
            return
        
        num_layers = self.cache_received_indicator.shape[1]
        
        self.logger.info("=" * 60)
        self.logger.info("Cache接收状态矩阵 (Chunk × Layer):")
        
        # 打印表头
        header = "Chunk".ljust(8) + " | " + " ".join([f"L{i}".ljust(2) for i in range(0, num_layers, 2)])
        self.logger.info(header)
        self.logger.info("-" * len(header))
        
        # 打印每一行
        for chunk_idx in range(self.num_chunks):
            row_str = f"Chunk{chunk_idx}".ljust(8) + " | "
            row_str += " ".join([
                f"{self.cache_received_indicator[chunk_idx, layer_idx].item()}" 
                for layer_idx in range(0, num_layers, 2)
            ])
            self.logger.info(row_str)
        
        # 统计信息
        total_cells = self.num_chunks * num_layers
        received_cells = self.cache_received_indicator.sum().item()
        self.logger.info(f"接收完成度: {received_cells}/{total_cells} ({100*received_cells/total_cells:.1f}%)")
        self.logger.info("=" * 60)
            
    def cleanup(self):
        """清理资源"""
        try:
            # 刷新所有日志
            for handler in self.logger.handlers:
                handler.flush()
            
            # 销毁进程组（可能会因为其他 rank 崩溃而失败）
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception as e:
                    self.logger.warning(f"销毁进程组时出错（可能其他 rank 已崩溃）: {e}")
        except Exception as e:
            # 确保 cleanup 不会抛出异常
            print(f"Rank {self.rank} cleanup 出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='SP+PP分布式推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--rank', type=int, required=True, help='当前设备rank (0, 1, 2)')
    parser.add_argument('--world_size', type=int, default=3, help='总设备数')
    parser.add_argument('--master_addr', type=str, default='localhost', help='主节点地址')
    parser.add_argument('--master_port', type=str, default='29500', help='主节点端口')
    parser.add_argument('--chunk_size', type=int, default=128, help='SP的chunk大小')
    parser.add_argument('--sync_strategy', type=str, default='pairwise', 
                       choices=['pairwise', 'ring'], help='Cache同步策略')
    parser.add_argument('--device_mode', type=str, default='single_node',
                       choices=['single_node', 'multi_node'], 
                       help='设备模式：single_node(单机多卡) 或 multi_node(多机单卡)')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'nccl', 'gloo'],
                       help='通信后端：auto(自动选择), nccl(GPU通信), gloo(CPU通信)')
    parser.add_argument('--prompt', type=str, default='请详细介绍一下人工智能的发展历史。', 
                       help='输入prompt')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='最大生成token数')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = DistributedInferenceEngine(
        model_path=args.model_path,
        rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        chunk_size=args.chunk_size,
        sync_strategy=args.sync_strategy,
        device_mode=args.device_mode,
        backend=args.backend
    )
    
    # 运行推理
    engine.run_inference(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
