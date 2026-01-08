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
        """
        self.rank = rank
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.sync_strategy = sync_strategy
        self.model_path = model_path
        self.device_mode = device_mode
        
        # 设置本地设备
        self.local_device = self._get_local_device()
        
        # 初始化分布式环境
        self._init_distributed(master_addr, master_port)
        
        # 加载模型和tokenizer
        self.logger = self._setup_logger()
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
            self.logger_temp = logging.getLogger('init')
            self.logger_temp.info(f"[Rank {self.rank}] 检测到 CUDA_VISIBLE_DEVICES，使用逻辑设备 cuda:0")
            return 0
            
        if self.device_mode == 'multi_node':
            # 多机单卡模式：每台机器使用GPU 0
            return 0
        else:  # single_node
            # 如果没有设置 CUDA_VISIBLE_DEVICES（比如直接 python script.py 启动），
            # 那么进程能看到所有卡，此时需要用 rank 来指定具体用哪张卡。
            print(f"[Rank {self.rank}] 单机多卡模式(无环境隔离)，使用GPU {self.rank}")
            return self.rank
        
    def _init_distributed(self, master_addr: str, master_port: str):
        """初始化分布式环境"""
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        # 自动选择通信后端
        # 单机多卡：使用nccl（更快）
        # 多机环境（尤其是无线连接的Jetson）：使用gloo（更稳定，支持TCP）
        if self.device_mode == 'single_node':
            # 单机多卡模式，使用nccl
            backend = 'nccl'
            self.logger_temp = logging.getLogger('init')
            self.logger_temp.info(f"检测到单机多卡环境，使用NCCL后端")
        else:
            # 多机单卡模式，使用gloo（适合无线网络）
            backend = 'gloo'
            self.logger_temp = logging.getLogger('init')
            self.logger_temp.info(f"检测到多机环境，使用Gloo后端（适合TCP/无线网络）")
        
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
            sync_strategy=self.sync_strategy
        )
        model.eval()        
        self.logger.info(f"模型加载完成，共{model.config.num_hidden_layers}层")        
        self.logger.info(f"模型加载完成，共{model.config.num_hidden_layers}层")
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
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(f"cuda:{self.local_device}")
        input_ids = inputs['input_ids']
        
        self.logger.info(f"Prompt长度: {input_ids.shape[1]} tokens")
        
        # 切分chunks
        chunks = self._split_prompt_chunks(input_ids)
        self.logger.info(f"切分为 {len(chunks)} 个chunks (chunk_size={self.chunk_size})")
        
        # 获取当前rank负责的层范围
        num_layers = self.model.config.num_hidden_layers
        start_layer, end_layer = self._get_layer_range(num_layers)
        self.logger.info(f"负责层范围: [{start_layer}, {end_layer})")
        
        # 设置模型的PP范围
        self.model.set_pipeline_range(start_layer, end_layer)
        
        # 初始化全局KV cache存储 (所有36层)
        all_kv_caches = [None] * num_layers
        
        # 逐chunk、逐层处理
        last_hidden = None
        
        for chunk_idx, chunk in enumerate(chunks):
            self.logger.info(f"处理 chunk {chunk_idx+1}/{len(chunks)}")
            
            # 第一个rank从embedding开始
            if self.rank == 0:
                hidden_states = self.model.model.embed_tokens(chunk)
                self.logger.debug(f"  Rank 0: 生成embedding, shape={hidden_states.shape}")
            else:
                # 其他rank接收上一个rank的hidden states
                hidden_states = self._receive_hidden_states()
                self.logger.debug(f"  Rank {self.rank}: 接收hidden states, shape={hidden_states.shape}")
            
            # 准备attention mask和position ids
            batch_size, seq_length = chunk.shape if self.rank == 0 else (hidden_states.shape[0], hidden_states.shape[1])
            
            # 计算past_key_values_length
            past_key_values_length = 0
            if all_kv_caches[0] is not None:
                past_key_values_length = all_kv_caches[0][0].shape[2]
            
            seq_length_with_past = seq_length + past_key_values_length
            
            # 生成position_ids
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=f"cuda:{self.local_device}",
            ).unsqueeze(0)
            
            # 生成attention_mask
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
            
            # 生成position_embeddings (rotary embedding)
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            
            # 逐层计算并同步
            for layer_idx in range(start_layer, end_layer):
                self.logger.debug(f"  计算层 {layer_idx}")
                
                # 获取该层的past_key_value
                layer_past_kv = all_kv_caches[layer_idx]
                
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
                
                # 立即同步该层的cache
                if new_kv_cache is not None:
                    self.logger.debug(f"  同步层 {layer_idx} cache")
                    synced_cache = self.model.sync_single_layer_cache(layer_idx, new_kv_cache)
                    
                    # 如果是第一个chunk，直接赋值；否则拼接
                    if all_kv_caches[layer_idx] is None:
                        all_kv_caches[layer_idx] = synced_cache
                    else:
                        # 在sequence维度拼接 [batch, num_heads, seq_len, head_dim]
                        key_old, value_old = all_kv_caches[layer_idx]
                        key_new, value_new = synced_cache
                        all_kv_caches[layer_idx] = (
                            torch.cat([key_old, key_new], dim=2),
                            torch.cat([value_old, value_new], dim=2)
                        )
            
            # 最后一个rank过norm
            if self.rank == self.world_size - 1:
                hidden_states = self.model.model.norm(hidden_states)
                self.logger.debug(f"  Rank {self.world_size-1}: 应用norm")
            
            last_hidden = hidden_states
            
            # 如果不是最后一个设备，发送hidden states到下一个设备
            if self.rank < self.world_size - 1:
                self._send_hidden_states(last_hidden)
                self.logger.debug(f"  Rank {self.rank}: 发送hidden states到 Rank {self.rank+1}")
        
        self.timing_stats['prefill_end'] = time.time()
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        self.logger.info(f"Prefill 完成，耗时: {prefill_time:.3f}s")
        
        torch.cuda.synchronize()
        dist.barrier()
        
        self.timing_stats['cache_sync_end'] = self.timing_stats['prefill_end']
        self.logger.info(f"Cache 已在计算过程中逐层同步完成")
        
        # 只提取当前rank负责的层的cache
        rank_kv_caches = [all_kv_caches[i] for i in range(start_layer, end_layer) if all_kv_caches[i] is not None]
        
        return last_hidden, rank_kv_caches
        
    def decode_phase(self, kv_caches: List, max_new_tokens: int = 100) -> str:
        """
        Decode阶段：所有设备进行相同的全量计算
        
        Args:
            kv_caches: Prefill阶段生成的KV cache
            max_new_tokens: 最大生成token数
            
        Returns:
            generated_text: 生成的文本
        """
        # 等待prefill和cache同步都完成
        dist.barrier()
        
        self.logger.info("=" * 60)
        self.logger.info("开始 Decode 阶段 (全量冗余计算)")
        self.timing_stats['decode_start'] = time.time()
        
        # 重置模型为全量模式
        self.model.set_full_model_mode()
        
        # 获取上一个token (从prefill的最后输出)
        # 从kv_cache推断当前位置
        current_position = kv_caches[0][0].shape[2]  # [batch, num_heads, seq_len, head_dim]
        
        # 生成初始token (所有设备都从相同的logits开始)
        if self.rank == self.world_size - 1:
            # 最后一个设备有完整的hidden states
            # 这里简化处理，实际应该从prefill的输出获取
            next_token_id = torch.tensor([[151643]], device=f"cuda:{self.local_device}")  # 使用一个默认token
        else:
            next_token_id = torch.tensor([[151643]], device=f"cuda:{self.local_device}")
            
        # 广播初始token (确保所有设备一致)
        dist.broadcast(next_token_id, src=self.world_size-1)
        
        generated_tokens = [next_token_id.item()]
        
        # 自回归生成
        for step in range(max_new_tokens):
            # 所有设备执行相同的forward
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=kv_caches,
                    use_cache=True
                )
                
            kv_caches = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            
            # 简单的greedy decoding
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]
            
            generated_tokens.append(next_token_id.item())
            
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
        dist.send(hidden_states.contiguous(), dst=self.rank + 1)
        
    def _receive_hidden_states(self) -> torch.Tensor:
        """从上一个设备接收hidden states"""
        # 需要知道shape，这里假设已知
        # 实际实现中可能需要先传输shape信息
        shape = torch.zeros(3, dtype=torch.long, device=f"cuda:{self.local_device}")
        dist.recv(shape, src=self.rank - 1)
        
        hidden_states = torch.zeros(
            tuple(shape.tolist()),
            dtype=torch.float16,
            device=f"cuda:{self.local_device}"
        )
        dist.recv(hidden_states, src=self.rank - 1)
        return hidden_states
        
    def _concat_kv_caches(self, cache1: List, cache2: List) -> List:
        """在sequence维度拼接两个KV cache"""
        if cache1 is None:
            return cache2
        if cache2 is None:
            return cache1
            
        concatenated = []
        for (k1, v1), (k2, v2) in zip(cache1, cache2):
            k_concat = torch.cat([k1, k2], dim=2)  # dim=2是seq_len维度
            v_concat = torch.cat([v1, v2], dim=2)
            concatenated.append((k_concat, v_concat))
        return concatenated
        
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
            generated_text = self.decode_phase(kv_caches, max_new_tokens)
            
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
            raise
        finally:
            self.cleanup()
            
    def cleanup(self):
        """清理资源"""
        dist.destroy_process_group()


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
        device_mode=args.device_mode
    )
    
    # 运行推理
    engine.run_inference(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
