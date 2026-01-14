"""
基于ZMQ的SP+PP分布式推理主脚本
使用ZMQ进行通信，适用于无线通信场景

特性：
1. 使用ZMQ替代torch.distributed，适用于无线网络
2. 六个队列（3发3收）分别处理token、hidden、cache
3. 优先级调度：token > hidden > cache
4. cache异步传输，不阻塞主计算

数据传输时机：
- cache: 每计算一层就传输一层（异步）
- hidden: 当前设备计算完成后发送（同步/阻塞）
- token: prefill阶段结束后同步（同步/阻塞）
"""

import os
import time
import argparse
import torch
from typing import List, Optional, Tuple, Dict
from transformers import AutoTokenizer, AutoConfig
import logging

from zmq_comm_manager import ZMQCommManager, create_zmq_comm_manager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ZMQDistributedInferenceEngine:
    """基于ZMQ的分布式推理引擎"""
    
    def __init__(
        self,
        model_path: str,
        rank: int,
        world_size: int,
        base_port: int = 29500,
        chunk_size: int = 128,
        comm_mode: str = "p2p",  # "p2p" or "ring"
        device: str = "cuda",
        node_addresses: Optional[Dict[int, str]] = None,
        startup_delay: float = 2.0,  # 启动延迟，等待所有节点就绪
    ):
        """
        初始化ZMQ分布式推理引擎
        
        Args:
            model_path: 模型路径
            rank: 当前设备rank (0, 1, 2)
            world_size: 总设备数 (3)
            base_port: 基础端口号
            chunk_size: SP的chunk大小
            comm_mode: 通信模式 ("p2p"或"ring")
            device: 设备类型
            node_addresses: 节点地址映射 {rank: ip}
            startup_delay: 启动延迟秒数
        """
        self.rank = rank
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.comm_mode = comm_mode
        self.model_path = model_path
        self.startup_delay = startup_delay
        
        # 初始化logger
        self.logger = self._setup_logger()
        
        # 设置本地设备
        self.local_device = self._get_local_device()
        self.device = f"cuda:{self.local_device}"
        
        # 创建ZMQ通信管理器
        self.logger.info(f"初始化ZMQ通信管理器 (mode={comm_mode})")
        self.comm = create_zmq_comm_manager(
            rank=rank,
            world_size=world_size,
            base_port=base_port,
            mode=comm_mode,
            device=self.device,
            node_addresses=node_addresses
        )
        
        # 等待其他节点启动
        self.logger.info(f"等待其他节点就绪 ({startup_delay}秒)...")
        time.sleep(startup_delay)
        
        # 启动通信管理器
        self.comm.start()
        
        # 加载模型和tokenizer
        self.logger.info(f"初始化设备 Rank {rank}/{world_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self._load_model()
        
        # 时间测量点
        self.timing_stats = {
            'prefill_start': 0,
            'prefill_compute_end': 0,  # 计算完成时间（不含cache同步）
            'prefill_end': 0,  # 包含cache同步的完成时间
            'decode_start': 0,
            'decode_end': 0,
            'cache_sync_start': 0,  # cache同步开始时间
            'cache_sync_end': 0,  # cache同步结束时间
        }
        
        # Cache接收状态追踪矩阵
        self.cache_received_indicator = None
        self.num_chunks = 0
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger(f"ZMQ-Rank-{self.rank}")
        logger.setLevel(logging.INFO)
        return logger
    
    def _get_local_device(self) -> int:
        """获取本地CUDA设备ID"""

        if torch.cuda.device_count() == 0:
            self.logger.info("本地无可用CUDA设备，默认使用逻辑设备 cuda:0")
            return 0

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            self.logger.info(f"检测到 CUDA_VISIBLE_DEVICES={visible_devices}，使用逻辑设备 cuda:0")
            return 0
        else:
            self.logger.info(f"使用GPU {self.rank}")
            return self.rank
    
    def _load_model(self):
        """加载模型（使用基础版本，不依赖torch.distributed）"""
        from modeling_qwen3_kv import Qwen3ForCausalLM
        
        self.logger.info(f"加载模型: {self.model_path}")
        
        model = Qwen3ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        model.eval()
        self.logger.info(f"模型加载完成，共{model.config.num_hidden_layers}层")
        
        # 初始化预分配的KV Cache
        self._initialize_kv_cache(model)
        
        return model
    
    def _initialize_kv_cache(self, model, max_length: int = 2200):
        """初始化KV Cache"""
        from kv_cache import initialize_past_key_values
        
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(model, max_length=max_length, batch_size=1)
        self.logger.info(f"KV Cache 预分配完成（最大长度: {max_length}）")
    
    def _reset_kv_cache(self):
        """重置KV Cache"""
        for layer_kv in self.past_key_values:
            layer_kv[0].reset()
            layer_kv[1].reset()
    
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
    
    def _get_owner_rank(self, layer_idx: int, num_layers: int) -> int:
        """获取负责某层的rank"""
        layers_per_rank = num_layers // self.world_size
        owner = layer_idx // layers_per_rank
        return min(owner, self.world_size - 1)
    
    def prefill_phase(self, prompt: str) -> Tuple[torch.Tensor, List]:
        """
        Prefill阶段：使用SP+PP
        
        数据流：
        - hidden states: 计算完当前rank的所有层后，发送给下一个rank（阻塞）
        - cache: 每计算一层就异步发送给其他rank
        - token: prefill结束后同步
        """
        self.logger.info("=" * 60)
        self.logger.info("开始 Prefill 阶段 (SP+PP with ZMQ)")
        self.timing_stats['prefill_start'] = time.time()
        self.timing_stats['cache_sync_start'] = time.time()
        
        # 重置KV Cache
        self._reset_kv_cache()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        self.logger.info(f"Prompt长度: {input_ids.shape[1]} tokens")
        
        # 切分chunks
        chunks = self._split_prompt_chunks(input_ids)
        self.num_chunks = len(chunks)
        self.logger.info(f"切分为 {len(chunks)} 个chunks (chunk_size={self.chunk_size})")
        
        # 获取层范围
        num_layers = self.model.config.num_hidden_layers
        start_layer, end_layer = self._get_layer_range(num_layers)
        self.logger.info(f"负责层范围: [{start_layer}, {end_layer})")
        
        # 初始化Cache状态矩阵
        self.cache_received_indicator = torch.zeros(
            (self.num_chunks, num_layers),
            dtype=torch.int8,
            device='cpu'
        )
        
        last_hidden = None
        
        for chunk_idx, chunk in enumerate(chunks):
            self.logger.info(f"处理 chunk {chunk_idx+1}/{len(chunks)}")
            
            # 第一个rank从embedding开始
            if self.rank == 0:
                hidden_states = self.model.model.embed_tokens(chunk)
                self.logger.debug(f"Rank 0: 生成embedding, shape={hidden_states.shape}")
                batch_size, seq_length = chunk.shape
            else:
                # 接收上一个rank的hidden states（阻塞）
                result = self.comm.recv_hidden(src_rank=self.rank - 1, timeout=60.0)
                if result is None:
                    raise RuntimeError(f"Rank {self.rank}: 接收hidden states超时")
                hidden_states, recv_chunk_idx = result
                batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
                self.logger.debug(f"Rank {self.rank}: 接收hidden states, shape={hidden_states.shape}")
            
            # 计算past_key_values_length
            past_key_values_length = 0
            if self.past_key_values[0][0].current_length > 0:
                past_key_values_length = self.past_key_values[0][0].current_length.item()
            
            seq_length_with_past = seq_length + past_key_values_length
            
            # 准备attention mask和position ids
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=self.device,
            )
            
            attention_mask = self.model.model._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
            
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            
            # 计算当前rank负责的层
            for layer_idx in range(start_layer, end_layer):
                self.logger.debug(f"  计算层 {layer_idx}")
                
                # 获取decoder layer
                decoder_layer = self.model.model.layers[layer_idx]
                
                # Forward - 模型内部会自动 cat 新的 KV 到 past_key_values
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_values[layer_idx],
                    output_attentions=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                
                hidden_states = layer_outputs[0]
                
                # 获取本层新增的 KV（用于发送给其他 rank）
                # 由于模型内部已经 cat 了，我们需要提取新增部分
                current_len = self.past_key_values[layer_idx][0].current_length.item()
                new_key = self.past_key_values[layer_idx][0].get_data()[:, :, current_len - seq_length:, :]
                new_value = self.past_key_values[layer_idx][1].get_data()[:, :, current_len - seq_length:, :]
                
                # 异步发送cache给其他rank
                for other_rank in range(self.world_size):
                    if other_rank != self.rank:
                        self.comm.send_cache_async(
                            kv_cache=(new_key, new_value),
                            dst_rank=other_rank,
                            layer_idx=layer_idx,
                            chunk_idx=chunk_idx
                        )
                
                self.cache_received_indicator[chunk_idx, layer_idx] = 1
            
            # 计算完成后，发送hidden states给下一个rank（阻塞）
            if self.rank < self.world_size - 1:
                self.comm.send_hidden(hidden_states, dst_rank=self.rank + 1, chunk_idx=chunk_idx)
                self.logger.debug(f"Rank {self.rank}: 发送hidden states到 Rank {self.rank+1}")
            
            # 最后一个rank过norm
            if self.rank == self.world_size - 1:
                hidden_states = self.model.model.norm(hidden_states)
            
            last_hidden = hidden_states
            
            # 处理接收到的其他rank的cache（非阻塞检查）
            self._process_received_caches(chunk_idx)
            
            self.logger.info(f"  chunk {chunk_idx+1} 处理完成")
        
        # Prefill结束，等待所有cache接收完成, 记录完成时间
        self.timing_stats['prefill_compute_end'] = time.time()
        
        # Token同步（只有最后一个rank有有效的hidden states）
        # 这里广播first token给所有rank
        if self.rank == self.world_size - 1:
            with torch.no_grad():
                logits = self.model.lm_head(last_hidden[:, -1:, :])
                first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            self.comm.send_token(first_token, dst_rank=-1)  # 广播
            self.first_token = first_token
        else:
            self.first_token = self.comm.recv_token(src_rank=self.world_size - 1)
        self.logger.info(f"First token 同步完成: {self.first_token.item()}")

        self._wait_for_all_caches()
        self.timing_stats['cache_sync_end'] = time.time()
        self.timing_stats['prefill_end'] = time.time()

        compute_time = self.timing_stats['prefill_compute_end'] - self.timing_stats['prefill_start']
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        cache_sync_time = self.timing_stats['cache_sync_end'] - self.timing_stats['cache_sync_start']
        self.logger.info(f"Prefill 总耗时: {prefill_time:.3f}s, cache同步耗时: {cache_sync_time:.3f}s, 计算耗时: {compute_time:.3f}s")
        
        return last_hidden, self.past_key_values
    
    def _process_received_caches(self, current_chunk_idx: int):
        """处理接收到的cache（非阻塞）"""
        processed = 0
        while True:
            result = self.comm.get_received_cache(timeout=0.001)
            if result is None:
                break
            
            kv_cache, src_rank, layer_idx, chunk_idx = result
            key, value = kv_cache
            
            # 追加到KV Cache
            self.past_key_values[layer_idx][0].cat(key)   # 这里应该检查chunk_idx后再cat吧，万一chunk3比chunk2先到？
            self.past_key_values[layer_idx][1].cat(value)
            
            self.cache_received_indicator[chunk_idx, layer_idx] = 1
            processed += 1
        
        if processed > 0:
            self.logger.debug(f"处理了 {processed} 个cache")
    
    def _wait_for_all_caches(self, timeout: float = 60.0):
        """等待接收所有其他rank的cache"""
        num_layers = self.model.config.num_hidden_layers
        start_layer, end_layer = self._get_layer_range(num_layers)
        
        # 计算需要接收的cache数量
        # 每个chunk，需要从其他rank接收他们负责的层的cache
        expected_caches = 0
        for chunk_idx in range(self.num_chunks):
            for layer_idx in range(num_layers):
                if not (start_layer <= layer_idx < end_layer):
                    # 这层不是我负责的，需要接收
                    if self.cache_received_indicator[chunk_idx, layer_idx] == 0:
                        expected_caches += 1
        
        self.logger.info(f"等待接收 {expected_caches} 个cache...")
        
        start_time = time.time()
        while expected_caches > 0 and (time.time() - start_time) < timeout:
            result = self.comm.get_received_cache(timeout=0.1)
            if result is None:
                continue
            
            kv_cache, src_rank, layer_idx, chunk_idx = result
            key, value = kv_cache
            
            if self.cache_received_indicator[chunk_idx, layer_idx] == 0:
                self.past_key_values[layer_idx][0].cat(key)
                self.past_key_values[layer_idx][1].cat(value)
                self.cache_received_indicator[chunk_idx, layer_idx] = 1
                expected_caches -= 1
        
        if expected_caches > 0:
            self.logger.warning(f"Cache接收超时，仍有 {expected_caches} 个未接收")
        else:
            self.logger.info("所有cache接收完成")
    
    def decode_phase(self, last_hidden: torch.Tensor, kv_caches: List, max_new_tokens: int = 100) -> str:
        """
        Decode阶段：所有设备进行相同的全量计算
        """
        self.logger.info("=" * 60)
        self.logger.info("开始 Decode 阶段 (全量冗余计算)")
        self.timing_stats['decode_start'] = time.time()
        
        # 使用prefill阶段同步的first token
        next_token_id = self.first_token
        generated_tokens = [next_token_id.item()]
        
        # 自回归生成
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=self.past_key_values,  # 直接传入KVCache对象
                    use_cache=True
                )
            
            # 模型内部已经更新了KV Cache，不需要手动cat
            
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token_id.item())
            
            if (step + 1) % 10 == 0:
                self.logger.info(f"已生成 {step+1} tokens")
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                self.logger.info(f"遇到EOS token，停止生成 (step {step+1})")
                break
        
        self.timing_stats['decode_end'] = time.time()
        decode_time = self.timing_stats['decode_end'] - self.timing_stats['decode_start']
        self.logger.info(f"Decode 完成，耗时: {decode_time:.3f}s")
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def print_timing_stats(self):
        """打印时间统计"""
        self.logger.info("=" * 60)
        self.logger.info("时间统计:")
        prefill_compute_time = self.timing_stats['prefill_compute_end'] - self.timing_stats['prefill_start']
        cache_sync_time = self.timing_stats['cache_sync_end'] - self.timing_stats['cache_sync_start']
        prefill_time = self.timing_stats['prefill_end'] - self.timing_stats['prefill_start']
        decode_time = self.timing_stats['decode_end'] - self.timing_stats['decode_start']
        total_time = self.timing_stats['decode_end'] - self.timing_stats['prefill_start']
        
        self.logger.info(f"  Prefill 计算时间:     {prefill_compute_time:.3f}s")
        self.logger.info(f"  Cache 同步时间:      {cache_sync_time:.3f}s")
        self.logger.info(f"  Prefill 总时间:      {prefill_time:.3f}s")
        self.logger.info(f"  通信开销占比:        {cache_sync_time/prefill_time*100:.1f}%")
        self.logger.info(f"  Decode 时间:        {decode_time:.3f}s")
        self.logger.info(f"  总时间:            {total_time:.3f}s")
        
        # 打印通信统计
        stats = self.comm.get_stats()
        self.logger.info(f"  Token 发送/接收:   {stats['token_sent']}/{stats['token_recv']}")
        self.logger.info(f"  Hidden 发送/接收:  {stats['hidden_sent']}/{stats['hidden_recv']}")
        self.logger.info(f"  Cache 发送/接收:   {stats['cache_sent']}/{stats['cache_recv']}")
        if self.comm_mode == 'ring':
            self.logger.info(f"  聚合发送次数:      {stats['aggregated_sends']}")
            self.logger.info(f"  转发消息数:        {stats['forwarded_messages']}")
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
            raise
        finally:
            self.cleanup()
    
    def get_timing_stats(self) -> dict:
        """获取时间统计信息（用于性能测试）"""
        return {
            'prefill_compute_time': self.timing_stats['prefill_compute_end'] - self.timing_stats['prefill_start'],
            'cache_sync_time': self.timing_stats['cache_sync_end'] - self.timing_stats['cache_sync_start'],
            'prefill_total_time': self.timing_stats['prefill_end'] - self.timing_stats['prefill_start'],
            'decode_time': self.timing_stats['decode_end'] - self.timing_stats['decode_start'],
            'comm_stats': self.comm.get_stats()
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理日志handler
            for handler in self.logger.handlers:
                handler.flush()
            
            # 清理通信管理器
            if self.comm:
                self.comm.stop()
                self.comm = None
            
            # 清理模型
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # 清理KV cache
            if hasattr(self, 'past_key_values') and self.past_key_values is not None:
                del self.past_key_values
                self.past_key_values = None
            
            # 清理tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # 清理GPU缓存
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"Cleanup出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='基于ZMQ的SP+PP分布式推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--rank', type=int, required=True, help='当前设备rank (0, 1, 2)')
    parser.add_argument('--world_size', type=int, default=3, help='总设备数')
    parser.add_argument('--base_port', type=int, default=29500, help='基础端口号')
    parser.add_argument('--chunk_size', type=int, default=128, help='SP的chunk大小')
    parser.add_argument('--comm_mode', type=str, default='p2p', 
                       choices=['p2p', 'ring'], help='通信模式')
    parser.add_argument('--prompt', type=str, default='请详细介绍一下人工智能的发展历史。', 
                       help='输入prompt')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='最大生成token数')
    parser.add_argument('--node_addrs', type=str, default=None,
                       help='节点地址，格式: "0:192.168.1.1,1:192.168.1.2,2:192.168.1.3"')
    parser.add_argument('--startup_delay', type=float, default=2.0,
                       help='启动延迟（秒），等待其他节点就绪')
    
    args = parser.parse_args()
    
    # 解析节点地址
    node_addresses = None
    if args.node_addrs:
        node_addresses = {}
        for item in args.node_addrs.split(','):
            rank_str, addr = item.split(':')
            node_addresses[int(rank_str)] = addr
    
    # 创建推理引擎
    engine = ZMQDistributedInferenceEngine(
        model_path=args.model_path,
        rank=args.rank,
        world_size=args.world_size,
        base_port=args.base_port,
        chunk_size=args.chunk_size,
        comm_mode=args.comm_mode,
        node_addresses=node_addresses,
        startup_delay=args.startup_delay
    )
    
    # 运行推理
    engine.run_inference(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
