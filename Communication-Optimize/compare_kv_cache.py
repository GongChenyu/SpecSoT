"""
对比单设备推理和分布式推理的KV Cache一致性测试

功能：
1. 使用相同的随机种子
2. 单设备推理（GPU 4）保存prefill后的KV Cache
3. 分布式推理（3个GPU）保存prefill后的KV Cache
4. 对比两者的KV Cache是否一致

使用方法：
1. 先运行推理阶段：python compare_kv_cache.py --mode run
2. 再运行分析阶段：python compare_kv_cache.py --mode analyze
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
import argparse
import logging
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def set_random_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_kv_cache(kv_cache_list: List, save_path: str, metadata: Dict = None):
    """
    保存KV Cache到文件
    
    Args:
        kv_cache_list: KV Cache列表，每层包含[key_cache, value_cache]
        save_path: 保存路径
        metadata: 元数据（如层数、形状等）
    """
    logger = logging.getLogger("SaveKVCache")
    
    # 准备保存的数据
    cache_data = {
        'metadata': metadata or {},
        'num_layers': len(kv_cache_list),
        'caches': []
    }
    
    for layer_idx, (key_cache, value_cache) in enumerate(kv_cache_list):
        # 提取有效的cache数据（到current_length）
        if hasattr(key_cache, 'get_data'):
            # 如果是KVCache对象
            key_data = key_cache.get_data().cpu()
            value_data = value_cache.get_data().cpu()
        else:
            # 如果是普通tensor
            key_data = key_cache.cpu()
            value_data = value_cache.cpu()
        
        cache_data['caches'].append({
            'layer_idx': layer_idx,
            'key': key_data,
            'value': value_data,
            'key_shape': tuple(key_data.shape),
            'value_shape': tuple(value_data.shape),
        })
        
        logger.info(f"Layer {layer_idx}: key_shape={key_data.shape}, value_shape={value_data.shape}")
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    logger.info(f"KV Cache已保存到: {save_path}")
    logger.info(f"总层数: {cache_data['num_layers']}")
    
    return cache_data


def load_kv_cache(load_path: str) -> Dict:
    """加载KV Cache"""
    logger = logging.getLogger("LoadKVCache")
    
    with open(load_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    logger.info(f"KV Cache已加载: {load_path}")
    logger.info(f"总层数: {cache_data['num_layers']}")
    
    return cache_data


# ==================== 单设备推理 ====================

def run_single_device_inference(
    model_path: str,
    prompt: str,
    chunk_size: int,
    gpu_id: int,
    save_path: str,
    seed: int = 42
):
    """
    单设备推理并保存KV Cache
    
    Args:
        model_path: 模型路径
        prompt: 输入prompt
        chunk_size: chunk大小
        gpu_id: GPU ID
        save_path: Cache保存路径
        seed: 随机种子
    """
    logger = logging.getLogger(f"SingleDevice-GPU{gpu_id}")
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'
    
    # 设置随机种子
    set_random_seed(seed)
    
    logger.info(f"开始单设备推理 (GPU {gpu_id})")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"随机种子: {seed}")
    
    # 加载模型
    from transformers import AutoTokenizer
    from modeling_qwen3_kv import Qwen3ForCausalLM
    from kv_cache import initialize_past_key_values
    
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info("加载模型...")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    logger.info(f"模型加载完成，共{num_layers}层")
    
    # 初始化KV Cache
    logger.info("初始化KV Cache...")
    past_key_values, past_key_values_data, current_length_data = \
        initialize_past_key_values(model, max_length=2200, batch_size=1)
    
    # Tokenize
    logger.info(f"Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt长度: {seq_len} tokens")
    
    # 切分chunks
    chunks = []
    for i in range(0, seq_len, chunk_size):
        chunk = input_ids[:, i:i+chunk_size]
        chunks.append(chunk)
    
    num_chunks = len(chunks)
    logger.info(f"切分为 {num_chunks} 个chunks (chunk_size={chunk_size})")
    
    # Prefill阶段
    logger.info("="*60)
    logger.info("开始 Prefill 阶段")
    start_time = time.time()
    
    with torch.no_grad():
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"处理 chunk {chunk_idx+1}/{num_chunks}")
            
            batch_size, seq_length = chunk.shape
            
            # Embedding
            hidden_states = model.model.embed_tokens(chunk)
            
            # 计算past_key_values_length
            past_key_values_length = 0
            if past_key_values[0][0].current_length > 0:
                past_key_values_length = past_key_values[0][0].current_length.item()
            
            seq_length_with_past = seq_length + past_key_values_length
            
            # 准备attention mask和position ids
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=device,
            )
            
            attention_mask = model.model._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
            
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
            
            # 逐层计算
            for layer_idx in range(num_layers):
                decoder_layer = model.model.layers[layer_idx]
                
                # Forward - 模型内部会自动 cat 新的 KV
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[layer_idx],
                    output_attentions=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                
                hidden_states = layer_outputs[0]
            
            # Norm
            hidden_states = model.model.norm(hidden_states)
            
            logger.info(f"  chunk {chunk_idx+1} 处理完成")

        logits = model.lm_head(hidden_states)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        print("Next tokens:", next_token, tokenizer.decode(next_token[0]))

    prefill_time = time.time() - start_time
    logger.info(f"Prefill 完成，耗时: {prefill_time:.3f}s")
    
    # 保存KV Cache
    logger.info("保存KV Cache...")
    metadata = {
        'model_path': model_path,
        'prompt': prompt,
        'chunk_size': chunk_size,
        'num_chunks': num_chunks,
        'seq_len': seq_len,
        'num_layers': num_layers,
        'seed': seed,
        'device': f'GPU {gpu_id}',
        'prefill_time': prefill_time,
    }
    
    save_kv_cache(past_key_values, save_path, metadata)
    
    logger.info("="*60)
    logger.info("单设备推理完成")
    
    return past_key_values


# ==================== 分布式推理 ====================

def run_distributed_rank(
    rank: int,
    world_size: int,
    model_path: str,
    prompt: str,
    chunk_size: int,
    base_port: int,
    gpu_ids: List[int],
    cache_save_dir: str,
    seed: int
):
    """分布式推理的单个rank"""
    logger = logging.getLogger(f"Distributed-Rank{rank}")
    
    # 设置GPU
    gpu_id = gpu_ids[rank]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 设置随机种子
    set_random_seed(seed)
    
    logger.info(f"Rank {rank} 启动 (GPU {gpu_id})")
    
    try:
        # 导入推理引擎
        from zmq_distributed_inference import ZMQDistributedInferenceEngine
        
        # 创建引擎
        engine = ZMQDistributedInferenceEngine(
            model_path=model_path,
            rank=rank,
            world_size=world_size,
            base_port=base_port,
            chunk_size=chunk_size,
            comm_mode='ring',   # 'p2p' 或 'ring'
            startup_delay=1.0
        )
        
        # 只运行prefill阶段
        logger.info("开始 Prefill 阶段")
        last_hidden, kv_caches = engine.prefill_phase(prompt)

        if rank == 2:
            logits = engine.model.lm_head(last_hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            print(f"Rank {rank} Next tokens:", next_token, engine.tokenizer.decode(next_token[0]))
        
        # 保存本rank的KV Cache
        save_path = os.path.join(cache_save_dir, f'distributed_rank{rank}_cache.pkl')
        
        metadata = {
            'rank': rank,
            'world_size': world_size,
            'model_path': model_path,
            'prompt': prompt,
            'chunk_size': chunk_size,
            'gpu_id': gpu_id,
            'seed': seed,
            'num_layers': engine.model.config.num_hidden_layers,
        }
        
        save_kv_cache(kv_caches, save_path, metadata)
        
        logger.info(f"Rank {rank} Prefill完成")
        
        # 清理
        engine.cleanup()
        
    except Exception as e:
        logger.error(f"Rank {rank} 出错: {e}", exc_info=True)
        sys.exit(1)


def run_distributed_inference(
    model_path: str,
    prompt: str,
    chunk_size: int,
    gpu_ids: List[int],
    cache_save_dir: str,
    base_port: int,
    seed: int = 42
):
    """运行分布式推理"""
    logger = logging.getLogger("DistributedInference")
    
    world_size = len(gpu_ids)
    logger.info(f"开始分布式推理 (world_size={world_size})")
    logger.info(f"GPU IDs: {gpu_ids}")
    logger.info(f"随机种子: {seed}")
    
    # 清理端口
    import subprocess
    ports_to_clean = set()
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                port = base_port + sender * world_size + receiver
                ports_to_clean.add(port)
    
    for port in ports_to_clean:
        subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
            shell=True,
            capture_output=True
        )
    
    logger.info("端口已清理，等待释放...")
    time.sleep(2)
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(
            target=run_distributed_rank,
            args=(rank, world_size, model_path, prompt, chunk_size, 
                  base_port, gpu_ids, cache_save_dir, seed)
        )
        p.start()
        processes.append(p)
        logger.info(f"启动 Rank {rank} (PID: {p.pid}, GPU: {gpu_ids[rank]})")
    
    # 等待完成
    for rank, p in enumerate(processes):
        p.join(timeout=600)
        if p.is_alive():
            logger.warning(f"Rank {rank} 超时，强制终止")
            p.terminate()
        elif p.exitcode == 0:
            logger.info(f"Rank {rank} 成功完成")
        else:
            logger.error(f"Rank {rank} 失败 (退出码: {p.exitcode})")
    
    logger.info("="*60)
    logger.info("分布式推理完成")


# ==================== 分析对比 ====================

def analyze_kv_cache_difference(
    single_cache_path: str,
    distributed_cache_dir: str,
    world_size: int = 3,
    rtol: float = 1e-3,
    atol: float = 1e-5
):
    """
    分析单设备和分布式KV Cache的差异
    
    Args:
        single_cache_path: 单设备cache文件路径
        distributed_cache_dir: 分布式cache目录
        world_size: 分布式设备数
        rtol: 相对容差
        atol: 绝对容差
    """
    logger = logging.getLogger("CacheAnalyzer")
    
    logger.info("="*60)
    logger.info("开始分析KV Cache差异")
    logger.info("="*60)
    
    # 加载单设备cache
    logger.info(f"加载单设备cache: {single_cache_path}")
    single_cache = load_kv_cache(single_cache_path)
    
    # 加载分布式cache
    distributed_caches = []
    for rank in range(world_size):
        cache_path = os.path.join(distributed_cache_dir, f'distributed_rank{rank}_cache.pkl')
        logger.info(f"加载分布式cache Rank {rank}: {cache_path}")
        distributed_caches.append(load_kv_cache(cache_path))
    
    # 合并分布式cache
    logger.info("\n合并分布式KV Cache...")
    num_layers = single_cache['num_layers']
    
    # 获取每个rank负责的层范围
    layers_per_rank = num_layers // world_size
    
    merged_distributed_cache = {'num_layers': num_layers, 'caches': [None] * num_layers}
    
    for rank in range(world_size):
        start_layer = rank * layers_per_rank
        end_layer = start_layer + layers_per_rank if rank < world_size - 1 else num_layers
        
        logger.info(f"Rank {rank} 负责层 [{start_layer}, {end_layer})")
        
        # 从该rank的cache中提取对应的层
        for layer_idx in range(start_layer, end_layer):
            # 在该rank的cache列表中找到对应层
            for cache_item in distributed_caches[rank]['caches']:
                if cache_item['layer_idx'] == layer_idx:
                    merged_distributed_cache['caches'][layer_idx] = cache_item
                    break
    
    # 检查是否所有层都有数据
    missing_layers = [i for i, cache in enumerate(merged_distributed_cache['caches']) if cache is None]
    if missing_layers:
        logger.error(f"缺失层: {missing_layers}")
        return
    
    logger.info("分布式cache合并完成")
    
    # 逐层对比
    logger.info("\n"+"="*60)
    logger.info("逐层对比KV Cache")
    logger.info("="*60)
    
    total_diff_key = 0.0
    total_diff_value = 0.0
    max_diff_key = 0.0
    max_diff_value = 0.0
    max_diff_layer = -1
    
    all_match = True
    
    for layer_idx in range(num_layers):
        single_layer = single_cache['caches'][layer_idx]
        dist_layer = merged_distributed_cache['caches'][layer_idx]
        
        single_key = single_layer['key']
        single_value = single_layer['value']
        dist_key = dist_layer['key']
        dist_value = dist_layer['value']
        
        # 检查形状
        if single_key.shape != dist_key.shape:
            logger.error(f"Layer {layer_idx}: Key形状不匹配! "
                        f"单设备={single_key.shape}, 分布式={dist_key.shape}")
            all_match = False
            continue
        
        if single_value.shape != dist_value.shape:
            logger.error(f"Layer {layer_idx}: Value形状不匹配! "
                        f"单设备={single_value.shape}, 分布式={dist_value.shape}")
            all_match = False
            continue
        
        # 计算差异
        key_diff = torch.abs(single_key - dist_key)
        value_diff = torch.abs(single_value - dist_value)
        
        key_max_diff = key_diff.max().item()
        value_max_diff = value_diff.max().item()
        
        key_mean_diff = key_diff.mean().item()
        value_mean_diff = value_diff.mean().item()
        
        total_diff_key += key_mean_diff
        total_diff_value += value_mean_diff
        
        if key_max_diff > max_diff_key:
            max_diff_key = key_max_diff
            max_diff_layer = layer_idx
        
        if value_max_diff > max_diff_value:
            max_diff_value = value_max_diff
        
        # 使用torch.allclose检查
        key_match = torch.allclose(single_key, dist_key, rtol=rtol, atol=atol)
        value_match = torch.allclose(single_value, dist_value, rtol=rtol, atol=atol)
        
        match_str = "✓" if (key_match and value_match) else "✗"
        
        logger.info(f"Layer {layer_idx:2d} {match_str}: "
                   f"Key(max={key_max_diff:.2e}, mean={key_mean_diff:.2e}), "
                   f"Value(max={value_max_diff:.2e}, mean={value_mean_diff:.2e})")
        
        if not (key_match and value_match):
            all_match = False
            
            # 详细分析差异分布
            logger.warning(f"  Key不匹配的元素比例: {(key_diff > atol).float().mean().item()*100:.2f}%")
            logger.warning(f"  Value不匹配的元素比例: {(value_diff > atol).float().mean().item()*100:.2f}%")
    
    # 总结
    logger.info("\n"+"="*60)
    logger.info("对比总结")
    logger.info("="*60)
    logger.info(f"总层数: {num_layers}")
    logger.info(f"Key平均差异: {total_diff_key/num_layers:.2e}")
    logger.info(f"Value平均差异: {total_diff_value/num_layers:.2e}")
    logger.info(f"Key最大差异: {max_diff_key:.2e} (Layer {max_diff_layer})")
    logger.info(f"Value最大差异: {max_diff_value:.2e}")
    logger.info(f"容差设置: rtol={rtol}, atol={atol}")
    
    if all_match:
        logger.info("\n✓ 所有层的KV Cache完全匹配!")
    else:
        logger.warning("\n✗ 存在KV Cache不匹配的层!")
        logger.warning("可能的原因:")
        logger.warning("  1. 分布式计算时cache到达顺序不一致，导致cat顺序错误")
        logger.warning("  2. 不同GPU的浮点计算精度差异")
        logger.warning("  3. 随机数生成器状态不同步")
    
    logger.info("="*60)
    
    return all_match


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='对比单设备和分布式推理的KV Cache')
    parser.add_argument('--mode', type=str, default='run',
                       choices=['run', 'analyze', 'all'],
                       help='运行模式: run(只运行推理), analyze(只分析), all(运行+分析)')
    parser.add_argument('--model_path', type=str, 
                       default='/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
                       help='模型路径')
    parser.add_argument('--prompt', type=str,
                       default='请详细介绍一下人工智能的发展历史。',
                       help='测试prompt')
    parser.add_argument('--chunk_size', type=int, default=128,
                       help='Chunk大小')
    parser.add_argument('--single_gpu', type=int, default=4,
                       help='单设备推理使用的GPU ID')
    parser.add_argument('--dist_gpus', type=int, nargs='+', default=[5, 6, 7],
                       help='分布式推理使用的GPU IDs')
    parser.add_argument('--base_port', type=int, default=47000,
                       help='分布式通信基础端口')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--cache_dir', type=str, 
                       default='/data/home/chenyu/Coding/SD+SoT/Speculative-Decoding-Enabled-Skeleton-of-Thought/Communication-Optimize/cache_comparison',
                       help='Cache保存目录')
    parser.add_argument('--rtol', type=float, default=1e-3,
                       help='相对容差')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='绝对容差')
    
    args = parser.parse_args()
    
    # 创建cache目录
    os.makedirs(args.cache_dir, exist_ok=True)
    
    single_cache_path = os.path.join(args.cache_dir, 'single_device_cache.pkl')
    
    # 运行推理
    if args.mode in ['run', 'all']:
        print("\n" + "="*60)
        print("阶段 1: 运行推理")
        print("="*60)
        
        # 单设备推理
        print("\n[1/2] 运行单设备推理...")
        run_single_device_inference(
            model_path=args.model_path,
            prompt=args.prompt,
            chunk_size=args.chunk_size,
            gpu_id=args.single_gpu,
            save_path=single_cache_path,
            seed=args.seed
        )
        
        # 等待GPU释放
        time.sleep(3)
        
        # 分布式推理
        print("\n[2/2] 运行分布式推理...")
        run_distributed_inference(
            model_path=args.model_path,
            prompt=args.prompt,
            chunk_size=args.chunk_size,
            gpu_ids=args.dist_gpus,
            cache_save_dir=args.cache_dir,
            base_port=args.base_port,
            seed=args.seed
        )
        
        print("\n推理阶段完成!")
    
    # 分析对比
    if args.mode in ['analyze', 'all']:
        print("\n" + "="*60)
        print("阶段 2: 分析KV Cache差异")
        print("="*60)
        
        # 等待一下确保文件写入完成
        if args.mode == 'all':
            time.sleep(2)
        
        analyze_kv_cache_difference(
            single_cache_path=single_cache_path,
            distributed_cache_dir=args.cache_dir,
            world_size=len(args.dist_gpus),
            rtol=args.rtol,
            atol=args.atol
        )
    
    print("\n所有任务完成!")


if __name__ == "__main__":
    main()
