"""
ZMQ通信模式性能对比测试
对比P2P和Ring两种通信模式在分布式推理中的性能表现

测试内容：
1. Prefill计算时间 vs Prefill总时间（包含cache同步）
2. Ring模式 vs P2P模式的性能对比
3. 通信开销占比分析
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing as mp
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
import gc

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def setup_logging(rank: int, test_name: str, log_dir: str = "./benchmark_logs"):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{test_name}_rank_{rank}_{timestamp}.log")
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # 控制台只显示警告及以上
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return log_file


def cleanup_ports(base_port: int, world_size: int):
    """清理被占用的端口"""
    ports_to_clean = set()
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                port = base_port + sender * world_size + receiver
                ports_to_clean.add(port)
    
    for port in ports_to_clean:
        try:
            subprocess.run(
                f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
                shell=True,
                capture_output=True,
                text=True
            )
        except Exception:
            pass


def benchmark_worker(rank: int, base_config: dict, task_queue: mp.Queue, result_queue: mp.Queue):
    """
    持续运行的benchmark worker进程
    模型只加载一次，接收测试任务并执行
    """
    logger = None
    engine = None
    
    try:
        # 设置GPU环境变量
        gpu_ids = base_config['gpu_ids']
        gpu_id = gpu_ids[rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 设置日志
        log_dir = base_config.get('log_dir', './benchmark_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"benchmark_worker_rank_{rank}_{timestamp}.log")
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        logger.info(f"Worker Rank {rank} 启动，日志: {log_file}")
        
        # 添加项目路径到sys.path
        comm_optimize_dir = os.path.dirname(os.path.abspath(__file__))
        if comm_optimize_dir not in sys.path:
            sys.path.insert(0, comm_optimize_dir)
        
        # 等待所有worker启动
        time.sleep(base_config.get('startup_delay', 2.0))
        
        # 导入推理引擎
        from zmq_distributed_inference import ZMQDistributedInferenceEngine
        
        current_engine = None
        current_mode = None
        
        # 持续处理任务
        while True:
            try:
                # 从队列获取任务，带超时
                task = task_queue.get(timeout=1.0)
                
                # 检查是否是停止信号
                if task == 'STOP' or (isinstance(task, dict) and task.get('command') == 'STOP'):
                    logger.info(f"Rank {rank} 收到停止信号")
                    break
                
                test_mode = task['comm_mode']
                test_id = task['test_id']
                is_warmup = task.get('is_warmup', False)
                
                logger.info(f"Rank {rank} 开始测试: mode={test_mode}, test_id={test_id}, warmup={is_warmup}")
                
                # 如果通信模式改变，需要重新创建引擎
                if current_mode != test_mode:
                    if current_engine is not None:
                        logger.info(f"Rank {rank} 切换模式: {current_mode} -> {test_mode}，清理旧引擎")
                        # 清理旧引擎
                        current_engine.cleanup()
                        del current_engine
                        current_engine = None
                        
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # 等待资源完全释放
                        logger.info(f"Rank {rank} 等待5秒让资源完全释放...")
                        time.sleep(5.0)
                    
                    logger.info(f"Rank {rank} 创建新引擎: mode={test_mode}")
                    current_engine = ZMQDistributedInferenceEngine(
                        model_path=base_config['model_path'],
                        rank=rank,
                        world_size=base_config['world_size'],
                        base_port=base_config['base_port'],
                        chunk_size=base_config['chunk_size'],
                        comm_mode=test_mode,
                        node_addresses=base_config.get('node_addresses'),
                        startup_delay=0.5  # 引擎已经等待过了
                    )
                    current_mode = test_mode
                    logger.info(f"Rank {rank} 引擎创建完成")
                
                # 重置KV cache
                current_engine._reset_kv_cache()
                
                # 运行Prefill阶段
                last_hidden, kv_caches = current_engine.prefill_phase(task['prompt'])
                
                # 获取时间统计
                timing_stats = current_engine.get_timing_stats()
                
                # 将结果放入队列
                result_queue.put({
                    'rank': rank,
                    'comm_mode': test_mode,
                    'test_id': test_id,
                    'is_warmup': is_warmup,
                    'prefill_compute_time': timing_stats['prefill_compute_time'],
                    'cache_sync_time': timing_stats['cache_sync_time'],
                    'prefill_total_time': timing_stats['prefill_total_time'],
                    'comm_stats': timing_stats['comm_stats']
                })
                
                logger.info(f"Rank {rank} 完成测试: test_id={test_id}")
                
            except mp.queues.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Rank {rank} 测试出错: {e}", exc_info=True)
                result_queue.put({
                    'rank': rank,
                    'error': str(e),
                    'test_id': task.get('test_id', -1)
                })
        
        # 清理
        if current_engine is not None:
            logger.info(f"Rank {rank} 清理引擎")
            current_engine.cleanup()
            del current_engine
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"Rank {rank} 退出")
        
    except Exception as e:
        error_msg = f"Worker Rank {rank} 严重错误: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        else:
            print(error_msg)
        import traceback
        traceback.print_exc()
    finally:
        if logger:
            for handler in logger.handlers:
                handler.flush()


def run_test_iteration(task_queue: mp.Queue, result_queue: mp.Queue, config: dict, test_id: int, is_warmup: bool = False) -> List[Dict]:
    """运行一次完整的测试迭代（通过任务队列）"""
    
    if not is_warmup:
        print(f"\n{'='*60}")
        print(f"运行测试: {config['comm_mode'].upper()} 模式 - 第 {test_id} 次")
        print(f"{'='*60}")
    else:
        print(f"预热: {config['comm_mode'].upper()} 模式 - 第 {test_id} 次")
    
    # 向所有worker发送任务
    task = {
        'comm_mode': config['comm_mode'],
        'test_id': test_id,
        'is_warmup': is_warmup,
        'prompt': config['prompt']
    }
    
    for _ in range(config['world_size']):
        task_queue.put(task)
    
    # 等待所有结果
    results = []
    timeout = 120  # 2分钟超时
    start_time = time.time()
    
    while len(results) < config['world_size']:
        if time.time() - start_time > timeout:
            print(f"⚠ 测试超时！已收到 {len(results)}/{config['world_size']} 个结果")
            break
        
        try:
            result = result_queue.get(timeout=5.0)
            if result.get('test_id') == test_id:
                results.append(result)
                if 'error' not in result:
                    print(f"  ✓ Rank {result['rank']} 完成")
                else:
                    print(f"  ✗ Rank {result['rank']} 失败: {result['error']}")
        except:
            continue
    
    return results


def analyze_results(all_results: Dict[str, List[List[Dict]]]):
    """分析测试结果"""
    print("\n" + "="*80)
    print("性能测试结果分析")
    print("="*80)
    
    for mode in ['ring', 'p2p']:
        if mode not in all_results:
            continue
        
        print(f"\n{'='*80}")
        print(f"{mode.upper()} 模式统计")
        print(f"{'='*80}")
        
        mode_results = all_results[mode]
        
        # 按rank整理数据
        rank_data = {}
        for test_results in mode_results:
            for result in test_results:
                if 'error' in result:
                    continue
                rank = result['rank']
                if rank not in rank_data:
                    rank_data[rank] = {
                        'prefill_compute_times': [],
                        'cache_sync_times': [],
                        'prefill_total_times': [],
                        'comm_stats': []
                    }
                timing = result.get('timing', result)  # 兼容新旧格式
                rank_data[rank]['prefill_compute_times'].append(timing.get('prefill_compute_time', result.get('prefill_compute_time', 0)))
                rank_data[rank]['cache_sync_times'].append(timing.get('cache_sync_time', result.get('cache_sync_time', 0)))
                rank_data[rank]['prefill_total_times'].append(timing.get('prefill_total_time', result.get('prefill_total_time', 0)))
                if 'comm_stats' in result:
                    rank_data[rank]['comm_stats'].append(result['comm_stats'])
        
        # 打印每个rank的统计
        for rank in sorted(rank_data.keys()):
            data = rank_data[rank]
            
            compute_times = np.array(data['prefill_compute_times'])
            sync_times = np.array(data['cache_sync_times'])
            total_times = np.array(data['prefill_total_times'])
            
            print(f"\nRank {rank}:")
            print(f"  Prefill计算时间: {compute_times.mean():.3f}s ± {compute_times.std():.3f}s (min: {compute_times.min():.3f}s, max: {compute_times.max():.3f}s)")
            print(f"  Cache同步时间:  {sync_times.mean():.3f}s ± {sync_times.std():.3f}s (min: {sync_times.min():.3f}s, max: {sync_times.max():.3f}s)")
            print(f"  Prefill总时间:   {total_times.mean():.3f}s ± {total_times.std():.3f}s (min: {total_times.min():.3f}s, max: {total_times.max():.3f}s)")
            print(f"  通信开销占比:    {(sync_times.mean()/total_times.mean()*100):.1f}%")
            
            # 通信统计
            if data['comm_stats']:
                avg_stats = {
                    'token_sent': np.mean([s['token_sent'] for s in data['comm_stats']]),
                    'cache_sent': np.mean([s['cache_sent'] for s in data['comm_stats']]),
                    'cache_recv': np.mean([s['cache_recv'] for s in data['comm_stats']]),
                }
                print(f"  平均通信次数: token发送={avg_stats['token_sent']:.0f}, cache发送={avg_stats['cache_sent']:.0f}, cache接收={avg_stats['cache_recv']:.0f}")
                
                if mode == 'ring' and 'aggregated_sends' in data['comm_stats'][0]:
                    avg_agg = np.mean([s['aggregated_sends'] for s in data['comm_stats']])
                    avg_fwd = np.mean([s.get('forwarded_messages', 0) for s in data['comm_stats']])
                    print(f"  聚合发送次数: {avg_agg:.0f}, 转发消息数: {avg_fwd:.0f}")
    
    # 模式对比
    if 'p2p' in all_results and 'ring' in all_results:
        print(f"\n{'='*80}")
        print("P2P vs Ring 性能对比")
        print(f"{'='*80}")
        
        # 对比rank 0的平均时间
        p2p_rank0 = [r for test in all_results['p2p'] for r in test if r.get('rank') == 0 and 'error' not in r]
        ring_rank0 = [r for test in all_results['ring'] for r in test if r.get('rank') == 0 and 'error' not in r]
        
        if p2p_rank0 and ring_rank0:
            # 兼容新旧格式
            def get_timing(r, key):
                timing = r.get('timing', r)
                return timing.get(key, r.get(key, 0))
            
            p2p_compute = np.mean([get_timing(r, 'prefill_compute_time') for r in p2p_rank0])
            p2p_sync = np.mean([get_timing(r, 'cache_sync_time') for r in p2p_rank0])
            p2p_total = np.mean([get_timing(r, 'prefill_total_time') for r in p2p_rank0])
            
            ring_compute = np.mean([get_timing(r, 'prefill_compute_time') for r in ring_rank0])
            ring_sync = np.mean([get_timing(r, 'cache_sync_time') for r in ring_rank0])
            ring_total = np.mean([get_timing(r, 'prefill_total_time') for r in ring_rank0])
            
            print(f"\nRank 0 平均时间对比:")
            print(f"  {'指标':<20} {'P2P':>12} {'Ring':>12} {'差异':>12}")
            print(f"  {'-'*60}")
            print(f"  {'Prefill计算时间':<20} {p2p_compute:>10.3f}s {ring_compute:>10.3f}s {(ring_compute-p2p_compute):>+10.3f}s")
            print(f"  {'Cache同步时间':<20} {p2p_sync:>10.3f}s {ring_sync:>10.3f}s {(ring_sync-p2p_sync):>+10.3f}s")
            print(f"  {'Prefill总时间':<20} {p2p_total:>10.3f}s {ring_total:>10.3f}s {(ring_total-p2p_total):>+10.3f}s")
            print(f"  {'通信开销占比':<20} {(p2p_sync/p2p_total*100):>10.1f}% {(ring_sync/ring_total*100):>10.1f}% {((ring_sync/ring_total-p2p_sync/p2p_total)*100):>+10.1f}%")
            
            # 性能提升百分比
            if p2p_total > 0:
                speedup = (p2p_total - ring_total) / p2p_total * 100
                print(f"\nRing相比P2P的性能变化: {speedup:+.1f}%")
                if speedup > 0:
                    print(f"  ✓ Ring模式更快")
                elif speedup < 0:
                    print(f"  ✗ P2P模式更快")
                else:
                    print(f"  = 性能相当")


def save_results(all_results: Dict[str, List[List[Dict]]], output_dir: str):
    """保存结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数 - 使用持久worker架构"""
    
    # ==================== 配置参数 ====================
    BASE_CONFIG = {
        'model_path': '/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
        'base_port': 45000,
        'chunk_size': 128,
        'world_size': 3,
        'prompt': '请详细介绍一下人工智能的发展历史，包括其起源、重要里程碑和未来发展方向。',
        'gpu_ids': [5, 6, 7],
        'log_dir': os.path.join(project_dir, 'Communication-Optimize', 'benchmark_logs'),
        'startup_delay': 2.0,
        'node_addresses': None,
    }
    
    # 测试配置
    NUM_WARMUP = 1  # 预热次数
    NUM_TESTS = 1   # 每种模式测试次数
    TEST_INTERVAL = 3  # 测试间隔（秒）
    # ==================================================
    
    os.makedirs(BASE_CONFIG['log_dir'], exist_ok=True)
    
    print("="*80)
    print("ZMQ通信模式性能对比测试 (持久Worker架构)")
    print("="*80)
    print(f"模型路径: {BASE_CONFIG['model_path']}")
    print(f"进程数量: {BASE_CONFIG['world_size']}")
    print(f"Chunk大小: {BASE_CONFIG['chunk_size']}")
    print(f"使用GPU: {BASE_CONFIG['gpu_ids']}")
    print(f"预热次数: {NUM_WARMUP}")
    print(f"测试次数: {NUM_TESTS}")
    print(f"测试间隔: {TEST_INTERVAL}秒")
    print("="*80)
    
    mp.set_start_method('spawn', force=True)
    
    # 创建任务队列和结果队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 启动持久worker进程
    print(f"\n启动 {BASE_CONFIG['world_size']} 个worker进程...")
    workers = []
    for rank in range(BASE_CONFIG['world_size']):
        p = mp.Process(
            target=benchmark_worker,
            args=(rank, BASE_CONFIG, task_queue, result_queue)
        )
        p.start()
        workers.append(p)
        print(f"  Worker {rank} 已启动 (PID: {p.pid})")
    
    # 等待所有worker完成初始化（模型加载）
    print("\n等待workers完成初始化...")
    time.sleep(8)  # 给足够时间加载模型
    print("所有worker已就绪\n")
    
    all_results = {
        'p2p': [],
        'ring': []
    }
    
    # 对每种模式进行测试
    for mode in ['p2p', 'ring']:
        print(f"\n{'#'*80}")
        print(f"开始测试 {mode.upper()} 模式")
        print(f"{'#'*80}")
        
        config = BASE_CONFIG.copy()
        config['comm_mode'] = mode
        
        # 预热
        print(f"\n预热阶段 ({NUM_WARMUP}次)...")
        for i in range(NUM_WARMUP):
            run_test_iteration(task_queue, result_queue, config, i, is_warmup=True)
            if i < NUM_WARMUP - 1:
                time.sleep(TEST_INTERVAL)
        
        print(f"\n正式测试阶段 ({NUM_TESTS}次)...")
        # 正式测试
        for i in range(NUM_TESTS):
            results = run_test_iteration(task_queue, result_queue, config, i, is_warmup=False)
            if results and len(results) == BASE_CONFIG['world_size']:
                all_results[mode].append(results)
                print(f"✓ 第 {i+1} 次测试完成")
            else:
                print(f"✗ 第 {i+1} 次测试失败 (收到 {len(results) if results else 0}/{BASE_CONFIG['world_size']} 个结果)")
            
            # 测试间隔
            if i < NUM_TESTS - 1:
                time.sleep(TEST_INTERVAL)
        
        # 模式间切换间隔 - 给worker足够时间清理旧引擎
        if mode == 'p2p':
            print(f"\n等待 100 秒后切换到Ring模式...")
            print("(给workers足够时间清理P2P引擎、释放GPU内存和端口)")
            time.sleep(100)
    
    # 发送停止信号
    print(f"\n发送停止信号给所有worker...")
    for _ in range(BASE_CONFIG['world_size']):
        task_queue.put('STOP')
    
    # 等待所有worker退出
    print("等待worker退出...")
    for rank, worker in enumerate(workers):
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"  强制终止 Worker {rank}")
            worker.terminate()
        else:
            print(f"  Worker {rank} 已退出")
    
    # 分析结果
    analyze_results(all_results)
    
    # 保存结果
    save_results(all_results, os.path.join(project_dir, 'Communication-Optimize', 'benchmark_results'))
    
    # 清理端口
    cleanup_ports(BASE_CONFIG['base_port'], BASE_CONFIG['world_size'])
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
