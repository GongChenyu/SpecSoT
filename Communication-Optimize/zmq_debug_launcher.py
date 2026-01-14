"""
ZMQ版本的调试启动器 - 单机多进程模式
用于在VSCode中方便地调试基于ZMQ的分布式推理程序
"""

import os
import sys
import subprocess
import multiprocessing as mp
import logging
from datetime import datetime

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_logging(rank: int, log_dir: str = "./logs"):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"zmq_rank_{rank}_{timestamp}.log")
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    logger.info(f"日志将保存到: {log_file}")
    return log_file


def cleanup_ports(base_port: int, world_size: int):
    """清理被占用的端口"""
    # 计算所有可能使用的端口
    ports_to_clean = set()
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                port = base_port + sender * world_size + receiver
                ports_to_clean.add(port)
    
    for port in ports_to_clean:
        try:
            result = subprocess.run(
                f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
                shell=True,
                capture_output=True,
                text=True
            )
        except Exception as e:
            pass
    
    print(f"已清理端口范围: {min(ports_to_clean)} - {max(ports_to_clean)}")


def run_rank(rank: int, args_dict: dict):
    """运行单个rank的推理"""
    logger = None
    try:
        # 设置GPU环境变量
        gpu_ids = args_dict['gpu_ids']
        gpu_id = gpu_ids[rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 设置日志
        log_file = setup_logging(rank, log_dir=args_dict.get('log_dir', './logs'))
        logger = logging.getLogger()
        
        print(f"\n{'='*60}")
        print(f"启动 Rank {rank} (使用GPU {gpu_id})")
        print(f"日志文件: {log_file}")
        print(f"{'='*60}\n")
        
        # 导入并创建推理引擎
        from zmq_distributed_inference import ZMQDistributedInferenceEngine
        
        engine = ZMQDistributedInferenceEngine(
            model_path=args_dict['model_path'],
            rank=rank,
            world_size=args_dict['world_size'],
            base_port=args_dict['base_port'],
            chunk_size=args_dict['chunk_size'],
            comm_mode=args_dict['comm_mode'],
            node_addresses=args_dict.get('node_addresses'),
            startup_delay=args_dict.get('startup_delay', 3.0)
        )
        
        # 运行推理
        result = engine.run_inference(
            args_dict['prompt'], 
            args_dict['max_new_tokens']
        )
        
        print(f"\n{'='*60}")
        print(f"Rank {rank} 完成")
        print(f"{'='*60}\n")
        
    except Exception as e:
        error_msg = f"Rank {rank} 出错: {e}"
        print(error_msg)
        if logger:
            logger.error(error_msg, exc_info=True)
            for handler in logger.handlers:
                handler.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if logger:
            for handler in logger.handlers:
                handler.flush()


def main():
    """主函数 - 启动所有rank进程"""
    
    # ==================== 配置参数 ====================
    CONFIG = {
        'model_path': '/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
        'base_port': 45000,  # ZMQ使用的基础端口
        'chunk_size': 128,
        'comm_mode': 'ring',  # 'p2p' 或 'ring'
        'world_size': 3,
        'prompt': '请详细介绍一下人工智能的发展历史。',
        'max_new_tokens': 50,
        'gpu_ids': [5, 6, 7],  # 使用的GPU ID
        'log_dir': os.path.join(project_dir, 'Communication-Optimize', 'logs'),
        'startup_delay': 1.0,  # 节点启动延迟
        'node_addresses': None,  # 单机模式下使用默认的localhost
    }
    # ==================================================
    
    os.makedirs(CONFIG.get('log_dir', './logs'), exist_ok=True)
    
    if len(CONFIG['gpu_ids']) != CONFIG['world_size']:
        print(f"\n错误：GPU数量({len(CONFIG['gpu_ids'])})必须等于world_size({CONFIG['world_size']})")
        sys.exit(1)
    
    # 清理端口
    print("\n正在清理端口...")
    cleanup_ports(CONFIG['base_port'], CONFIG['world_size'])
    
    # 等待端口完全释放
    import time
    print("等待端口释放...")
    time.sleep(2)
    
    print("\n" + "="*60)
    print("ZMQ分布式推理调试启动器")
    print("="*60)
    print(f"模型路径: {CONFIG['model_path']}")
    print(f"通信模式: {CONFIG['comm_mode']}")
    print(f"基础端口: {CONFIG['base_port']}")
    print(f"进程数量: {CONFIG['world_size']}")
    print(f"Chunk大小: {CONFIG['chunk_size']}")
    print(f"使用GPU: {CONFIG['gpu_ids']}")
    print(f"启动延迟: {CONFIG['startup_delay']}秒")
    print("="*60 + "\n")
    
    mp.set_start_method('spawn', force=True)
    
    processes = []
    
    try:
        for rank in range(CONFIG['world_size']):
            p = mp.Process(
                target=run_rank,
                args=(rank, CONFIG)
            )
            p.start()
            processes.append(p)
            print(f"已启动 Rank {rank} (PID: {p.pid}, GPU: {CONFIG['gpu_ids'][rank]})")
        
        print("\n所有进程已启动，等待完成...\n")
        
        timeout = 6000  # 1分钟超时
        import time
        start_time = time.time()
        
        for rank, p in enumerate(processes):
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                print(f"\n⚠ 超时！强制终止所有进程...")
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                break
            
            p.join(timeout=remaining_time)
            
            if p.is_alive():
                print(f"⚠ Rank {rank} 超时，强制终止")
                p.terminate()
                p.join(timeout=5)
            elif p.exitcode == 0:
                print(f"✓ Rank {rank} 成功完成")
            else:
                print(f"✗ Rank {rank} 失败 (退出码: {p.exitcode})")
                for other_rank, other_p in enumerate(processes):
                    if other_rank != rank and other_p.is_alive():
                        other_p.terminate()
        
        print("\n" + "="*60)
        print("所有进程已完成")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在终止所有进程...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("所有进程已终止")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n启动器出错: {e}")
        import traceback
        traceback.print_exc()
        
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
