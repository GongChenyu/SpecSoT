"""
调试启动器 - 单机多卡模式
用于在VSCode中方便地调试分布式推理程序
"""

import os
import sys
import subprocess
import multiprocessing as mp
import logging
from datetime import datetime
from distributed_inference import DistributedInferenceEngine


def setup_logging(rank: int, log_dir: str = "./logs"):
    """
    设置日志系统，将日志同时输出到控制台和文件
    
    Args:
        rank: 当前进程的rank
        log_dir: 日志文件保存目录
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 获取root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 添加文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 添加控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    logger.info(f"日志将保存到: {log_file}")
    return log_file


def cleanup_port(port: int):
    """清理被占用的端口"""
    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"端口 {port} 已清理")
    except Exception as e:
        print(f"清理端口 {port} 失败: {e}")


def run_rank(rank: int, args_dict: dict):
    """运行单个rank的推理"""
    logger = None
    try:
        # 设置GPU环境变量
        gpu_ids = args_dict['gpu_ids']
        gpu_id = gpu_ids[rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 设置日志（在初始化引擎之前）
        log_file = setup_logging(rank, log_dir=args_dict.get('log_dir', './logs'))
        logger = logging.getLogger()
        
        print(f"\n{'='*60}")
        print(f"启动 Rank {rank} (使用GPU {gpu_id})")
        print(f"日志文件: {log_file}")
        print(f"{'='*60}\n")
        
        # 创建推理引擎
        engine = DistributedInferenceEngine(
            model_path=args_dict['model_path'],
            rank=rank,
            world_size=args_dict['world_size'],
            master_addr=args_dict['master_addr'],
            master_port=args_dict['master_port'],
            chunk_size=args_dict['chunk_size'],
            sync_strategy=args_dict['sync_strategy'],
            device_mode=args_dict['device_mode'],
            backend=args_dict['backend']
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
            # 确保日志被写入
            for handler in logger.handlers:
                handler.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 确保日志被刷新
        if logger:
            for handler in logger.handlers:
                handler.flush()


def main():
    """主函数 - 启动所有rank进程"""
    
    # ==================== 配置参数 ====================
    # 在这里修改参数以适应你的调试需求
    
    CONFIG = {
        'model_path': '/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
        'master_addr': 'localhost',
        'master_port': '29500',
        'chunk_size': 128,
        'sync_strategy': 'pairwise',  # 'pairwise' 或 'ring'
        'device_mode': 'single_node',  # 单机多卡
        'backend': 'gloo',  # 'auto', 'nccl', 或 'gloo'
        'world_size': 3,
        'prompt': '请详细介绍一下人工智能的发展历史。',
        'max_new_tokens': 50,  # 调试时用较小的值
        'gpu_ids': [5, 6, 7],  # 指定使用的GPU ID，长度必须等于world_size
        'log_dir': './logs',  # 日志保存目录
    }
    
    # ==================================================
    
    # 创建日志目录
    os.makedirs(CONFIG.get('log_dir', './logs'), exist_ok=True)
    
    # 验证GPU配置
    if len(CONFIG['gpu_ids']) != CONFIG['world_size']:
        print(f"\n错误：GPU数量({len(CONFIG['gpu_ids'])})必须等于world_size({CONFIG['world_size']})")
        sys.exit(1)
    
    # 清理端口
    print("\n正在清理端口...")
    cleanup_port(int(CONFIG['master_port']))
    
    print("\n" + "="*60)
    print("分布式推理调试启动器")
    print("="*60)
    print(f"模型路径: {CONFIG['model_path']}")
    print(f"设备模式: {CONFIG['device_mode']}")    
    print(f"通信后端: {CONFIG['backend']}")    
    print(f"同步策略: {CONFIG['sync_strategy']}")
    print(f"进程数量: {CONFIG['world_size']}")
    print(f"Chunk大小: {CONFIG['chunk_size']}")
    print(f"使用GPU: {CONFIG['gpu_ids']}")
    print(f"日志目录: {CONFIG.get('log_dir', './logs')}")
    print("="*60 + "\n")
    
    # 设置启动方法为spawn（更稳定）
    mp.set_start_method('spawn', force=True)
    
    # 创建进程列表
    processes = []
    
    try:
        # 启动所有rank进程
        for rank in range(CONFIG['world_size']):
            p = mp.Process(
                target=run_rank,
                args=(rank, CONFIG)
            )
            p.start()
            processes.append(p)
            print(f"已启动 Rank {rank} (PID: {p.pid}, GPU: {CONFIG['gpu_ids'][rank]})")
        
        print("\n所有进程已启动，等待完成...\n")
        
        # 等待所有进程完成（添加超时机制）
        timeout = 3000  # 5分钟超时
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
                # 如果一个进程失败，终止其他进程避免挂起
                print(f"  终止其他进程以避免挂起...")
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
        
        # 终止所有进程
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
