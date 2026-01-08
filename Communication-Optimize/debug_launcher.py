"""
调试启动器 - 单机多卡模式
用于在VSCode中方便地调试分布式推理程序
"""

import os
import sys
import subprocess
import multiprocessing as mp
from distributed_inference import DistributedInferenceEngine


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


def run_rank(rank: int, world_size: int, args_dict: dict, gpu_ids: list):
    """运行单个rank的推理"""
    try:
        # 设置GPU环境变量
        gpu_id = gpu_ids[rank]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"\n{'='*60}")
        print(f"启动 Rank {rank} (使用GPU {gpu_id})")
        print(f"{'='*60}\n")
        
        # 创建推理引擎
        engine = DistributedInferenceEngine(
            model_path=args_dict['model_path'],
            rank=rank,
            world_size=world_size,
            master_addr=args_dict['master_addr'],
            master_port=args_dict['master_port'],
            chunk_size=args_dict['chunk_size'],
            sync_strategy=args_dict['sync_strategy'],
            device_mode=args_dict['device_mode']
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
        print(f"Rank {rank} 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
        'world_size': 3,
        'prompt': '请详细介绍一下人工智能的发展历史。',
        'max_new_tokens': 50,  # 调试时用较小的值
        'gpu_ids': [5, 6, 7],  # 指定使用的GPU ID，长度必须等于world_size
    }
    
    # ==================================================
    
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
    print(f"同步策略: {CONFIG['sync_strategy']}")
    print(f"进程数量: {CONFIG['world_size']}")
    print(f"Chunk大小: {CONFIG['chunk_size']}")
    print(f"使用GPU: {CONFIG['gpu_ids']}")
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
                args=(rank, CONFIG['world_size'], CONFIG, CONFIG['gpu_ids'])
            )
            p.start()
            processes.append(p)
            print(f"已启动 Rank {rank} (PID: {p.pid}, GPU: {CONFIG['gpu_ids'][rank]})")
        
        print("\n所有进程已启动，等待完成...\n")
        
        # 等待所有进程完成
        for rank, p in enumerate(processes):
            p.join()
            if p.exitcode == 0:
                print(f"✓ Rank {rank} 成功完成")
            else:
                print(f"✗ Rank {rank} 失败 (退出码: {p.exitcode})")
        
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
