"""
调试启动器 - Ring同步策略
用于调试Ring cache同步策略
"""

import os
import sys
import multiprocessing as mp
from distributed_inference import DistributedInferenceEngine


def run_rank(rank: int, world_size: int, args_dict: dict):
    """运行单个rank的推理"""
    try:
        print(f"\n{'='*60}")
        print(f"启动 Rank {rank} (Ring同步)")
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
    
    CONFIG = {
        'model_path': '/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
        'master_addr': 'localhost',
        'master_port': '29501',  # 使用不同的端口避免冲突
        'chunk_size': 128,
        'sync_strategy': 'ring',  # Ring同步策略
        'device_mode': 'single_node',
        'world_size': 3,
        'prompt': '请详细介绍一下人工智能的发展历史。',
        'max_new_tokens': 50
    }
    
    # ==================================================
    
    print("\n" + "="*60)
    print("分布式推理调试启动器 - Ring同步策略")
    print("="*60)
    print(f"模型路径: {CONFIG['model_path']}")
    print(f"设备模式: {CONFIG['device_mode']}")
    print(f"同步策略: {CONFIG['sync_strategy']}")
    print(f"进程数量: {CONFIG['world_size']}")
    print(f"Chunk大小: {CONFIG['chunk_size']}")
    print("="*60 + "\n")
    
    mp.set_start_method('spawn', force=True)
    
    processes = []
    
    try:
        for rank in range(CONFIG['world_size']):
            p = mp.Process(
                target=run_rank,
                args=(rank, CONFIG['world_size'], CONFIG)
            )
            p.start()
            processes.append(p)
            print(f"已启动 Rank {rank} (PID: {p.pid})")
        
        print("\n所有进程已启动，等待完成...\n")
        
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
        
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
