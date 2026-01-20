# coding=utf-8
"""
SpecSoT 分布式推理启动器

该脚本用于启动多进程分布式推理，支持：
1. 多GPU分布式层拆分推理
2. ZMQ通信协调
3. 自动进程管理

使用示例：
    # 3台设备（GPU 0, 1, 2）运行分布式推理
    python run_specsot_distributed.py \
        --world_size 3 \
        --layer_splits 14,28 \
        --gpu_ids 0,1,2 \
        --task planning

    # 指定模型路径
    python run_specsot_distributed.py \
        --world_size 2 \
        --layer_splits 18 \
        --base_model_path /path/to/model \
        --eagle_model_path /path/to/eagle
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import logging
import threading
from datetime import datetime
from typing import List, Optional

import random
import numpy as np
import torch

# 获取项目根目录
project_dir = os.path.abspath(os.path.dirname(__file__))

# 保存子进程引用以便清理
child_processes: List[subprocess.Popen] = []
# 保存输出转发线程，确保主进程退出前可以等待
stream_threads: List[threading.Thread] = []


def setup_logging(rank: int, log_dir: str = None):
    """设置日志系统"""
    if log_dir is None:
        log_dir = os.path.join(project_dir, 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rank_{rank}_{timestamp}.log")
    
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
    
    print(f"已清理端口范围: {min(ports_to_clean)} - {max(ports_to_clean)}")


def cleanup_processes():
    """清理所有子进程"""
    for proc in child_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("[Launcher] 所有子进程已清理")


def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n[Launcher] 收到信号 {signum}，正在清理...")
    cleanup_processes()
    sys.exit(0)


def run_rank_subprocess(
    rank: int,
    world_size: int,
    layer_splits: str,
    gpu_id: int,
    base_model_path: str,
    eagle_model_path: str,
    task: str,
    max_new_tokens: int,
    num_samples: int,
    base_port: int,
    comm_mode: str,
    chunk_size: int,
    log_dir: str,
    seed: int = 42,
    disable_parallel: bool = False,
):
    """使用子进程方式运行单个rank"""
    script_path = os.path.join(project_dir, "run_specsot.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"找不到运行脚本: {script_path}")
    
    cmd = [
        sys.executable, script_path,
        "--distributed",
        "--rank", str(rank),
        "--world_size", str(world_size),
        "--layer_splits", layer_splits,
        "--base_port", str(base_port),
        "--comm_mode", comm_mode,
        "--chunk_size", str(chunk_size),
        "--base_model_path", base_model_path,
        "--eagle_model_path", eagle_model_path,
        "--task", task,
        "--max_new_tokens", str(max_new_tokens),
        "--num_samples", str(num_samples),
        "--seed", str(seed + rank),  # 每个rank使用不同的种子
    ]
    
    # 默认启用并行，只有明确禁用才不传递
    if not disable_parallel:
        cmd.append("--enable_parallel")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log_file = os.path.join(log_dir, f"rank_{rank}.log")
    
    print(f"[Launcher] 启动 Rank {rank} (GPU {gpu_id}, Seed {seed + rank})...")
    print(f"           日志: {log_file}")
    
    # 同时输出到控制台和文件
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    child_processes.append(proc)

    def _stream_output(p: subprocess.Popen, logfile: str, r: int):
        with open(logfile, "w") as f:
            for line in p.stdout:
                sys.stdout.write(f"[Rank {r}] {line}")
                f.write(line)
                f.flush()
            p.stdout.close()

    t = threading.Thread(target=_stream_output, args=(proc, log_file, rank), daemon=True)
    t.start()
    stream_threads.append(t)
    
    return proc


def launch_distributed(args):
    """启动分布式推理（支持多台设备）"""
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    if len(gpu_ids) != args.world_size:
        raise ValueError(f"GPU数量 ({len(gpu_ids)}) 必须等于 world_size ({args.world_size})")
    
    splits = [int(x.strip()) for x in args.layer_splits.split(",") if x.strip()]
    if len(splits) != args.world_size - 1:
        raise ValueError(f"layer_splits长度必须为 world_size - 1 = {args.world_size - 1}")
    
    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("SpecSoT 分布式推理启动器")
    print("=" * 60)
    print(f"World Size: {args.world_size}")
    print(f"Layer Splits: {args.layer_splits}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Base Port: {args.base_port}")
    print(f"Comm Mode: {args.comm_mode}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 清理端口
    cleanup_ports(args.base_port, args.world_size)
    time.sleep(2)
    
    # 启动各rank的进程
    for rank in range(args.world_size):
        run_rank_subprocess(
            rank=rank,
            world_size=args.world_size,
            layer_splits=args.layer_splits,
            gpu_id=gpu_ids[rank],
            base_model_path=args.base_model_path,
            eagle_model_path=args.eagle_model_path,
            task=args.task,
            disable_parallel=args.disable_parallel,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            base_port=args.base_port,
            comm_mode=args.comm_mode,
            chunk_size=args.chunk_size,
            log_dir=log_dir,
            seed=args.seed,
        )
        time.sleep(1.0)
    
    print(f"\n[Launcher] 所有进程已启动，等待完成...")
    print(f"           查看日志: {log_dir}/rank_*.log")
    
    try:
        for i, proc in enumerate(child_processes):
            proc.wait()
            if proc.returncode == 0:
                print(f"✓ Rank {i} 成功完成")
            else:
                print(f"✗ Rank {i} 失败 (退出码: {proc.returncode})")
    except KeyboardInterrupt:
        print("\n[Launcher] 收到中断信号，正在清理...")
        cleanup_processes()
        sys.exit(0)

    # 确保输出转发线程结束
    for t in stream_threads:
        t.join(timeout=1.0)
    
    success = all(proc.returncode == 0 for proc in child_processes)
    if success:
        print("\n" + "=" * 60)
        print("✓ 分布式推理完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 部分进程失败，请检查日志文件")
        print("=" * 60)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SpecSoT 分布式推理启动器")
    
    # 分布式配置
    parser.add_argument(
        "--world_size",
        type=int,
        default=3,
        help="总进程数（设备数）"
    )
    parser.add_argument(
        "--layer_splits",
        type=str,
        default="12,24",
        help="层拆分策略，如 '14,28' 表示3台设备拆分36层模型"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="5,6,7",
        help="GPU设备ID列表，逗号分隔，如 '0,1,2'"
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=45000,
        help="ZMQ通信基础端口"
    )
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="p2p",
        choices=["p2p", "ring"],
        help="通信模式"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Sequence Parallel的chunk大小"
    )
    
    # 模型配置
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B",
        help="Base Model 路径"
    )
    parser.add_argument(
        "--eagle_model_path",
        type=str,
        default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3",
        help="Eagle Model 路径"
    )
    
    # 任务配置
    parser.add_argument(
        "--task",
        type=str,
        default="planning",
        choices=["retrieval", "planning", "multi-doc-qa"],
        help="评估任务类型"
    )
    parser.add_argument(
        "--disable_parallel",
        action="store_true",
        help="禁用骨架并行模式（默认启用）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10000,
        help="最大生成token数"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="测试样本数量"
    )
    
    # 随机种子配置
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子基础值（每个rank会使用seed+rank作为实际种子）"
    )
    
    args = parser.parse_args()
    
    launch_distributed(args)


if __name__ == "__main__":
    main()
