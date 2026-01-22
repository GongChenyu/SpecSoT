# coding=utf-8
"""
SpecSoT Unified Inference Script
集成了 单机推理、分布式Launcher调度 和 分布式Worker执行 功能。

功能特性：
1. 单机单卡推理 (--distributed False)
2. 单机多卡分布式推理 (--role launcher)
3. 分布式Worker执行 (--role worker)
4. 完整的日志记录（同时输出到控制台和文件）
5. GPU显存监控
6. 详细的统计信息输出

使用示例：
    # 单机单卡
    python run_specsot_final.py --distributed False

    # 单机多卡分布式 (3 GPUs)
    python run_specsot_final.py --world_size 3 --gpu_ids 5,6,7 --layer_splits 14,28
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import logging
import threading
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
import pynvml
from threading import Thread, Event

# 引入项目依赖 (假设 SpecSoT 包在 python path 下)
from SpecSoT import SpecSoTModel
from SpecSoT.distributed.distributed_config import DistributedConfig
from SpecSoT.logging_utils import (
    FlushingStreamHandler,
    FlushingFileHandler,
    get_unified_logger,
    cleanup_loggers,
)

# 获取项目根目录
project_dir = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# 日志配置
# =============================================================================


def setup_logging(rank: int = -1, log_dir: str = None) -> logging.Logger:
    """
    设置日志系统，同时输出到控制台和文件
    
    使用统一的日志模块，确保日志实时显示，解决输出延迟问题
    
    注意：
    - 控制台只显示 INFO 级别以上的关键信息
    - 文件记录所有 DEBUG 级别的详细信息
    
    Args:
        rank: 当前进程的rank，-1表示单机模式或launcher
        log_dir: 日志目录
        
    Returns:
        配置好的logger对象
    """
    if log_dir is None:
        log_dir = os.path.join(project_dir, 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    if rank >= 0:
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
        logger_name = f"SpecSoT-Rank{rank}"
    else:
        log_file = os.path.join(log_dir, f"launcher.log")
        logger_name = "SpecSoT-Launcher"
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # 获取或创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # 文件处理器（带刷新）- 记录所有级别
    file_handler = FlushingFileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 控制台处理器（带刷新）- 只记录INFO以上
    console_handler = FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    logger.info(f"日志将保存到: {log_file}")
    
    return logger


def cleanup_ports(base_port: int, world_size: int, logger: logging.Logger = None):
    """
    清理被占用的端口
    
    Args:
        base_port: 基础端口号
        world_size: 总进程数
        logger: 日志记录器
    """
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
    
    msg = f"已清理端口范围: {min(ports_to_clean)} - {max(ports_to_clean)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# =============================================================================
# 工具类 (GPUMemoryMonitor)
# =============================================================================

class GPUMemoryMonitor:
    """
    GPU 显存监控器
    
    使用上下文管理器模式，在推理过程中后台监控显存峰值。
    
    Example:
        >>> with GPUMemoryMonitor(device_index=0) as monitor:
        ...     # 执行推理
        ...     pass
        >>> print(f"Peak memory: {monitor.peak_usage:.2f} MB")
    """
    
    def __init__(self, device_index: int = None, interval: float = 0.01):
        """
        Args:
            device_index: GPU 设备索引
            interval: 采样间隔（秒）
        """
        if device_index is None and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        self.device_index = device_index
        self.interval = interval
        self.peak_usage = 0
        self.monitor_thread = None
        self.stop_event = Event()

    def _monitor(self):
        """后台监控线程"""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            while not self.stop_event.is_set():
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                current_usage = info.used / (1024 ** 2)  # 转换为 MB
                if current_usage > self.peak_usage:
                    self.peak_usage = current_usage
                time.sleep(self.interval)
            pynvml.nvmlShutdown()
        except Exception:
            pass  # 避免非NVIDIA环境报错

    def __enter__(self):
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return False


# =============================================================================
# 数据加载
# =============================================================================

def load_dataset(task: str, logger: logging.Logger = None) -> pd.DataFrame:
    """
    加载评估数据集
    
    Args:
        task: 任务类型 ["retrieval", "planning", "multi-doc-qa"]
        logger: 日志记录器
        
    Returns:
        DataFrame: 包含 task_prompt 列的数据集
    """
    if task == "retrieval":
        df_path = os.path.join(project_dir, "datasets/student_resume_logic_retrieval", "logic_gpa_resume_10.jsonl")
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['prompt']
    elif task == "planning":
        df_path = os.path.join(project_dir, "datasets/planning", "industry_tasks_no_ascii.jsonl")
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['task']
    else:  # multi-doc-qa
        df_path = os.path.join(project_dir, "datasets/multi-doc-qa", "2wikimqa.jsonl")
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['input']
    
    msg = f"加载数据集: {df_path}, 共 {len(df)} 条样本"
    if logger:
        logger.info(msg)
    else:
        print(msg)
        
    return df


def get_special_token_ids(tokenizer) -> dict:
    """
    获取骨架解析所需的特殊 token IDs
    
    骨架格式：
    ####标题1:...
    ####标题2:...
    ####%%%%
    
    Args:
        tokenizer: 分词器
        
    Returns:
        dict: 特殊 token ID 映射
    """
    return {
        "para_begin_token_id": tokenizer.encode("####")[0],
        "para_end_token_id": tokenizer.encode("%%%%")[0],
        "ellipsis_token_id": tokenizer.encode("......")[0],
        "half_ellipsis_token_id": tokenizer.encode("...")[0],
        "line_break_token_id": tokenizer.encode("\n\n")[0],
        "colon_token_id": tokenizer.encode(":")[0],
        "cn_colon_token_id": tokenizer.encode("：")[0],  # 中文冒号
        "colon_new_line_token_id": tokenizer.encode(":\n")[0],
    }

# =============================================================================
# Worker 逻辑 (执行实际的推理任务)
# =============================================================================

def run_worker(args):
    """
    执行实际的推理任务
    
    该函数在每个GPU设备上执行，负责：
    1. 初始化分布式配置
    2. 加载模型
    3. 执行推理
    4. 保存结果 (仅 Rank 0)
    """
    
    # 设置日志
    log_dir = os.path.join(project_dir, 'logs')
    logger = setup_logging(rank=args.rank, log_dir=log_dir)
    
    # # 设置随机种子
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 60)
    logger.info(f"SpecSoT Worker 启动")
    logger.info("=" * 60)
    logger.info(f"Base Model: {args.base_model_path}")
    logger.info(f"Eagle Model: {args.eagle_model_path}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Enable Parallel: {args.enable_parallel}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")
    logger.info(f"Random Seed: {args.seed}")
    
    if args.distributed:
        logger.info(f"Distributed Mode: rank={args.rank}/{args.world_size-1}")
        logger.info(f"Layer Splits: {args.layer_splits}")
        logger.info(f"Comm Mode: {args.comm_mode}")
        logger.info(f"Chunk Size: {args.chunk_size}")
        logger.info(f"Base Port: {args.base_port}")
    logger.info("=" * 60)

    # =========================================================================
    # 1. 分布式配置
    # =========================================================================
    distributed_config = None
    if args.distributed:
        if not args.layer_splits:
            raise ValueError("分布式模式必须指定 --layer_splits 参数")
        
        distributed_config = DistributedConfig.from_layer_splits_str(
            layer_splits_str=args.layer_splits,
            rank=args.rank,
            world_size=args.world_size,
            base_port=args.base_port,
            comm_mode=args.comm_mode,
            chunk_size=args.chunk_size,
        )
        logger.info(f"分布式配置: {distributed_config}")

    # =========================================================================
    # 2. 加载模型
    # =========================================================================
    logger.info(">>> 正在加载 SpecSoT 模型...")
    model_load_start = time.time()
    
    model = SpecSoTModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.eagle_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0" if args.distributed else "cuda:0", 
        total_token=40,
        depth=4,
        top_k=6,
        seed=args.seed,
        distributed_config=distributed_config,
    )
    
    model_load_time = time.time() - model_load_start
    logger.info(f"模型加载完成，耗时: {model_load_time:.2f}s")
    
    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"
    para_token_ids = get_special_token_ids(tokenizer)
    
    logger.info("Special Token IDs:")
    for key, value in para_token_ids.items():
        token_str = tokenizer.decode([value])
        logger.debug(f"  {key}: {value} -> '{token_str}'")

    # =========================================================================
    # 3. 加载数据集
    # =========================================================================
    df = load_dataset(args.task, logger)
    df = df[:args.num_samples]
    logger.info(f"使用 {len(df)} 条样本进行评估")

    # =========================================================================
    # 4. 执行推理
    # =========================================================================
    results = []
    monitor_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
    
    total_inference_start = time.time()

    for i in range(len(df)):
        # 获取任务prompt
        # task_prompt = df.loc[i, "task_prompt"]
        task_prompt = "请问打篮球时，如何提高投篮命中率？请给出详细的建议。"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"样本 {i+1}/{len(df)}: {task_prompt[:100]}...")
        logger.info("=" * 60)

        with GPUMemoryMonitor(device_index=monitor_device) as monitor:
            start_time = time.time()
            
            # 调用 SpecSoT 生成
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time = \
                model.generate(
                    task_prompt=task_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    enable_parallel=args.enable_parallel,
                    para_token_ids=para_token_ids,
                )
            
            total_time = time.time() - start_time

        # 解码响应
        response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        output_length = len(output_ids[0])
        throughput = output_length / total_time if total_time > 0 else 0

        # 打印统计信息
        logger.info(f"\n[统计信息]")
        logger.info(f"  总耗时: {total_time:.2f}s")
        logger.info(f"  峰值显存: {monitor.peak_usage:.2f} MB")
        logger.info(f"  输出长度: {output_length} tokens")
        logger.info(f"  吞吐量: {throughput:.1f} tokens/s")
        logger.info(f"  并行分支数: {num_para}")
        logger.info(f"  平均接受长度: {avg_accept_len:.2f}")
        logger.info(f"  平均Draft时间: {avg_draft_time*1000:.2f} ms")
        logger.info(f"  平均Verify时间: {avg_verify_time*1000:.2f} ms")
        logger.info(f"  平均Update时间: {avg_update_time*1000:.2f} ms")
        
        # 打印响应预览
        logger.info(f"\n[响应预览]")
        logger.info(response[:500] + "..." if len(response) > 500 else response)

        results.append({
            "prompt": task_prompt,
            "response": response,
            "time": total_time,
            "memory": monitor.peak_usage,
            "length": output_length,
            "throughput": throughput,
            "num_para": num_para,
            "avg_accept_len": avg_accept_len,
            "avg_draft_time": avg_draft_time,
            "avg_verify_time": avg_verify_time,
            "avg_update_time": avg_update_time,
        })
    
    total_inference_time = time.time() - total_inference_start

    # =========================================================================
    # 5. 保存结果 (仅 Rank 0 或单机模式)
    # =========================================================================
    if not args.distributed or args.rank == 0:
        save_dir = os.path.join(project_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        
        mode = "parallel" if args.enable_parallel else "standard"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_path = os.path.join(save_dir, f"results_specsot_{mode}_{args.task}_{timestamp}.json")
        save_path = os.path.join(save_dir, f"results_specsot_{mode}_{args.task}.json")
        pd.DataFrame(results).to_json(save_path, orient='records', indent=4, force_ascii=False)
        logger.info(f"\n结果已保存到: {save_path}")

        # 打印汇总统计
        logger.info("\n" + "=" * 60)
        logger.info("汇总统计")
        logger.info("=" * 60)
        
        avg_time = sum(r['time'] for r in results) / len(results)
        avg_length = sum(r['length'] for r in results) / len(results)
        avg_memory = sum(r['memory'] for r in results) / len(results)
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        avg_accept = sum(r['avg_accept_len'] for r in results) / len(results)
        
        logger.info(f"  总样本数: {len(results)}")
        logger.info(f"  总推理时间: {total_inference_time:.2f}s")
        logger.info(f"  平均单样本时间: {avg_time:.2f}s")
        logger.info(f"  平均输出长度: {avg_length:.1f} tokens")
        logger.info(f"  平均峰值显存: {avg_memory:.2f} MB")
        logger.info(f"  平均吞吐量: {avg_throughput:.1f} tokens/s")
        logger.info(f"  平均接受长度: {avg_accept:.2f}")
        logger.info("=" * 60)
    
    # 清理分布式资源
    if args.distributed and model.is_distributed():
        logger.info("正在清理分布式资源...")
        model.cleanup_distributed()
        logger.info("分布式资源已清理")
    
    logger.info("Worker 执行完成")

# =============================================================================
# Launcher 逻辑 (分布式调度器)
# =============================================================================

# 保存子进程引用以便清理
child_processes: List[subprocess.Popen] = []
# 保存输出转发线程，确保主进程退出前可以等待
stream_threads: List[threading.Thread] = []


def cleanup_processes(signum=None, frame=None):
    """清理所有子进程"""
    for proc in child_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
    print("[Launcher] 所有子进程已清理")
    if signum is not None:
        sys.exit(0)


def run_launcher(args):
    """
    作为 Launcher 启动多个 Worker 子进程
    
    该函数负责：
    1. 解析GPU配置和层拆分策略
    2. 清理被占用的端口
    3. 启动各个rank的Worker子进程
    4. 转发子进程输出到控制台和日志文件
    5. 等待所有进程完成
    """
    # 设置日志
    log_dir = os.path.join(project_dir, 'logs')
    logger = setup_logging(rank=-1, log_dir=log_dir)
    
    logger.info("=" * 60)
    logger.info("SpecSoT 分布式推理启动器")
    logger.info("=" * 60)
    
    # 解析 GPU 配置
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    if len(gpu_ids) != args.world_size:
        raise ValueError(f"GPU数量 ({len(gpu_ids)}) 必须等于 world_size ({args.world_size})")
    
    # 验证层拆分策略
    splits = [int(x.strip()) for x in args.layer_splits.split(",") if x.strip()]
    if len(splits) != args.world_size - 1:
        raise ValueError(f"layer_splits长度必须为 world_size - 1 = {args.world_size - 1}")
    
    logger.info(f"World Size: {args.world_size}")
    logger.info(f"Layer Splits: {args.layer_splits}")
    logger.info(f"GPU IDs: {gpu_ids}")
    logger.info(f"Base Port: {args.base_port}")
    logger.info(f"Comm Mode: {args.comm_mode}")
    logger.info(f"Chunk Size: {args.chunk_size}")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Enable Parallel: {args.enable_parallel}")
    logger.info(f"Max New Tokens: {args.max_new_tokens}")
    logger.info(f"Num Samples: {args.num_samples}")
    logger.info("=" * 60)

    # 注册信号处理
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)

    # 清理端口
    logger.info("正在清理端口...")
    cleanup_ports(args.base_port, args.world_size, logger)
    time.sleep(1)

    # 获取当前脚本的绝对路径
    current_script = os.path.abspath(__file__)

    # 启动各rank的进程
    for rank in range(args.world_size):
        # 构建子进程命令
        cmd = [
            sys.executable, current_script,
            "--role", "worker",  # 显式指定角色为 worker
            "--distributed", "True",
            "--rank", str(rank),
            "--world_size", str(args.world_size),
            "--layer_splits", args.layer_splits,
            "--base_port", str(args.base_port),
            "--comm_mode", args.comm_mode,
            "--chunk_size", str(args.chunk_size),
            "--base_model_path", args.base_model_path,
            "--eagle_model_path", args.eagle_model_path,
            "--task", args.task,
            "--max_new_tokens", str(args.max_new_tokens),
            "--num_samples", str(args.num_samples),
            "--seed", str(args.seed),
        ]
        
        if args.enable_parallel:
            cmd.append("--enable_parallel")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])

        log_file = os.path.join(log_dir, f"rank_{rank}.log")
        logger.info(f"启动 Rank {rank} (GPU {gpu_ids[rank]})...")
        logger.info(f"  日志文件: {log_file}")
        logger.debug(f"  命令: {' '.join(cmd)}")
        
        proc = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        child_processes.append(proc)

        # 输出转发线程
        def _stream_output(p, logfile, r, main_logger):
            with open(logfile, "w", encoding='utf-8') as f:
                for line in p.stdout:
                    # 写入日志文件
                    f.write(line)
                    f.flush()
                    # 输出到控制台（添加rank前缀）
                    sys.stdout.write(f"[Rank {r}] {line}")
                    sys.stdout.flush()
                p.stdout.close()
        
        t = threading.Thread(
            target=_stream_output, 
            args=(proc, log_file, rank, logger), 
            daemon=True
        )
        t.start()
        stream_threads.append(t)
        
        # 等待一小段时间，让进程初始化
        time.sleep(1.0)

    logger.info(f"\n所有进程已启动，等待完成...")
    logger.info(f"日志目录: {log_dir}")

    # 等待所有进程完成
    try:
        for i, proc in enumerate(child_processes):
            proc.wait()
            if proc.returncode == 0:
                logger.info(f"✓ Rank {i} 成功完成")
            else:
                logger.error(f"✗ Rank {i} 失败 (退出码: {proc.returncode})")
    except KeyboardInterrupt:
        logger.warning("\n收到中断信号，正在清理...")
        cleanup_processes()
        sys.exit(0)

    # 确保输出转发线程结束
    for t in stream_threads:
        t.join(timeout=2.0)
    
    # 检查结果
    success = all(proc.returncode == 0 for proc in child_processes)
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✓ 分布式推理完成！")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("✗ 部分进程失败，请检查日志文件")
        logger.error("=" * 60)
        sys.exit(1)

# =============================================================================
# 统一入口
# =============================================================================

def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(
        description="SpecSoT Unified Runner - 统一的推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单机单卡推理
  python run_specsot_final.py --distributed False

  # 单机多卡分布式推理 (3 GPUs)
  python run_specsot_final.py --world_size 3 --gpu_ids 5,6,7 --layer_splits 14,28

  # 作为分布式Worker运行
  python run_specsot_final.py --role worker --rank 0 --world_size 3 --layer_splits 14,28
        """
    )
    
    # 角色控制
    parser.add_argument("--role", type=str, default="launcher", choices=["launcher", "worker"], help="运行角色: launcher(调度器) 或 worker(计算节点)")
    
    # 模型配置
    parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B", help="Base Model 路径")
    parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3", help="Eagle Model 路径")
    parser.add_argument("--enable_parallel", action="store_true", default=True, help="启用骨架并行模式")
    parser.add_argument("--max_new_tokens", type=int, default=3000,help="最大生成token数")
    
    # 任务配置
    parser.add_argument("--task", type=str, default="planning", choices=["retrieval", "planning", "multi-doc-qa"],help="评估任务类型")
    parser.add_argument("--num_samples", type=int, default=1,help="测试样本数量")
    
    # 分布式配置
    parser.add_argument("--distributed", type=str2bool, default=True, help="是否启用分布式模式（单机单卡设为False）")
    parser.add_argument("--world_size", type=int, default=3, help="总进程数（设备数）")
    parser.add_argument("--rank", type=int, default=0, help="当前进程的rank")
    parser.add_argument("--layer_splits", type=str, default="14,28", help="层拆分策略，如 '14,28' 表示3台设备拆分36层模型")
    parser.add_argument("--gpu_ids", type=str, default="1,2,3", help="GPU设备ID列表，逗号分隔，如 '0,1,2'")
    parser.add_argument("--base_port", type=int, default=45000, help="ZMQ通信基础端口")
    parser.add_argument("--comm_mode", type=str, default="p2p", choices=["p2p", "ring"], help="通信模式")
    parser.add_argument("--chunk_size", type=int, default=128, help="Sequence Parallel的chunk大小")
    
    # 其他配置
    parser.add_argument("--seed", type=int, default=76, help="随机种子")  # 66 work 

    args = parser.parse_args()

    # 逻辑分流
    if args.role == "worker" or (not args.distributed):
        # 如果显式指定为 worker，或者是单机模式，直接运行 worker 逻辑
        run_worker(args)
    elif args.role == "launcher" and args.distributed:
        # 如果是分布式模式且作为入口运行，则启动 Launcher
        run_launcher(args)
    else:
        print(f"未知的运行模式: role={args.role}, distributed={args.distributed}")
        sys.exit(1)


if __name__ == "__main__":
    main()


