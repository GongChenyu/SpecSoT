# coding=utf-8
"""
SpecSoT Master Engine - 主控引擎

该模块实现 Master 的核心逻辑：
1. 分布式模式：启动和管理 Worker 子进程
2. 单机模式：直接执行推理

执行模式：
- 单机模式 (distributed=False): Master 直接执行推理
- 分布式模式 (distributed=True): Master 启动多个 Worker 子进程

调度功能通过 use_scheduling 参数控制，与分布式正交。
"""

import os
import sys
import time
import signal
import threading
import subprocess
from typing import List, Optional
from dataclasses import dataclass

from utils.utils import DeviceConfig, parse_devices, cleanup_ports


@dataclass
class MasterConfig:
    """Master 配置"""
    distributed: bool
    use_scheduling: bool
    world_size: int
    devices: str
    layer_splits: str
    base_port: int
    comm_mode: str
    chunk_size: int
    
    # 模型配置
    base_model_path: str
    eagle_model_path: str
    use_eagle3: bool
    enable_parallel: bool
    use_semantic_constraint: bool
    max_new_tokens: int
    
    # 任务配置
    task: str
    num_samples: int
    seed: int
    max_parallel: int


class MasterEngine:
    """
    Master 引擎 - 负责系统协调
    
    功能：
    1. 分布式模式：启动和管理 Worker 子进程
    2. 单机模式：直接执行推理（委托给 WorkerEngine）
    
    执行模式组合：
    - distributed=False, use_scheduling=False: 纯单机推理
    - distributed=False, use_scheduling=True:  单机 + 调度
    - distributed=True,  use_scheduling=False: 分布式推理
    - distributed=True,  use_scheduling=True:  分布式 + 调度
    """

    def __init__(self, args):
        """
        初始化 Master 引擎
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.project_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.log_dir = os.path.join(self.project_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 子进程管理（仅分布式模式使用）
        self.child_processes: List[subprocess.Popen] = []
        self.stream_threads: List[threading.Thread] = []
        
        # 分布式配置
        if args.distributed:
            self.device_configs = parse_devices(args.devices)
            self.world_size = len(self.device_configs)
            self._validate_distributed_config()
            
            # 注册信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            self.device_configs = None
            self.world_size = 1

    def _validate_distributed_config(self):
        """验证分布式配置"""
        if self.world_size < 2:
            raise ValueError(f"分布式模式至少需要 2 个设备，当前: {self.world_size}")
        
        splits = [int(x.strip()) for x in self.args.layer_splits.split(",") if x.strip()]
        expected_splits = self.world_size - 1
        if len(splits) != expected_splits:
            raise ValueError(
                f"layer_splits 数量不匹配: 需要 {expected_splits} 个分割点，"
                f"实际 {len(splits)} 个"
            )

    def _signal_handler(self, signum=None, frame=None):
        """信号处理：清理子进程"""
        self._cleanup_processes()
        if signum is not None:
            sys.exit(0)

    def _cleanup_processes(self):
        """清理所有子进程"""
        for proc in self.child_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass

    def run(self) -> bool:
        """
        运行 Master 主流程
        
        Returns:
            bool: 是否成功
        """
        self._print_config()
        
        if self.args.distributed:
            return self._run_distributed()
        else:
            return self._run_single()

    def _print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("SpecSoT 推理系统")
        print("=" * 60)
        print(f"执行模式: {'分布式' if self.args.distributed else '单机'}")
        print(f"调度模式: {'启用' if self.args.use_scheduling else '禁用'}")
        print(f"骨架并行: {'启用' if self.args.enable_parallel else '禁用'}")
        print(f"语义约束: {'启用' if self.args.use_semantic_constraint else '禁用'}")
        print(f"任务类型: {self.args.task}")
        print(f"样本数量: {self.args.num_samples}")
        print(f"最大Token: {self.args.max_new_tokens}")
        
        if self.args.distributed:
            print(f"设备数量: {self.world_size}")
            print(f"设备列表: {self.args.devices}")
            print(f"层分割: {self.args.layer_splits}")
            print(f"通信模式: {self.args.comm_mode}")
            print(f"Chunk大小: {self.args.chunk_size}")
        
        print("=" * 60)

    def _run_single(self) -> bool:
        """单机模式：直接执行推理"""
        from SpecSoT_v2.worker import WorkerEngine
        
        # 创建 worker 并运行
        worker = WorkerEngine(self.args)
        worker.run()
        return True

    def _run_distributed(self) -> bool:
        """分布式模式：启动多个 Worker 子进程"""
        # 清理端口
        print("正在清理端口...")
        cleanup_ports(self.args.base_port, self.world_size, None)
        time.sleep(1)
        
        # 启动所有 Worker 进程
        script_path = os.path.join(self.project_dir, "run_specsot.py")
        
        for device_cfg in self.device_configs:
            self._start_worker_process(device_cfg, script_path)
            time.sleep(1.0)
        
        print(f"\n所有进程已启动 ({self.world_size} workers)")
        print(f"日志目录: {self.log_dir}")
        
        # 等待所有进程完成
        return self._wait_for_workers()

    def _start_worker_process(self, device_cfg: DeviceConfig, script_path: str):
        """启动单个 Worker 子进程"""
        rank = device_cfg.rank
        
        cmd = [
            sys.executable, script_path,
            "--role", "worker",
            "--distributed", "True",
            "--rank", str(rank),
            "--world_size", str(self.world_size),
            "--devices", self.args.devices,
            "--layer_splits", self.args.layer_splits,
            "--base_port", str(self.args.base_port),
            "--comm_mode", self.args.comm_mode,
            "--chunk_size", str(self.args.chunk_size),
            "--base_model_path", self.args.base_model_path,
            "--eagle_model_path", self.args.eagle_model_path,
            "--use_eagle3", str(self.args.use_eagle3),
            "--enable_parallel", str(self.args.enable_parallel),
            "--use_semantic_constraint", str(self.args.use_semantic_constraint),
            "--task", self.args.task,
            "--max_new_tokens", str(self.args.max_new_tokens),
            "--num_samples", str(self.args.num_samples),
            "--seed", str(self.args.seed),
            "--use_scheduling", str(self.args.use_scheduling),
            "--max_parallel", str(self.args.max_parallel),
            "--use_bim_mode", str(self.args.use_bim_mode),
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_cfg.gpu_id)
        
        log_file = os.path.join(self.log_dir, f"rank_{rank}.log")
        print(f"启动 Worker Rank {rank} (GPU {device_cfg.gpu_id})")
        
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.child_processes.append(proc)
        
        # 输出转发线程
        t = threading.Thread(
            target=self._stream_output,
            args=(proc, log_file, rank),
            daemon=True
        )
        t.start()
        self.stream_threads.append(t)

    def _stream_output(self, proc: subprocess.Popen, log_file: str, rank: int):
        """转发子进程输出到文件和控制台"""
        with open(log_file, "w", encoding='utf-8') as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                # 只转发 Rank 0 的输出到控制台
                if rank == 0:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            proc.stdout.close()

    def _wait_for_workers(self) -> bool:
        """等待所有 Worker 完成"""
        try:
            for i, proc in enumerate(self.child_processes):
                proc.wait()
                status = "成功" if proc.returncode == 0 else f"失败(退出码:{proc.returncode})"
                print(f"Rank {i} {status}")
        except KeyboardInterrupt:
            print("\n收到中断信号，正在清理...")
            self._cleanup_processes()
            return False
        
        # 等待输出线程结束
        for t in self.stream_threads:
            t.join(timeout=2.0)
        
        success = all(proc.returncode == 0 for proc in self.child_processes)
        
        print("\n" + "=" * 60)
        if success:
            print("推理完成!")
        else:
            print("部分进程失败，请检查日志")
        print("=" * 60)
        
        return success
