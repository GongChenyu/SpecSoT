# coding=utf-8
"""
SpecSoT Master Engine - 主控引擎

该模块实现 Master 的核心逻辑：
1. 分布式模式：启动和管理 Worker 子进程 (支持本地/SSH远程)
2. 单机模式：直接执行推理

运行模式：
- inference 模式: 交互式推理，使用用户输入的 prompt
- evaluation 模式: 评估模式，从数据集加载任务

执行模式：
- 单机模式 (distributed=False): Master 直接执行推理
- 分布式模式 (distributed=True): Master 启动多个 Worker 子进程

调度功能通过 use_scheduling 参数控制，与分布式正交。
数据加载和结果保存统一在 Master 端处理。
"""

import os
import sys
import time
import signal
import socket
import threading
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .utils import DeviceConfig, parse_devices, cleanup_ports


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
    
    # 运行模式配置
    mode: str  # inference / evaluation
    prompt: str  # inference 模式下的输入
    
    # 任务配置 (evaluation 模式)
    task: str
    num_samples: int
    
    # SSH 配置 (多机分布式)
    ssh_user: str
    ssh_key: str
    remote_python: str
    remote_workdir: str
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
        
        # 准备任务数据
        task_data = self._prepare_task_data()
        
        if self.args.distributed:
            return self._run_distributed(task_data)
        else:
            return self._run_single(task_data)
    
    def _prepare_task_data(self) -> List[Dict[str, Any]]:
        """
        准备任务数据
        
        Returns:
            List[Dict]: 任务数据列表
        """
        mode = getattr(self.args, 'mode', 'evaluation')
        
        if mode == 'inference':
            # inference 模式: 使用用户输入的 prompt
            prompt = getattr(self.args, 'prompt', '')
            return [{"task_prompt": prompt, "raw_data": {}}]
        else:
            # evaluation 模式: 从数据集加载
            from ...evaluate import load_task_dataset
            return load_task_dataset(
                task=self.args.task,
                num_samples=self.args.num_samples,
                seed=self.args.seed,
                project_dir=self.project_dir,
            )

    def _print_config(self):
        """打印配置信息"""
        mode = getattr(self.args, 'mode', 'evaluation')
        
        print("=" * 60)
        print("SpecSoT 推理系统")
        print("=" * 60)
        print(f"运行模式: {mode}")
        print(f"执行模式: {'分布式' if self.args.distributed else '单机'}")
        print(f"调度模式: {'启用' if self.args.use_scheduling else '禁用'}")
        print(f"骨架并行: {'启用' if self.args.enable_parallel else '禁用'}")
        print(f"语义约束: {'启用' if self.args.use_semantic_constraint else '禁用'}")
        
        if mode == 'inference':
            prompt = getattr(self.args, 'prompt', '')[:50]
            print(f"Prompt: {prompt}...")
        else:
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

    def _run_single(self, task_data: List[Dict[str, Any]]) -> bool:
        """单机模式：直接执行推理"""
        from .worker import WorkerEngine
        
        # 创建 worker 并运行，传递任务数据
        worker = WorkerEngine(self.args, task_data=task_data)
        results = worker.run()
        
        # 保存结果
        self._save_results(results)
        
        return True

    def _run_distributed(self, task_data: List[Dict[str, Any]]) -> bool:
        """分布式模式：启动多个 Worker 子进程"""
        # 清理端口
        print("正在清理端口...")
        cleanup_ports(self.args.base_port, self.world_size, None)
        time.sleep(1)
        
        # 将任务数据保存到临时文件，供 Worker 读取
        task_data_path = self._save_task_data_temp(task_data)
        
        # 启动所有 Worker 进程
        script_path = os.path.join(self.project_dir, "run_specsot.py")
        
        for device_cfg in self.device_configs:
            if self._is_local_device(device_cfg):
                self._start_local_worker(device_cfg, script_path, task_data_path)
            else:
                self._start_remote_worker(device_cfg, script_path, task_data_path)
            time.sleep(1.0)
        
        print(f"\n所有进程已启动 ({self.world_size} workers)")
        print(f"日志目录: {self.log_dir}")
        
        # 等待所有进程完成
        return self._wait_for_workers()

    def _build_worker_cmd(self, rank: int, task_data_path: str) -> List[str]:
        """构建 Worker 启动命令"""
        return [
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
            "--mode", getattr(self.args, 'mode', 'evaluation'),
            "--max_new_tokens", str(self.args.max_new_tokens),
            "--num_samples", str(self.args.num_samples),
            "--seed", str(self.args.seed),
            "--use_scheduling", str(self.args.use_scheduling),
            "--max_parallel", str(self.args.max_parallel),
            "--task_data_path", task_data_path,
        ]

    def _start_local_worker(self, device_cfg: DeviceConfig, script_path: str, task_data_path: str):
        """启动本地 Worker 子进程"""
        rank = device_cfg.rank
        
        cmd = [sys.executable, script_path] + self._build_worker_cmd(rank, task_data_path)
        
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
    
    def _start_remote_worker(self, device_cfg: DeviceConfig, script_path: str, task_data_path: str):
        """
        通过 SSH 启动远程 Worker
        
        注意: 远程启动需要：
        1. 配置 SSH 免密登录
        2. 远程机器有相同的代码和模型路径
        3. 远程机器可以访问临时文件 (共享存储或需要传输)
        """
        rank = device_cfg.rank
        ssh_user = getattr(self.args, 'ssh_user', '')
        ssh_key = getattr(self.args, 'ssh_key', '')
        remote_python = getattr(self.args, 'remote_python', 'python')
        remote_workdir = getattr(self.args, 'remote_workdir', '') or self.project_dir
        
        if not ssh_user:
            raise ValueError(f"远程设备 {device_cfg.ip} 需要配置 --ssh_user 参数")
        
        # 构建远程执行命令
        remote_script = os.path.join(remote_workdir, "run_specsot.py")
        worker_args = self._build_worker_cmd(rank, task_data_path)
        remote_cmd = f"cd {remote_workdir} && CUDA_VISIBLE_DEVICES={device_cfg.gpu_id} {remote_python} {remote_script} {' '.join(worker_args)}"
        
        # 构建 SSH 命令
        ssh_cmd = ["ssh"]
        if ssh_key:
            ssh_cmd.extend(["-i", ssh_key])
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            f"{ssh_user}@{device_cfg.ip}",
            remote_cmd
        ])
        
        log_file = os.path.join(self.log_dir, f"rank_{rank}.log")
        print(f"启动远程 Worker Rank {rank} ({device_cfg.ip}:{device_cfg.gpu_id})")
        
        proc = subprocess.Popen(
            ssh_cmd,
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
    
    def _save_task_data_temp(self, task_data: List[Dict[str, Any]]) -> str:
        """
        将任务数据保存到临时文件
        
        Args:
            task_data: 任务数据列表
            
        Returns:
            str: 临时文件路径
        """
        import json
        import tempfile
        
        temp_dir = os.path.join(self.project_dir, "logs", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, f"task_data_{int(time.time())}.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False)
        
        return temp_path
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """
        保存推理结果
        
        Args:
            results: 推理结果列表
        """
        if not results:
            return
        
        from ...evaluate import save_results
        
        mode = getattr(self.args, 'mode', 'evaluation')
        output_dir = os.path.join(self.project_dir, "evaluate", "results")
        
        extra_info = {
            "model": os.path.basename(self.args.base_model_path),
            "enable_parallel": self.args.enable_parallel,
            "distributed": self.args.distributed,
        }
        
        save_results(
            results=results,
            task=self.args.task,
            output_dir=output_dir,
            mode=mode,
            extra_info=extra_info,
        )
    
    def _get_local_ip(self) -> str:
        """获取本机 IP 地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _is_local_device(self, device_cfg: DeviceConfig) -> bool:
        """判断是否为本地设备"""
        local_ip = self._get_local_ip()
        return device_cfg.ip in ("127.0.0.1", "localhost", local_ip)

