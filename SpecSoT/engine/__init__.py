# coding=utf-8
"""
SpecSoT Engine 模块

提供 Master 和 Worker 引擎的统一入口。

执行模式：
- 单机模式: MasterEngine 直接执行推理
- 分布式模式: MasterEngine 启动多个 WorkerEngine 子进程

使用方式：
    from SpecSoT.engine import MasterEngine, WorkerEngine

    # 入口点（自动选择模式）
    engine = MasterEngine(args)
    engine.run()
    
    # 或直接作为 Worker 运行（分布式子进程）
    worker = WorkerEngine(args)
    worker.run()
"""

from .master import MasterEngine
from .worker import WorkerEngine, GPUMemoryMonitor
from .utils import (
    DeviceConfig,
    parse_devices,
    cleanup_ports,
    str2bool,
)

# 保持向后兼容
LauncherEngine = MasterEngine

__all__ = [
    'MasterEngine',
    'WorkerEngine',
    'LauncherEngine',  # 兼容旧名称
    'DeviceConfig',
    'parse_devices',
    'cleanup_ports',
    'str2bool',
    'GPUMemoryMonitor',
]
