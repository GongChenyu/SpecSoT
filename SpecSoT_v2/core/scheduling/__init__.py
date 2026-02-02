# coding=utf-8
"""
SpecSoT 分支调度模块

该模块提供分支调度功能，独立于分布式模块，单机也可用。

主要组件：
- DeviceProfile: 设备能力描述
- BranchInfo: 分支信息
- DeviceExecutionPlan: 设备执行计划
- SchedulePlan: 全局调度计划
- BranchStatus/BranchRuntimeState: 分支运行时状态
- BranchScheduler/HeuristicScheduler: 调度算法
- BranchExecutionManager: 执行管理器

使用示例：
    >>> from SpecSoT.scheduling import HeuristicScheduler, DeviceProfile, BranchInfo
    >>> scheduler = HeuristicScheduler()
    >>> devices = [DeviceProfile.default_single_device(0)]
    >>> branches = [BranchInfo(branch_id=0, title="Branch 0", predicted_length=100, prompt_tokens=[1,2,3])]
    >>> plan = scheduler.schedule(branches, devices)
"""

from .branch_config import (
    DeviceProfile,
    BranchInfo,
    DeviceExecutionPlan,
    SchedulePlan,
    BranchStatus,
    BranchRuntimeState,
)
from .branch_scheduler import (
    BranchScheduler,
    HeuristicScheduler,
    SimpleDistributedScheduler,
)
from .branch_manager import BranchExecutionManager

__all__ = [
    # 数据结构
    "DeviceProfile",
    "BranchInfo",
    "DeviceExecutionPlan",
    "SchedulePlan",
    "BranchStatus",
    "BranchRuntimeState",
    # 调度器
    "BranchScheduler",
    "HeuristicScheduler",
    "SimpleDistributedScheduler",
    # 执行管理器
    "BranchExecutionManager",
]
