# coding=utf-8
"""
分支调度系统 - 调度算法

该模块实现分支调度算法：
- BranchScheduler: 调度器基类
- HeuristicScheduler: 启发式调度器（LPT + 轮询）

调度策略：
1. 长序列优先 (LPT - Longest Processing Time)
2. 轮询分配到负载最小的设备
3. 为每个设备生成执行批次（考虑 max_parallel）

设计原则：
- 调度模块独立于分布式，单机也可用
- 支持异构设备（不同算力、不同 max_parallel）
- 预留高级调度算法接口（Roofline、能耗感知等）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import heapq

from .branch_config import (
    DeviceProfile,
    BranchInfo,
    DeviceExecutionPlan,
    SchedulePlan,
)


# =============================================================================
# 调度器基类
# =============================================================================

class BranchScheduler(ABC):
    """
    分支调度器基类

    所有调度器必须实现 schedule() 方法，将分支分配到设备。
    """

    @abstractmethod
    def schedule(
        self,
        branches: List[BranchInfo],
        devices: List[DeviceProfile],
    ) -> SchedulePlan:
        """
        将分支分配到设备

        Args:
            branches: 分支信息列表
            devices: 设备能力列表

        Returns:
            SchedulePlan: 调度计划
        """
        raise NotImplementedError


# =============================================================================
# 启发式调度器 (LPT + 轮询)
# =============================================================================

class HeuristicScheduler(BranchScheduler):
    """
    启发式调度器：长序列优先 + 轮询设备

    策略：
    1. 按预估长度降序排序分支
    2. 依次将分支分配给当前负载最小的设备（考虑算力权重）
    3. 为每个设备生成执行批次（考虑 max_parallel）

    算力权重：
    - 设备负载 = 分配的总 token 数 / 设备算力
    - 算力高的设备可以承担更多分支
    """

    def __init__(self, use_compute_weight: bool = True):
        """
        Args:
            use_compute_weight: 是否使用算力权重进行负载均衡
        """
        self.use_compute_weight = use_compute_weight

    def schedule(
        self,
        branches: List[BranchInfo],
        devices: List[DeviceProfile],
    ) -> SchedulePlan:
        """
        执行调度

        Args:
            branches: 分支信息列表
            devices: 设备能力列表

        Returns:
            SchedulePlan: 调度计划
        """
        if not branches:
            return SchedulePlan(
                num_branches=0,
                num_devices=len(devices),
                scheduler_type="heuristic",
            )

        if not devices:
            # 默认单设备
            devices = [DeviceProfile.default_single_device()]

        # 1. 按预估长度降序排序
        sorted_branches = sorted(
            branches,
            key=lambda b: b.predicted_length,
            reverse=True
        )

        # 2. LPT 分配到设备
        branch_to_device, device_branches = self._lpt_assign(
            sorted_branches, devices
        )

        # 3. 为每个设备生成执行批次
        device_plans = {}
        for device in devices:
            branch_ids = device_branches[device.device_id]
            batches = self._generate_execution_batches(
                branch_ids, device.max_parallel
            )
            device_plans[device.device_id] = DeviceExecutionPlan(
                device_id=device.device_id,
                execution_batches=batches,
                assigned_branches=branch_ids,
                max_parallel=device.max_parallel,
            )

        return SchedulePlan(
            num_branches=len(branches),
            num_devices=len(devices),
            branch_to_device=branch_to_device,
            device_plans=device_plans,
            global_branch_info=branches,
            scheduler_type="heuristic",
        )

    def _lpt_assign(
        self,
        sorted_branches: List[BranchInfo],
        devices: List[DeviceProfile],
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        LPT 分配算法：将分支分配到负载最小的设备

        使用最小堆维护设备负载，每次将分支分配给负载最小的设备。

        Args:
            sorted_branches: 按长度降序排序的分支列表
            devices: 设备列表

        Returns:
            branch_to_device: 分支 ID -> 设备 ID 映射
            device_branches: 设备 ID -> 分支 ID 列表
        """
        # 初始化设备负载堆: (负载, 设备ID)
        # 负载 = 总 token 数 / 算力（如果启用算力权重）
        device_heap = []
        for device in devices:
            heapq.heappush(device_heap, (0.0, device.device_id))

        # 设备算力映射
        compute_map = {d.device_id: d.compute_capability for d in devices}

        # 分配结果
        branch_to_device = {}
        device_branches = {d.device_id: [] for d in devices}

        for branch in sorted_branches:
            # 取出负载最小的设备
            load, device_id = heapq.heappop(device_heap)

            # 分配分支
            branch_to_device[branch.branch_id] = device_id
            device_branches[device_id].append(branch.branch_id)

            # 更新负载
            if self.use_compute_weight:
                new_load = load + branch.predicted_length / compute_map[device_id]
            else:
                new_load = load + branch.predicted_length

            heapq.heappush(device_heap, (new_load, device_id))

        return branch_to_device, device_branches

    def _generate_execution_batches(
        self,
        branch_ids: List[int],
        max_parallel: int,
    ) -> List[List[int]]:
        """
        生成执行批次

        将分支按 max_parallel 分组，每组内并行执行。

        示例:
            branch_ids=[70, 40, 10], max_parallel=2
            返回: [[70, 40], [10]]
            - 第一批: 70 和 40 并行执行
            - 第二批: 10 单独执行

        Note:
            这是静态调度的初始计划。在 Continuous Batching 模式下，
            当某分支提前完成时，下一批次的分支可以提前加入。

        Args:
            branch_ids: 分支 ID 列表（已按长度降序排序）
            max_parallel: 最大并行度

        Returns:
            执行批次列表
        """
        if not branch_ids:
            return []

        batches = []
        for i in range(0, len(branch_ids), max_parallel):
            batch = branch_ids[i:i + max_parallel]
            batches.append(batch)

        return batches
