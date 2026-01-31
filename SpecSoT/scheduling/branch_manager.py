# coding=utf-8
"""
分支调度系统 - 设备端执行管理器

该模块实现设备端的分支执行管理：
- BranchExecutionManager: 设备端执行管理器，负责 Continuous Batching 执行

核心功能：
1. 管理分支的生命周期（Pending -> Prefilling -> Decoding -> Completed）
2. 实现 Continuous Batching（分支完成后立即加入新分支）
3. 动态更新 BIM/Position/Mask（复用现有 specsot_model 逻辑）

设计原则：
- 适配现有 KV Cache 管理方式（追加写入，不释放空间）
- 复用现有 verify_step_parallel 和 update_state_parallel 逻辑
- 支持单机和分布式两种模式

关键设计：
- BranchExecutionManager 不直接调用模型方法，而是通过回调函数与模型交互
- 这样可以保持调度模块与模型实现的解耦
- 模型负责提供 prefill_branches 和 decode_step 的具体实现
"""

import time
from typing import List, Dict, Optional, Tuple, Any, Callable

import torch

from .branch_config import (
    BranchInfo,
    BranchStatus,
    BranchRuntimeState,
    DeviceExecutionPlan,
    SchedulePlan,
)
from ..logging_utils import get_unified_logger


# =============================================================================
# 分支执行管理器
# =============================================================================

class BranchExecutionManager:
    """
    设备端分支执行管理器

    负责在单个设备上执行分配的分支，支持 Continuous Batching。

    设计说明：
    - 该类不直接调用模型方法，而是管理分支的调度逻辑
    - 实际的 Prefill/Decode 由 SpecSoTModel 中的方法执行
    - 通过 execute_with_model() 方法与模型集成

    Attributes:
        execution_plan: 该设备的执行计划
        branch_infos: 分支信息字典
        branch_states: 分支运行时状态
        pending_branches: 等待执行的分支 ID 列表
        active_branches: 当前活跃的分支 ID 列表
    """

    def __init__(
        self,
        execution_plan: DeviceExecutionPlan,
        branch_infos: Dict[int, BranchInfo],
        rank: int = 0,
    ):
        """
        初始化执行管理器

        Args:
            execution_plan: 该设备的执行计划
            branch_infos: 分支信息字典 {branch_id: BranchInfo}
            rank: 设备 rank
        """
        self.execution_plan = execution_plan
        self.branch_infos = branch_infos
        self.rank = rank
        self.max_parallel = execution_plan.max_parallel

        # 日志
        self.logger = get_unified_logger(rank=rank, name_suffix="-BranchMgr")

        # 分支状态管理
        self.branch_states: Dict[int, BranchRuntimeState] = {}
        self.pending_branches: List[int] = list(execution_plan.assigned_branches)
        self.active_branches: List[int] = []
        self.pending_prefill = [] # 新增：等待 Prefill 的分支队列

        # 结果存储
        self.results: Dict[int, List[int]] = {}

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'prefill_count': 0,
            'completed_count': 0,
        }

        self.logger.info(
            f"BranchExecutionManager 初始化: "
            f"{len(self.pending_branches)} 分支, max_parallel={self.max_parallel}"
        )

    # =========================================================================
    # 主执行入口
    # =========================================================================

    def get_branches_to_add(self) -> List[int]:
        """
        获取需要加入活跃批次的分支列表

        Continuous Batching 策略：
        - 当活跃分支数 < max_parallel 时，从 pending 中取出分支

        Returns:
            需要加入的分支 ID 列表
        """
        branches_to_add = []
        while (
            self.pending_branches and
            len(self.active_branches) + len(branches_to_add) < self.max_parallel
        ):
            branch_id = self.pending_branches.pop(0)
            branches_to_add.append(branch_id)

            # 初始化分支状态
            state = BranchRuntimeState(branch_id=branch_id)
            state.status = BranchStatus.PREFILLING
            self.branch_states[branch_id] = state
            self.stats['prefill_count'] += 1

        return branches_to_add

    def activate_branches(self, branch_ids: List[int]) -> None:
        """
        将分支标记为活跃状态（Prefill 完成后调用）

        Args:
            branch_ids: 分支 ID 列表
        """
        for branch_id in branch_ids:
            # 从 pending 中移除（如果存在）
            if branch_id in self.pending_branches:
                self.pending_branches.remove(branch_id)
            # 初始化状态（如果尚未初始化）
            if branch_id not in self.branch_states:
                self.branch_states[branch_id] = BranchRuntimeState(branch_id=branch_id)
            self.branch_states[branch_id].status = BranchStatus.DECODING
            # 加入活跃列表（避免重复）
            if branch_id not in self.active_branches:
                self.active_branches.append(branch_id)
            self.logger.debug(f"分支 {branch_id} 加入活跃列表")

    def handle_completed_branches(
        self,
        completed_branch_ids: List[int],
        branch_outputs: Dict[int, List[int]],
    ) -> None:
        """
        处理完成的分支

        Args:
            completed_branch_ids: 完成的分支 ID 列表
            branch_outputs: 分支输出 {branch_id: token_list}
        """
        for branch_id in completed_branch_ids:
            if branch_id not in self.branch_states:
                continue

            state = self.branch_states[branch_id]
            state.mark_completed()

            # 从活跃列表移除
            if branch_id in self.active_branches:
                self.active_branches.remove(branch_id)

            # 存储结果
            if branch_id in branch_outputs:
                self.results[branch_id] = branch_outputs[branch_id]

            self.stats['completed_count'] += 1
            self.logger.info(
                f"分支 {branch_id} 完成: "
                f"{len(self.results.get(branch_id, []))} tokens"
            )

    def step(self) -> None:
        """记录一步解码"""
        self.stats['total_steps'] += 1

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_branch_state(self, branch_id: int) -> Optional[BranchRuntimeState]:
        """获取分支状态"""
        return self.branch_states.get(branch_id)

    def is_all_completed(self) -> bool:
        """是否所有分支都已完成"""
        return not self.active_branches and not self.pending_branches

    def get_progress(self) -> Tuple[int, int]:
        """获取进度 (已完成, 总数)"""
        total = len(self.execution_plan.assigned_branches)
        completed = len(self.results)
        return completed, total

    def get_active_branch_ids(self) -> List[int]:
        """获取当前活跃的分支 ID 列表"""
        return list(self.active_branches)

    def get_branch_prompts(self, branch_ids: List[int]) -> List[List[int]]:
        """
        获取指定分支的 prompt tokens

        Args:
            branch_ids: 分支 ID 列表

        Returns:
            prompt tokens 列表
        """
        return [self.branch_infos[bid].prompt_tokens for bid in branch_ids]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        completed, total = self.get_progress()
        return {
            **self.stats,
            'completed': completed,
            'total': total,
            'active': len(self.active_branches),
            'pending': len(self.pending_branches),
        }

    def summary(self) -> str:
        """生成执行摘要"""
        completed, total = self.get_progress()
        return (
            f"BranchExecutionManager: "
            f"{completed}/{total} completed, "
            f"{len(self.active_branches)} active, "
            f"{len(self.pending_branches)} pending, "
            f"{self.stats['total_steps']} steps"
        )
    
    def add_branch(self, branch_info):
        """添加新分支"""
        # 创建分支状态对象
        state = BranchRuntimeState(
            branch_id=branch_info['id'],
            # ...
        )
        self.branch_states[branch_info['id']] = state
        self.active_branches.add(branch_info['id'])
        
        # 加入等待 Prefill 队列
        # 我们存储完整的 info 以便获取 prompt tokens
        self.pending_prefill.append(branch_info) 

    def pop_pending_prefill_branches(self):
        """获取并清除当前等待 Prefill 的分支"""
        batch = self.pending_prefill[:]
        self.pending_prefill.clear()
        return batch

    def update_step(self, step_results):
        """更新一步状态，返回完成的分支"""
        finished = {}
        # ... 根据 step_results 更新 branch_states ...
        # 如果检测到 EOS，将其移入 finished 并在 active 中删除
        return finished
