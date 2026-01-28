# coding=utf-8
"""
分支调度系统 - 通信管理器

该模块实现分支级别的通信功能：
- BranchMessageType: 分支调度相关的消息类型
- BranchCommManager: 分支级通信管理器

通信流程：
1. 主节点广播调度计划 (SCHEDULE_PLAN)
2. 主节点发送分支 Prompt (BRANCH_PROMPT)
3. 各设备返回分支输出 (BRANCH_OUTPUT)
4. 主节点发送完成信号 (ALL_COMPLETE)

关键设计：
- 无需传输 Skeleton KV Cache（各设备独立 Prefill）
- 只传输 prompt tokens（几百~几千 tokens）
- 复用现有 ZMQ 通信基础设施
"""

import pickle
import time
import logging
from enum import IntEnum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from ..scheduling import (
    BranchInfo,
    SchedulePlan,
    DeviceExecutionPlan,
)
from .comm_utils import MessageSerializer
from ..logging_utils import get_unified_logger


# =============================================================================
# 分支消息类型
# =============================================================================

class BranchMessageType(IntEnum):
    """
    分支调度相关的消息类型

    调度相关:
    - SCHEDULE_PLAN: 调度计划广播
    - BRANCH_PROMPT: 分支 Prompt (token IDs)

    结果相关:
    - BRANCH_OUTPUT: 分支输出 (token IDs)
    - BRANCH_COMPLETE: 单个分支完成

    控制相关:
    - ALL_COMPLETE: 全部完成
    - ABORT: 中止执行
    """
    # 调度相关
    SCHEDULE_PLAN = 10
    BRANCH_PROMPT = 11

    # 结果相关
    BRANCH_OUTPUT = 20
    BRANCH_COMPLETE = 21

    # 控制相关
    ALL_COMPLETE = 30
    ABORT = 31


# =============================================================================
# 分支消息结构
# =============================================================================

@dataclass
class BranchMessage:
    """
    分支级通信消息结构

    Attributes:
        msg_type: 消息类型
        src_rank: 源设备 rank
        dst_rank: 目标设备 rank (-1 表示广播)
        data: 消息数据
        branch_id: 分支 ID (-1 表示不针对特定分支)
        timestamp: 时间戳
    """
    msg_type: BranchMessageType
    src_rank: int
    dst_rank: int
    data: Any
    branch_id: int = -1
    timestamp: float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        """序列化消息"""
        return pickle.dumps({
            'msg_type': int(self.msg_type),
            'src_rank': self.src_rank,
            'dst_rank': self.dst_rank,
            'data': self.data,
            'branch_id': self.branch_id,
            'timestamp': self.timestamp,
        })

    @classmethod
    def deserialize(cls, data: bytes) -> "BranchMessage":
        """反序列化消息"""
        msg_dict = pickle.loads(data)
        return cls(
            msg_type=BranchMessageType(msg_dict['msg_type']),
            src_rank=msg_dict['src_rank'],
            dst_rank=msg_dict['dst_rank'],
            data=msg_dict['data'],
            branch_id=msg_dict['branch_id'],
            timestamp=msg_dict['timestamp'],
        )


# =============================================================================
# 分支通信管理器
# =============================================================================

class BranchCommManager:
    """
    分支级通信管理器

    负责分支调度相关的通信：
    - 调度计划广播
    - 分支 Prompt 分发
    - 分支输出收集
    - 完成信号同步

    Note:
        该类复用现有的 ZMQ 通信基础设施，
        通过 base_comm_manager 进行底层通信。
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        base_comm_manager: Optional[Any] = None,
        device: str = "cuda",
    ):
        """
        初始化分支通信管理器

        Args:
            rank: 当前设备 rank
            world_size: 总设备数
            base_comm_manager: 底层通信管理器（可选，用于复用现有 ZMQ 连接）
            device: 设备类型
        """
        self.rank = rank
        self.world_size = world_size
        self.base_comm_manager = base_comm_manager
        self.device = device

        # 日志
        self.logger = get_unified_logger(rank=rank, name_suffix="-BranchComm")

        # 结果收集缓存
        self._branch_outputs: Dict[int, List[int]] = {}
        self._completed_branches: set = set()
        
        # 消息缓冲区（用于存储类型不匹配的消息）
        self._message_buffer: List[BranchMessage] = []

        self.logger.info(
            f"BranchCommManager 初始化完成: rank={rank}, world_size={world_size}"
        )

    @property
    def is_main_rank(self) -> bool:
        """是否为主节点（rank 0）"""
        return self.rank == 0

    # =========================================================================
    # 调度计划广播 (主节点 -> 所有设备)
    # =========================================================================

    def broadcast_schedule_plan(
        self,
        schedule_plan: SchedulePlan,
    ) -> None:
        """
        广播调度计划到所有设备（仅主节点调用）

        Args:
            schedule_plan: 调度计划
        """
        if not self.is_main_rank:
            self.logger.warning("非主节点调用 broadcast_schedule_plan，忽略")
            return

        self.logger.info(f"广播调度计划: {schedule_plan.num_branches} 分支")

        # 序列化调度计划
        plan_data = {
            'num_branches': schedule_plan.num_branches,
            'num_devices': schedule_plan.num_devices,
            'branch_to_device': schedule_plan.branch_to_device,
            'device_plans': {
                did: {
                    'device_id': plan.device_id,
                    'execution_batches': plan.execution_batches,
                    'assigned_branches': plan.assigned_branches,
                    'max_parallel': plan.max_parallel,
                }
                for did, plan in schedule_plan.device_plans.items()
            },
            'scheduler_type': schedule_plan.scheduler_type,
        }

        # 广播到所有其他设备
        for dst_rank in range(1, self.world_size):
            msg = BranchMessage(
                msg_type=BranchMessageType.SCHEDULE_PLAN,
                src_rank=self.rank,
                dst_rank=dst_rank,
                data=plan_data,
            )
            self._send_message(msg, dst_rank)

        self.logger.info("调度计划广播完成")

    def receive_schedule_plan(self, timeout: float = 30.0) -> Optional[SchedulePlan]:
        """
        接收调度计划（非主节点调用）

        Args:
            timeout: 超时时间（秒）

        Returns:
            调度计划，超时返回 None
        """
        if self.is_main_rank:
            self.logger.warning("主节点调用 receive_schedule_plan，忽略")
            return None

        msg = self._receive_message(
            msg_type=BranchMessageType.SCHEDULE_PLAN,
            timeout=timeout
        )
        if msg is None:
            self.logger.error("接收调度计划超时")
            return None

        # 反序列化调度计划
        plan_data = msg.data
        device_plans = {}
        for did, pdata in plan_data['device_plans'].items():
            device_plans[int(did)] = DeviceExecutionPlan(
                device_id=pdata['device_id'],
                execution_batches=pdata['execution_batches'],
                assigned_branches=pdata['assigned_branches'],
                max_parallel=pdata['max_parallel'],
            )

        schedule_plan = SchedulePlan(
            num_branches=plan_data['num_branches'],
            num_devices=plan_data['num_devices'],
            branch_to_device=plan_data['branch_to_device'],
            device_plans=device_plans,
            scheduler_type=plan_data['scheduler_type'],
        )

        self.logger.info(f"收到调度计划: {schedule_plan.num_branches} 分支")
        return schedule_plan

    # =========================================================================
    # 分支 Prompt 分发 (主节点 -> 各设备)
    # =========================================================================

    def send_branch_prompts(
        self,
        branch_infos: List[BranchInfo],
        schedule_plan: SchedulePlan,
    ) -> None:
        """
        发送分支 Prompt 到各设备（仅主节点调用）

        Args:
            branch_infos: 分支信息列表
            schedule_plan: 调度计划
        """
        if not self.is_main_rank:
            return

        # 按设备分组分支
        device_branches: Dict[int, List[BranchInfo]] = {}
        for info in branch_infos:
            device_id = schedule_plan.get_device_for_branch(info.branch_id)
            if device_id not in device_branches:
                device_branches[device_id] = []
            device_branches[device_id].append(info)

        # 发送到各设备
        for device_id, infos in device_branches.items():
            if device_id == 0:
                continue  # 主节点不需要发送给自己

            prompt_data = [
                {
                    'branch_id': info.branch_id,
                    'title': info.title,
                    'predicted_length': info.predicted_length,
                    'prompt_tokens': info.prompt_tokens,
                }
                for info in infos
            ]

            msg = BranchMessage(
                msg_type=BranchMessageType.BRANCH_PROMPT,
                src_rank=self.rank,
                dst_rank=device_id,
                data=prompt_data,
            )
            self._send_message(msg, device_id)

        self.logger.info(f"分支 Prompt 分发完成: {len(branch_infos)} 分支")

    def receive_branch_prompts(
        self,
        timeout: float = 30.0
    ) -> List[BranchInfo]:
        """
        接收分支 Prompt（非主节点调用）

        Returns:
            分支信息列表
        """
        if self.is_main_rank:
            return []

        msg = self._receive_message(
            msg_type=BranchMessageType.BRANCH_PROMPT,
            timeout=timeout
        )
        if msg is None:
            self.logger.error("接收分支 Prompt 超时")
            return []

        branch_infos = [
            BranchInfo(
                branch_id=data['branch_id'],
                title=data['title'],
                predicted_length=data['predicted_length'],
                prompt_tokens=data['prompt_tokens'],
            )
            for data in msg.data
        ]

        self.logger.info(f"收到 {len(branch_infos)} 个分支 Prompt")
        return branch_infos

    # =========================================================================
    # 分支输出收集 (各设备 -> 主节点)
    # =========================================================================

    def send_branch_output(
        self,
        branch_id: int,
        output_tokens: List[int],
    ) -> None:
        """
        发送分支输出到主节点（非主节点调用）

        Args:
            branch_id: 分支 ID
            output_tokens: 生成的 token 列表
        """
        if self.is_main_rank:
            # 主节点直接存储
            self._branch_outputs[branch_id] = output_tokens
            self._completed_branches.add(branch_id)
            return

        msg = BranchMessage(
            msg_type=BranchMessageType.BRANCH_OUTPUT,
            src_rank=self.rank,
            dst_rank=0,
            data=output_tokens,
            branch_id=branch_id,
        )
        self._send_message(msg, 0)
        self.logger.debug(f"发送分支 {branch_id} 输出: {len(output_tokens)} tokens")

    def collect_all_outputs(
        self,
        num_branches: int,
        timeout: float = 300.0,
    ) -> Dict[int, List[int]]:
        """
        收集所有分支输出（仅主节点调用）

        Args:
            num_branches: 预期分支总数
            timeout: 超时时间（秒）

        Returns:
            分支 ID -> 输出 tokens 映射
        """
        if not self.is_main_rank:
            return {}

        deadline = time.time() + timeout

        while len(self._completed_branches) < num_branches:
            remaining = deadline - time.time()
            if remaining <= 0:
                self.logger.error(
                    f"收集分支输出超时: {len(self._completed_branches)}/{num_branches}"
                )
                break

            msg = self._receive_message(
                msg_type=BranchMessageType.BRANCH_OUTPUT,
                timeout=min(1.0, remaining)
            )
            if msg is not None:
                self._branch_outputs[msg.branch_id] = msg.data
                self._completed_branches.add(msg.branch_id)
                self.logger.debug(
                    f"收到分支 {msg.branch_id} 输出: "
                    f"{len(self._completed_branches)}/{num_branches}"
                )

        self.logger.info(f"收集完成: {len(self._branch_outputs)} 分支")
        return self._branch_outputs

    # =========================================================================
    # 完成信号 (主节点 -> 所有设备)
    # =========================================================================

    def broadcast_all_complete(self) -> None:
        """广播全部完成信号（仅主节点调用）"""
        if not self.is_main_rank:
            return

        for dst_rank in range(1, self.world_size):
            msg = BranchMessage(
                msg_type=BranchMessageType.ALL_COMPLETE,
                src_rank=self.rank,
                dst_rank=dst_rank,
                data=None,
            )
            self._send_message(msg, dst_rank)

        self.logger.info("全部完成信号已广播")

    def wait_for_complete(self, timeout: float = 300.0) -> bool:
        """等待完成信号（非主节点调用）"""
        if self.is_main_rank:
            return True

        msg = self._receive_message(
            msg_type=BranchMessageType.ALL_COMPLETE,
            timeout=timeout
        )
        return msg is not None

    # =========================================================================
    # 底层通信方法
    # =========================================================================

    def _send_message(self, msg: BranchMessage, dst_rank: int) -> None:
        """
        发送消息到指定设备

        Args:
            msg: 消息
            dst_rank: 目标设备 rank
        """
        if self.base_comm_manager is not None:
            # 使用现有通信管理器
            data = msg.serialize()
            self._send_raw_to_rank(data, dst_rank)
        else:
            # 单机模式：直接存储（用于测试）
            self.logger.warning("无底层通信管理器，消息未发送")

    def _receive_message(
        self,
        msg_type: BranchMessageType,
        timeout: float = 30.0,
    ) -> Optional[BranchMessage]:
        """
        接收指定类型的消息

        Args:
            msg_type: 消息类型
            timeout: 超时时间（秒）

        Returns:
            消息，超时返回 None
        """
        if self.base_comm_manager is not None:
            data = self._recv_raw(timeout=timeout)
            if data is not None:
                msg = BranchMessage.deserialize(data)
                if msg.msg_type == msg_type:
                    return msg
                else:
                    # 消息类型不匹配，存入缓冲区
                    self._message_buffer.append(msg)
        return None

    def _send_raw_to_rank(self, data: bytes, dst_rank: int) -> None:
        """
        发送原始字节数据到指定 rank

        Args:
            data: 原始字节数据
            dst_rank: 目标 rank
        """
        if self.base_comm_manager is None:
            self.logger.warning("无底层通信管理器，无法发送数据")
            return

        # 使用底层通信管理器的发送功能
        # 通过 send_queue 发送，复用现有的异步发送机制
        from .comm_utils import Message, MessageType
        msg = Message(
            msg_type=MessageType.DRAFT_TOKENS,  # 复用 DRAFT_TOKENS 信道
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=data,
            layer_idx=-1,
            chunk_idx=-1,
        )
        self.base_comm_manager.send_queue.put(msg)

    def _recv_raw(self, timeout: float = 30.0) -> Optional[bytes]:
        """
        接收原始字节数据

        Args:
            timeout: 超时时间

        Returns:
            原始字节数据，超时返回 None
        """
        if self.base_comm_manager is None:
            return None

        # 从底层通信管理器接收数据
        # 尝试从 draft_tokens 队列获取
        from .comm_utils import MessageType
        msg = self.base_comm_manager.recv_queue.get_by_type(
            MessageType.DRAFT_TOKENS,
            timeout=timeout
        )
        if msg is not None and isinstance(msg.data, bytes):
            return msg.data
        elif msg is not None:
            # 数据不是 bytes 类型，可能是其他格式
            self.logger.debug(f"收到非 bytes 类型数据: {type(msg.data)}")
        return None

    # =========================================================================
    # 分支 Prompt 发送（支持单独发送）
    # =========================================================================

    def send_branch_info_to_device(
        self,
        branch_info: BranchInfo,
        dst_rank: int,
    ) -> None:
        """
        发送单个分支信息到指定设备

        Args:
            branch_info: 分支信息
            dst_rank: 目标设备 rank
        """
        if dst_rank == self.rank:
            return  # 不需要发送给自己

        prompt_data = {
            'branch_id': branch_info.branch_id,
            'title': branch_info.title,
            'predicted_length': branch_info.predicted_length,
            'prompt_tokens': branch_info.prompt_tokens,
        }

        msg = BranchMessage(
            msg_type=BranchMessageType.BRANCH_PROMPT,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=[prompt_data],  # 使用列表格式保持兼容
            branch_id=branch_info.branch_id,
        )
        self._send_message(msg, dst_rank)
        self.logger.debug(f"发送分支 {branch_info.branch_id} 到设备 {dst_rank}")

    # =========================================================================
    # 清理方法
    # =========================================================================

    def reset(self) -> None:
        """重置状态（每次推理前调用）"""
        self._branch_outputs.clear()
        self._completed_branches.clear()
        self._message_buffer.clear()

    def cleanup(self) -> None:
        """清理资源"""
        self.reset()
        self.logger.info("BranchCommManager 已清理")
