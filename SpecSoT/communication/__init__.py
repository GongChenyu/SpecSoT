# coding=utf-8
"""
SpecSoT 通信模块

该模块提供 SpecSoT 模型的通信支持，包括：
- ZMQ 通信管理器（P2P 和 Ring 模式）
- 消息类型和序列化

主要组件：
- CommManager: ZMQ 通信管理器（P2P 和 Ring 模式）
- MessageType: 消息类型枚举

通信优先级设计：
1. DRAFT_TOKENS: 最高优先级，decoding阶段运行的基础（打包传输）
2. HIDDEN: 第二优先级，层间hidden states传输
3. EAGLE_INPUT_HIDDEN: 第三优先级，eagle layer输入hidden states
4. BASE_CACHE: 第四优先级，base model kv cache
5. EAGLE_CACHE: 第五优先级，eagle layer kv cache

使用示例：
    >>> from SpecSoT.communication import create_zmq_comm_manager
    >>> comm = create_zmq_comm_manager(config, rank)
"""

from .comm_manager import (
    ZMQCommManagerBase,
    P2PCommManager,
    RingCommManager,
    create_zmq_comm_manager,
)
from .comm_utils import (
    Message,
    MessageType,
    MessagePriority,
    MessageSerializer,
    TensorSerializer,
    ThreadSafeQueue,
    AggregatedMessage,
)

# 分支调度模块 (从 scheduling 模块导入)
from ..scheduling import (
    DeviceProfile,
    BranchInfo,
    DeviceExecutionPlan,
    SchedulePlan,
    BranchStatus,
    BranchRuntimeState,
    BranchScheduler,
    HeuristicScheduler,
    SimpleDistributedScheduler,
    BranchExecutionManager,
)

# 日志工具从父模块导入
from ..logging_utils import (
    FlushingStreamHandler,
    FlushingFileHandler,
    get_unified_logger,
    get_comm_logger,
    get_prefill_logger,
    get_tensor_info,
    format_tensor_brief,
    format_timing,
    format_message_info,
)

__all__ = [
    # 通信管理器
    "ZMQCommManagerBase",
    "P2PCommManager",
    "RingCommManager",
    "create_zmq_comm_manager",
    "MessageSerializer",
    "MessageType",
    "MessagePriority",
    "Message",

    # 分支调度模块
    "DeviceProfile",
    "BranchInfo",
    "DeviceExecutionPlan",
    "SchedulePlan",
    "BranchStatus",
    "BranchRuntimeState",
    "BranchScheduler",
    "HeuristicScheduler",
    "SimpleDistributedScheduler",
    "BranchExecutionManager",

    # 日志工具
    "FlushingStreamHandler",
    "FlushingFileHandler",
    "get_unified_logger",
    "get_comm_logger",
    "get_prefill_logger",
    "get_tensor_info",
    "format_tensor_brief",
    "format_timing",
    "format_message_info",
]
