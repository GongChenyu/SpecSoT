# coding=utf-8
"""
SpecSoT 分布式推理模块

该模块提供 SpecSoT 模型的分布式推理支持，包括：
- 分布式配置管理
- ZMQ 通信管理器
- 分布式 Prefill 管理器
- 分布式推理流程

主要组件：
- DistributedConfig: 分布式配置类，管理层拆分策略、通信配置等
- CommManager: ZMQ 通信管理器（P2P 和 Ring 模式）
- DistributedPrefillManager: 分布式 Prefill 阶段管理器
- DistributedInference: 完整的分布式推理流程

通信优先级设计：
1. DRAFT_TOKENS: 最高优先级，decoding阶段运行的基础（打包传输）
2. HIDDEN: 第二优先级，层间hidden states传输
3. EAGLE_INPUT_HIDDEN: 第三优先级，eagle layer输入hidden states
4. BASE_CACHE: 第四优先级，base model kv cache
5. EAGLE_CACHE: 第五优先级，eagle layer kv cache

使用示例：
    >>> from SpecSoT.distributed import DistributedConfig
    >>> config = DistributedConfig.from_layer_splits_str(
    ...     layer_splits_str="14,28",
    ...     rank=0,
    ...     world_size=3
    ... )
    >>> # 在 SpecSoT 模型中使用
    >>> model = SpecSoTModel.from_pretrained(
    ...     base_model_path="...",
    ...     ea_model_path="...",
    ...     distributed_config=config
    ... )
"""

from .distributed_config import DistributedConfig
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
from .distributed_prefill import DistributedPrefillManager

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
    # 配置
    "DistributedConfig",
    
    # 通信管理器
    "ZMQCommManagerBase",
    "P2PCommManager",
    "RingCommManager",
    "create_zmq_comm_manager",
    "MessageSerializer",
    "MessageType",
    "MessagePriority",
    "Message",
    
    # 推理管理器
    "DistributedPrefillManager",
    
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
