# coding=utf-8
"""
ZMQ通信管理器模块 (向后兼容入口)

此文件作为向后兼容入口，实际实现已拆分为：
- base_comm_manager.py: 基类和通用组件
- p2p_comm_manager.py: P2P通信管理器
- ring_comm_manager.py: Ring通信管理器

为保持现有代码的兼容性，所有公开接口仍可从此模块导入。
"""

from typing import Dict, Optional

from .base_comm_manager import ZMQCommManagerBase
from .comm_utils import (
    Message,
    MessageType,
    MessagePriority,
    MessageSerializer,
    TensorSerializer,
    ThreadSafeQueue,
    AggregatedMessage,
)
from .p2p_comm_manager import P2PCommManager
from .ring_comm_manager import RingCommManager


def create_zmq_comm_manager(
    rank: int,
    world_size: int,
    base_port: int = 29500,
    mode: str = "p2p",
    device: str = "cuda",
    node_addresses: Optional[Dict[int, str]] = None
) -> ZMQCommManagerBase:
    """
    创建ZMQ通信管理器
    
    Args:
        rank: 当前设备rank
        world_size: 总设备数
        base_port: 基础端口号
        mode: 通信模式 ("p2p" 或 "ring")
        device: 设备类型
        node_addresses: 节点地址映射
        
    Returns:
        ZMQCommManagerBase实例（P2PCommManager或RingCommManager）
    """
    if mode == "p2p":
        manager = P2PCommManager(
            rank=rank,
            world_size=world_size,
            base_port=base_port,
            device=device,
            node_addresses=node_addresses
        )
    elif mode == "ring":
        manager = RingCommManager(
            rank=rank,
            world_size=world_size,
            base_port=base_port,
            device=device,
            node_addresses=node_addresses
        )
    else:
        raise ValueError(f"不支持的通信模式: {mode}，请使用 'p2p' 或 'ring'")
    
    return manager


# 保持向后兼容的别名
# ZMQCommManager = P2PCommManager


__all__ = [
    # 基类和接口
    "ZMQCommManagerBase",
    
    # 具体实现
    "P2PCommManager",
    "RingCommManager",
    
    # 工厂函数
    "create_zmq_comm_manager",
    
    # 消息类型
    "Message",
    "MessageType",
    "MessagePriority",
    "AggregatedMessage",
    
    # 序列化工具
    "MessageSerializer",
    "TensorSerializer",
    
    # 队列
    "ThreadSafeQueue",
    
    # 向后兼容
    "ZMQCommManager",
]
