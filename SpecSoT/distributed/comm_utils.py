# coding=utf-8
"""
ZMQ通信管理器工具

包含消息类型、优先级、序列化工具等通信基础设施
"""

import zmq
import torch
import threading
import queue
import time
import pickle
import io
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum


# ==================== 日志设置 ====================

def setup_comm_utils_logger(rank: int = -1, log_dir: str = "logs") -> logging.Logger:
    """
    设置comm_utils模块的日志器
    
    Args:
        rank: 当前进程rank，-1表示自动检测
        log_dir: 日志目录
        
    Returns:
        配置好的logger
    """
    if rank < 0:
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    
    logger_name = f"CommUtils.Rank{rank}"
    logger = logging.getLogger(logger_name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # 格式化器
        formatter = logging.Formatter(
            f'%(asctime)s.%(msecs)03d - Rank{rank} - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"comm_utils_rank{rank}.log"),
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.propagate = False
    
    return logger


def get_data_info(data: Any) -> str:
    """
    获取数据的详细信息字符串
    
    Args:
        data: 数据（可以是tensor、tuple等）
        
    Returns:
        描述字符串
    """
    if isinstance(data, torch.Tensor):
        size_bytes = data.numel() * data.element_size()
        size_kb = size_bytes / 1024
        if size_kb > 1024:
            size_str = f"{size_kb/1024:.2f}MB"
        else:
            size_str = f"{size_kb:.2f}KB"
        return f"Tensor(shape={list(data.shape)}, dtype={data.dtype}, device={data.device}, size={size_str})"
    elif isinstance(data, tuple) and len(data) == 2:
        if isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor):
            key, value = data
            total_size = (key.numel() + value.numel()) * key.element_size()
            size_str = f"{total_size/1024/1024:.2f}MB"
            return f"KVCache(key={list(key.shape)}, value={list(value.shape)}, size={size_str})"
    elif isinstance(data, tuple) and len(data) == 4:
        # draft tokens: (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
        return f"DraftTokens({len(data)} tensors)"
    return f"{type(data).__name__}"


# 模块级logger（延迟初始化）
_module_logger = None

def _get_logger() -> logging.Logger:
    """获取或创建模块级logger"""
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_comm_utils_logger()
    return _module_logger

# ==================== 枚举和数据类 ====================

class MessagePriority(IntEnum):
    """
    消息优先级（数值越小优先级越高）
    
    优先级设计说明：
    1. DRAFT_TOKENS (0): 最高优先级，decoding阶段运行的基础
    2. HIDDEN (1): 第二优先级，会阻塞推理流程
    3. EAGLE_INPUT_HIDDEN (2): 第三优先级，eagle layer输入
    4. BASE_CACHE (3): 第四优先级，base model的kv cache
    5. EAGLE_CACHE (4): 第五优先级，eagle layer的kv cache
    """
    DRAFT_TOKENS = 0         # 最高优先级：draft tree结果
    HIDDEN = 1               # 第二优先级：层间hidden states
    EAGLE_INPUT_HIDDEN = 2   # 第三优先级：eagle layer输入hidden states
    BASE_CACHE = 3           # 第四优先级：base model kv cache
    EAGLE_CACHE = 4          # 第五优先级：eagle layer kv cache


class MessageType(IntEnum):
    """
    消息类型
    
    与优先级一一对应，用于区分不同类型的消息
    """
    DRAFT_TOKENS = 0         # draft tree打包数据
    HIDDEN = 1               # 层间hidden states
    EAGLE_INPUT_HIDDEN = 2   # eagle layer输入hidden states
    BASE_CACHE = 3           # base model kv cache
    EAGLE_CACHE = 4          # eagle layer kv cache


@dataclass
class Message:
    """通信消息结构"""
    msg_type: MessageType           # 消息类型
    src_rank: int                   # 源设备rank（当前发送者）
    dst_rank: int                   # 目标设备rank
    data: Any                       # 数据（tensor或其他）
    chunk_idx: int = -1             # chunk索引（用于SP）
    layer_idx: int = -1             # 层索引（用于cache和PP）
    seq_id: int = 0                 # 序列ID（用于排序和追踪）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    original_src: int = -1          # 原始来源（用于ring模式转发追踪）
    
    @property
    def priority(self) -> int:
        """获取消息优先级"""
        return MessagePriority(self.msg_type).value
    
    def get_effective_src(self) -> int:
        """获取有效的源rank（考虑ring模式转发）"""
        return self.original_src if self.original_src >= 0 else self.src_rank


@dataclass  
class AggregatedMessage:
    """聚合消息（用于ring模式的合并发送）"""
    dst_rank: int                           # 目标rank
    messages: List[Message] = field(default_factory=list)  # 消息列表
    
    def add_message(self, msg: Message):
        """添加消息到聚合包"""
        self.messages.append(msg)
        # 按优先级排序（优先级高的在前）
        self.messages.sort(key=lambda m: (m.priority, m.timestamp))
    
    def get_size(self) -> int:
        """获取聚合包中的消息数量"""
        return len(self.messages)
    
    def clear(self):
        """清空消息列表"""
        self.messages.clear()


# ==================== 线程安全队列 ====================

class ThreadSafeQueue:
    """
    线程安全的队列，支持按消息类型存储和获取
    """
    
    def __init__(self, rank: int = -1):
        self._queues: Dict[MessageType, queue.Queue] = {
            MessageType.DRAFT_TOKENS: queue.Queue(),
            MessageType.HIDDEN: queue.Queue(),
            MessageType.EAGLE_INPUT_HIDDEN: queue.Queue(),
            MessageType.BASE_CACHE: queue.Queue(),
            MessageType.EAGLE_CACHE: queue.Queue(),
        }
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._rank = rank if rank >= 0 else int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        self._put_count = 0
        self._get_count = 0
        self.logger = logging.getLogger(f"ThreadSafeQueue.Rank{self._rank}")
        
    def put(self, msg: Message):
        """放入消息"""
        msg_type = msg.msg_type
        with self._not_empty:
            self._queues[msg_type].put(msg)
            self._put_count += 1
            self._not_empty.notify_all()
        self.logger.debug(
            f"[QUEUE_PUT] type={MessageType(msg_type).name}, "
            f"src={msg.src_rank}, dst={msg.dst_rank}, "
            f"layer={msg.layer_idx}, chunk={msg.chunk_idx}, "
            f"queue_size={self._queues[msg_type].qsize()}"
        )
    
    def get_by_type(self, msg_type: MessageType, timeout: float = None) -> Optional[Message]:
        """获取特定类型的消息"""
        deadline = time.time() + timeout if timeout else None
        
        while True:
            try:
                # 尝试非阻塞获取
                msg = self._queues[msg_type].get_nowait()
                self._get_count += 1
                self.logger.debug(
                    f"[QUEUE_GET] type={MessageType(msg_type).name}, "
                    f"src={msg.src_rank}, dst={msg.dst_rank}, "
                    f"layer={msg.layer_idx}, chunk={msg.chunk_idx}"
                )
                return msg
            except queue.Empty:
                pass
            
            # 检查超时
            if timeout is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.logger.debug(f"[QUEUE_TIMEOUT] type={MessageType(msg_type).name}, timeout={timeout}s")
                    return None
                # 等待一小段时间后重试
                with self._not_empty:
                    self._not_empty.wait(min(0.01, remaining))
            else:
                with self._not_empty:
                    self._not_empty.wait(0.01)
    
    def get_by_priority(self, timeout: float = None) -> Optional[Message]:
        """按优先级获取消息（优先级高的先出）"""
        deadline = time.time() + timeout if timeout else None
        
        while True:
            # 按优先级顺序检查队列
            for msg_type in MessageType:
                try:
                    return self._queues[msg_type].get_nowait()
                except queue.Empty:
                    continue
            
            # 检查超时
            if timeout is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                with self._not_empty:
                    self._not_empty.wait(min(0.01, remaining))
            else:
                with self._not_empty:
                    self._not_empty.wait(0.01)
    
    def get_all_by_dst(self, dst_rank: int, max_count: int = -1) -> List[Message]:
        """
        获取所有发往指定目标的消息（用于ring模式聚合）
        非阻塞，立即返回当前可用的消息
        
        Args:
            dst_rank: 目标rank
            max_count: 最大获取数量，-1表示无限制
            
        Returns:
            符合条件的消息列表
        """
        result = []
        temp_hold = []
        count = 0
        
        # 按优先级顺序扫描所有队列
        for msg_type in MessageType:
            q = self._queues[msg_type]
            items_to_check = []
            
            # 暂时取出所有消息
            while True:
                try:
                    msg = q.get_nowait()
                    items_to_check.append(msg)
                except queue.Empty:
                    break
            
            # 筛选符合条件的消息
            for msg in items_to_check:
                if msg.dst_rank == dst_rank and (max_count < 0 or count < max_count):
                    result.append(msg)
                    count += 1
                else:
                    temp_hold.append(msg)
        
        # 把不符合条件的放回去
        for msg in temp_hold:
            self._queues[msg.msg_type].put(msg)
        
        return result
    
    def peek_by_type(self, msg_type: MessageType) -> bool:
        """检查特定类型是否有消息（不取出）"""
        return not self._queues[msg_type].empty()
    
    def empty(self) -> bool:
        """检查所有队列是否为空"""
        return all(q.empty() for q in self._queues.values())
    
    def size(self) -> int:
        """获取队列总大小"""
        return sum(q.qsize() for q in self._queues.values())
    
    def size_by_type(self, msg_type: MessageType) -> int:
        """获取特定类型的消息数量"""
        return self._queues[msg_type].qsize()


# ==================== 序列化工具 ====================

class TensorSerializer:
    """Tensor序列化/反序列化工具"""
    
    @staticmethod
    def serialize(tensor: torch.Tensor) -> bytes:
        """将tensor序列化为bytes"""
        buffer = io.BytesIO()
        torch.save({
            'data': tensor.cpu(),
            'dtype': tensor.dtype,
            'device': str(tensor.device)
        }, buffer)
        data = buffer.getvalue()
        logger = _get_logger()
        logger.debug(
            f"[SERIALIZE] Tensor | shape={list(tensor.shape)}, "
            f"dtype={tensor.dtype}, size={len(data)/1024:.2f}KB"
        )
        return data
    
    @staticmethod
    def deserialize(data: bytes, device: str = 'cuda') -> torch.Tensor:
        """从bytes反序列化为tensor"""
        buffer = io.BytesIO(data)
        saved = torch.load(buffer, weights_only=False)
        tensor = saved['data'].to(device)
        logger = _get_logger()
        logger.debug(
            f"[DESERIALIZE] Tensor | shape={list(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={device}, size={len(data)/1024:.2f}KB"
        )
        return tensor


class MessageSerializer:
    """消息序列化/反序列化工具"""
    
    @staticmethod
    def serialize(msg: Message) -> bytes:
        """序列化消息"""
        logger = _get_logger()
        serialize_start = time.time()
        
        # 处理tensor数据
        data_to_serialize = msg.data
        data_info = get_data_info(msg.data)
        
        if isinstance(msg.data, torch.Tensor):
            data_to_serialize = ('tensor', TensorSerializer.serialize(msg.data))
        elif isinstance(msg.data, tuple) and len(msg.data) == 2:
            # KV cache: (key_tensor, value_tensor)
            if isinstance(msg.data[0], torch.Tensor):
                data_to_serialize = ('kv_cache', 
                    TensorSerializer.serialize(msg.data[0]),
                    TensorSerializer.serialize(msg.data[1]))
        elif isinstance(msg.data, tuple) and len(msg.data) == 4:
            # Draft tokens: 包含多个tensor
            serialized_items = []
            for item in msg.data:
                if isinstance(item, torch.Tensor):
                    serialized_items.append(('tensor', TensorSerializer.serialize(item)))
                else:
                    serialized_items.append(item)
            data_to_serialize = ('draft_tokens', serialized_items)
        
        msg_dict = {
            'msg_type': int(msg.msg_type),
            'src_rank': msg.src_rank,
            'dst_rank': msg.dst_rank,
            'data': data_to_serialize,
            'chunk_idx': msg.chunk_idx,
            'layer_idx': msg.layer_idx,
            'seq_id': msg.seq_id,
            'timestamp': msg.timestamp,
            'original_src': msg.original_src,
        }
        result = pickle.dumps(msg_dict)
        
        serialize_time = (time.time() - serialize_start) * 1000
        logger.debug(
            f"[MSG_SERIALIZE] type={MessageType(msg.msg_type).name}, "
            f"src={msg.src_rank}->dst={msg.dst_rank}, "
            f"layer={msg.layer_idx}, chunk={msg.chunk_idx}, "
            f"data={data_info}, size={len(result)/1024:.2f}KB, "
            f"time={serialize_time:.2f}ms"
        )
        return result
    
    @staticmethod
    def deserialize(data: bytes, device: str = 'cuda') -> Message:
        """反序列化消息"""
        logger = _get_logger()
        deserialize_start = time.time()
        
        msg_dict = pickle.loads(data)
        
        # 处理tensor数据
        raw_data = msg_dict['data']
        if isinstance(raw_data, tuple):
            if raw_data[0] == 'tensor':
                msg_dict['data'] = TensorSerializer.deserialize(raw_data[1], device)
            elif raw_data[0] == 'kv_cache':
                key = TensorSerializer.deserialize(raw_data[1], device)
                value = TensorSerializer.deserialize(raw_data[2], device)
                msg_dict['data'] = (key, value)
            elif raw_data[0] == 'draft_tokens':
                deserialized_items = []
                for item in raw_data[1]:
                    if isinstance(item, tuple) and item[0] == 'tensor':
                        deserialized_items.append(TensorSerializer.deserialize(item[1], device))
                    else:
                        deserialized_items.append(item)
                msg_dict['data'] = tuple(deserialized_items)
        
        msg = Message(
            msg_type=MessageType(msg_dict['msg_type']),
            src_rank=msg_dict['src_rank'],
            dst_rank=msg_dict['dst_rank'],
            data=msg_dict['data'],
            chunk_idx=msg_dict['chunk_idx'],
            layer_idx=msg_dict['layer_idx'],
            seq_id=msg_dict['seq_id'],
            timestamp=msg_dict['timestamp'],
            original_src=msg_dict.get('original_src', -1),
        )
        
        deserialize_time = (time.time() - deserialize_start) * 1000
        data_info = get_data_info(msg.data)
        logger.debug(
            f"[MSG_DESERIALIZE] type={MessageType(msg.msg_type).name}, "
            f"src={msg.src_rank}->dst={msg.dst_rank}, "
            f"layer={msg.layer_idx}, chunk={msg.chunk_idx}, "
            f"data={data_info}, size={len(data)/1024:.2f}KB, "
            f"time={deserialize_time:.2f}ms"
        )
        return msg
    
    @staticmethod
    def serialize_batch(messages: List[Message]) -> bytes:
        """批量序列化消息（用于ring模式聚合发送）"""
        logger = _get_logger()
        batch_start = time.time()
        
        serialized_list = [MessageSerializer.serialize(msg) for msg in messages]
        result = pickle.dumps(serialized_list)
        
        batch_time = (time.time() - batch_start) * 1000
        logger.debug(
            f"[BATCH_SERIALIZE] count={len(messages)}, "
            f"total_size={len(result)/1024:.2f}KB, time={batch_time:.2f}ms"
        )
        return result
    
    @staticmethod
    def deserialize_batch(data: bytes, device: str = 'cuda') -> List[Message]:
        """批量反序列化消息"""
        logger = _get_logger()
        batch_start = time.time()
        
        serialized_list = pickle.loads(data)
        messages = [MessageSerializer.deserialize(s, device) for s in serialized_list]
        
        batch_time = (time.time() - batch_start) * 1000
        logger.debug(
            f"[BATCH_DESERIALIZE] count={len(messages)}, "
            f"size={len(data)/1024:.2f}KB, time={batch_time:.2f}ms"
        )
        return messages

