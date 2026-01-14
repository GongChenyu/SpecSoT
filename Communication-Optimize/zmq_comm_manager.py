"""
基于ZMQ的分布式通信管理器 (重构版 - 继承架构)
支持无线通信场景，使用优先级队列进行数据传输

架构设计：
1. ZMQCommManagerBase: 抽象基类，定义通用接口和公共功能
2. P2PCommManager: 点对点通信模式，直接发送到目标节点
3. RingCommManager: 环形通信模式，只向下一个节点发送，支持消息聚合

特性：
1. 六个队列：3个发送（token, hidden, cache）+ 3个接收（token, hidden, cache）
2. 优先级：token(最高) > hidden state > kv cache(最低)
3. 线程模型：统一收发线程 + 主线程从队列取数据

数据传输时机：
- cache: 每计算一层就传输一层
- hidden: 当前设备计算完成后发送
- token: prefill阶段结束后同步

Ring模式通信流程（以cache同步为例，3个节点）：
- P2P模式: rank0和1相互收发，rank0和2相互收发，rank1和2相互收发
- Ring模式: rank0→rank1→rank2→rank0 环形传递
  * rank0 发送 cache0 给 rank1
  * rank1 收到后转发 cache0，同时发送 cache1 给 rank2  
  * rank2 收到后转发 cache0+cache1，同时发送 cache2 给 rank0
  * 经过 world_size-1 轮后，所有节点都有所有cache
"""

import zmq
import torch
import threading
import queue
import time
import pickle
import io
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict


# ==================== 枚举和数据类 ====================

class MessagePriority(IntEnum):
    """消息优先级（数值越小优先级越高）"""
    TOKEN = 0       # 最高优先级
    HIDDEN = 1      # 中等优先级
    CACHE = 2       # 最低优先级


class MessageType(IntEnum):
    """消息类型"""
    TOKEN = 0
    HIDDEN = 1
    CACHE = 2


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
    
    def __init__(self):
        self._queues: Dict[MessageType, queue.Queue] = {
            MessageType.TOKEN: queue.Queue(),
            MessageType.HIDDEN: queue.Queue(),
            MessageType.CACHE: queue.Queue(),
        }
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
    def put(self, msg: Message):
        """放入消息"""
        msg_type = msg.msg_type
        with self._not_empty:
            self._queues[msg_type].put(msg)
            self._not_empty.notify_all()
    
    def get_by_type(self, msg_type: MessageType, timeout: float = None) -> Optional[Message]:
        """获取特定类型的消息"""
        deadline = time.time() + timeout if timeout else None
        
        while True:
            try:
                # 尝试非阻塞获取
                return self._queues[msg_type].get_nowait()
            except queue.Empty:
                pass
            
            # 检查超时
            if timeout is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
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
        return buffer.getvalue()
    
    @staticmethod
    def deserialize(data: bytes, device: str = 'cuda') -> torch.Tensor:
        """从bytes反序列化为tensor"""
        buffer = io.BytesIO(data)
        saved = torch.load(buffer, weights_only=False)
        tensor = saved['data'].to(device)
        return tensor


class MessageSerializer:
    """消息序列化/反序列化工具"""
    
    @staticmethod
    def serialize(msg: Message) -> bytes:
        """序列化消息"""
        # 处理tensor数据
        data_to_serialize = msg.data
        if isinstance(msg.data, torch.Tensor):
            data_to_serialize = ('tensor', TensorSerializer.serialize(msg.data))
        elif isinstance(msg.data, tuple) and len(msg.data) == 2:
            # KV cache: (key_tensor, value_tensor)
            if isinstance(msg.data[0], torch.Tensor):
                data_to_serialize = ('kv_cache', 
                    TensorSerializer.serialize(msg.data[0]),
                    TensorSerializer.serialize(msg.data[1]))
        
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
        return pickle.dumps(msg_dict)
    
    @staticmethod
    def deserialize(data: bytes, device: str = 'cuda') -> Message:
        """反序列化消息"""
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
        
        return Message(
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
    
    @staticmethod
    def serialize_batch(messages: List[Message]) -> bytes:
        """批量序列化消息（用于ring模式聚合发送）"""
        serialized_list = [MessageSerializer.serialize(msg) for msg in messages]
        return pickle.dumps(serialized_list)
    
    @staticmethod
    def deserialize_batch(data: bytes, device: str = 'cuda') -> List[Message]:
        """批量反序列化消息"""
        serialized_list = pickle.loads(data)
        return [MessageSerializer.deserialize(s, device) for s in serialized_list]


# ==================== 抽象基类 ====================

class ZMQCommManagerBase(ABC):
    """
    ZMQ通信管理器基类
    
    定义通用接口和公共功能，具体的发送策略由子类实现
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        base_port: int = 29500,
        device: str = "cuda",
        node_addresses: Optional[Dict[int, str]] = None,
    ):
        """
        初始化ZMQ通信管理器
        
        Args:
            rank: 当前设备rank
            world_size: 总设备数
            base_port: 基础端口号
            device: 设备类型
            node_addresses: 节点地址映射 {rank: ip}，默认全部使用localhost
        """
        self.rank = rank
        self.world_size = world_size
        self.base_port = base_port
        self.device = device
        
        # 节点地址配置
        if node_addresses is None:
            self.node_addresses = {r: "127.0.0.1" for r in range(world_size)}
        else:
            self.node_addresses = node_addresses
        
        # 设置logger
        self.logger = logging.getLogger(f"ZMQComm-Rank{rank}")
        
        # ZMQ上下文
        self.context = zmq.Context()
        
        # 发送队列（统一，按消息类型分）
        self.send_queue = ThreadSafeQueue()
        
        # 接收队列（统一，按消息类型分）
        self.recv_queue = ThreadSafeQueue()
        
        # 发送socket（到其他rank）
        self.send_sockets: Dict[int, zmq.Socket] = {}
        
        # 接收socket（从其他rank）
        self.recv_sockets: Dict[int, zmq.Socket] = {}
        
        # 工作线程
        self.send_thread: Optional[threading.Thread] = None
        self.recv_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # 序列ID计数器
        self.seq_counter = 0
        self._seq_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'token_sent': 0,
            'token_recv': 0,
            'hidden_sent': 0,
            'hidden_recv': 0,
            'cache_sent': 0,
            'cache_recv': 0,
            'aggregated_sends': 0,  # ring模式下聚合发送次数
            'forwarded_messages': 0,  # ring模式下转发消息数
        }
        
        # 初始化socket连接（由子类决定具体连接方式）
        self._setup_sockets()
    
    def _get_next_seq_id(self) -> int:
        """获取下一个序列ID"""
        with self._seq_lock:
            self.seq_counter += 1
            return self.seq_counter
    
    def _get_port(self, sender_rank: int, receiver_rank: int) -> int:
        """计算两个rank之间通信使用的端口"""
        return self.base_port + sender_rank * self.world_size + receiver_rank
    
    @abstractmethod
    def _setup_sockets(self):
        """设置ZMQ socket连接（由子类实现）"""
        pass
    
    @abstractmethod
    def _send_worker(self):
        """发送工作线程（由子类实现）"""
        pass
    
    def _recv_worker(self):
        """
        统一接收工作线程
        持续监听所有recv_socket，收到消息后根据类型放入对应的recv_queue
        """
        self.logger.info("接收线程启动")
        
        # 创建poller监听所有接收socket
        poller = zmq.Poller()
        for rank, socket in self.recv_sockets.items():
            poller.register(socket, zmq.POLLIN)
        
        while self.is_running:
            try:
                socks = dict(poller.poll(100))
                
                for rank, socket in self.recv_sockets.items():
                    if socket in socks:
                        try:
                            data = socket.recv(zmq.NOBLOCK)
                            self._process_received_data(data)
                        except zmq.Again:
                            continue
                        except Exception as e:
                            self.logger.error(f"接收消息出错: {e}", exc_info=True)
                            
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"接收线程出错: {e}", exc_info=True)
        
        self.logger.info("接收线程退出")
    
    def _process_received_data(self, data: bytes):
        """
        处理接收到的原始数据
        子类可以重写此方法来处理批量消息
        """
        msg = MessageSerializer.deserialize(data, self.device)
        self._handle_received_message(msg)
    
    def _handle_received_message(self, msg: Message):
        """处理单条接收到的消息"""
        # 放入接收队列
        self.recv_queue.put(msg)
        
        # 更新统计
        if msg.msg_type == MessageType.TOKEN:
            self.stats['token_recv'] += 1
            self.logger.debug(f"接收token from rank {msg.get_effective_src()}")
        elif msg.msg_type == MessageType.HIDDEN:
            self.stats['hidden_recv'] += 1
            self.logger.debug(f"接收hidden from rank {msg.get_effective_src()}, chunk={msg.chunk_idx}")
        elif msg.msg_type == MessageType.CACHE:
            self.stats['cache_recv'] += 1
            self.logger.debug(f"接收cache from rank {msg.get_effective_src()}, layer={msg.layer_idx}, chunk={msg.chunk_idx}")
    
    def start(self):
        """启动通信管理器"""
        if self.is_running:
            self.logger.warning("通信管理器已经在运行")
            return
        
        self.is_running = True
        
        # 启动接收线程
        self.recv_thread = threading.Thread(
            target=self._recv_worker,
            daemon=True,
            name=f"Recv-Rank{self.rank}"
        )
        
        # 启动发送线程
        self.send_thread = threading.Thread(
            target=self._send_worker, 
            daemon=True,
            name=f"Send-Rank{self.rank}"
        )
        
        self.recv_thread.start()
        self.send_thread.start()
        
        self.logger.info(f"通信管理器已启动 ({self.__class__.__name__})")
    
    def stop(self):
        """停止通信管理器"""
        self.is_running = False
        
        # 等待线程结束
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=5.0)
        if self.recv_thread and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=5.0)
        
        # 关闭所有socket
        for socket in self.send_sockets.values():
            socket.close()
        for socket in self.recv_sockets.values():
            socket.close()
        
        self.context.term()
        self.logger.info("通信管理器已停止")
    
    # ==================== Token 通信 ====================
    
    @abstractmethod
    def send_token(self, token: torch.Tensor, dst_rank: int = -1) -> None:
        """发送token"""
        pass
    
    def recv_token(self, src_rank: int = -1, timeout: float = 30.0) -> Optional[torch.Tensor]:
        """
        接收token（从接收队列获取，阻塞等待）
        
        Args:
            src_rank: 源rank，-1表示从任意rank接收
            timeout: 超时时间（秒）
            
        Returns:
            接收到的token tensor
        """
        deadline = time.time() + timeout
        pending = []
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.TOKEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            effective_src = msg.get_effective_src()
            if src_rank == -1 or effective_src == src_rank:
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return msg.data
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(f"接收token超时")
        return None
    
    def broadcast_token(self, token: torch.Tensor) -> torch.Tensor:
        """
        广播token（所有rank都调用）
        使用最后一个rank作为源，其他rank接收
        """
        src_rank = self.world_size - 1
        
        if self.rank == src_rank:
            self.send_token(token, dst_rank=-1)
            return token
        else:
            received = self.recv_token(src_rank=src_rank)
            return received
    
    # ==================== Hidden State 通信 ====================
    
    @abstractmethod
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """发送hidden state"""
        pass
    
    def recv_hidden(self, src_rank: int, timeout: float = 30.0) -> Optional[Tuple[torch.Tensor, int]]:
        """
        接收hidden state（从接收队列获取，阻塞等待）
        
        Args:
            src_rank: 源rank
            timeout: 超时时间（秒）
            
        Returns:
            (hidden_tensor, chunk_idx) 或 None
        """
        deadline = time.time() + timeout
        pending = []
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.HIDDEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            effective_src = msg.get_effective_src()
            if effective_src == src_rank:
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return (msg.data, msg.chunk_idx)
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(f"从rank {src_rank}接收hidden超时")
        return None
    
    # ==================== Cache 通信 ====================
    
    @abstractmethod
    def send_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """异步发送cache"""
        pass
    
    def get_received_cache(
        self, 
        timeout: float = 0.1
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], int, int, int]]:
        """
        从接收队列获取cache（非阻塞）
        
        Returns:
            (kv_cache, src_rank, layer_idx, chunk_idx) 或 None
        """
        msg = self.recv_queue.get_by_type(MessageType.CACHE, timeout=timeout)
        if msg:
            return (msg.data, msg.get_effective_src(), msg.layer_idx, msg.chunk_idx)
        return None
    
    def wait_for_cache(
        self,
        src_rank: int,
        layer_idx: int,
        chunk_idx: int,
        timeout: float = 30.0
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        等待特定的cache（用于需要同步等待的场景）
        """
        deadline = time.time() + timeout
        pending = []
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.CACHE, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            effective_src = msg.get_effective_src()
            if (effective_src == src_rank and 
                msg.layer_idx == layer_idx and 
                msg.chunk_idx == chunk_idx):
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return msg.data
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(
            f"等待cache超时: src={src_rank}, layer={layer_idx}, chunk={chunk_idx}"
        )
        return None
    
    # ==================== 辅助方法 ====================
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
    
    def has_pending_send(self) -> bool:
        """检查是否有待发送的消息"""
        return not self.send_queue.empty()
    
    def get_pending_send_count(self) -> int:
        """获取待发送消息数量"""
        return self.send_queue.size()
    
    def get_recv_queue_count(self, msg_type: MessageType = None) -> int:
        """获取接收队列中的消息数量"""
        if msg_type:
            return self.recv_queue.size_by_type(msg_type)
        return self.recv_queue.size()


# ==================== P2P通信管理器 ====================

class P2PCommManager(ZMQCommManagerBase):
    """
    点对点通信管理器
    
    特点：
    - 每个rank可以直接向任意其他rank发送消息
    - 消息立即发送，不做聚合
    - 适用于网络质量好、延迟低的场景
    
    通信拓扑（3节点示例）：
        rank0 <---> rank1
        rank0 <---> rank2
        rank1 <---> rank2
    """
    
    def _setup_sockets(self):
        """设置P2P模式的socket连接"""
        self.logger.info("设置P2P模式 ZMQ sockets")
        
        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue
            
            # 发送socket（PUSH）
            send_socket = self.context.socket(zmq.PUSH)
            send_port = self._get_port(self.rank, other_rank)
            addr = self.node_addresses[other_rank]
            send_socket.connect(f"tcp://{addr}:{send_port}")
            send_socket.setsockopt(zmq.SNDHWM, 1000)
            send_socket.setsockopt(zmq.LINGER, 0)
            self.send_sockets[other_rank] = send_socket
            self.logger.debug(f"连接发送socket到 rank {other_rank}: tcp://{addr}:{send_port}")
            
            # 接收socket（PULL）
            recv_socket = self.context.socket(zmq.PULL)
            recv_port = self._get_port(other_rank, self.rank)
            recv_socket.setsockopt(zmq.LINGER, 0)
            recv_socket.bind(f"tcp://*:{recv_port}")
            recv_socket.setsockopt(zmq.RCVHWM, 1000)
            self.recv_sockets[other_rank] = recv_socket
            self.logger.debug(f"绑定接收socket从 rank {other_rank}: tcp://*:{recv_port}")
        
        self.logger.info("P2P模式 ZMQ sockets设置完成")
    
    def _send_worker(self):
        """
        P2P模式发送工作线程
        按优先级从send_queue取消息并立即发送
        """
        self.logger.info("P2P发送线程启动")
        
        while self.is_running:
            try:
                msg = self.send_queue.get_by_priority(timeout=0.1)
                
                if msg is None:
                    continue
                
                dst_rank = msg.dst_rank
                serialized = MessageSerializer.serialize(msg)
                self.send_sockets[dst_rank].send(serialized)
                
                # 更新统计
                if msg.msg_type == MessageType.TOKEN:
                    self.stats['token_sent'] += 1
                    self.logger.debug(f"发送token到 rank {dst_rank}")
                elif msg.msg_type == MessageType.HIDDEN:
                    self.stats['hidden_sent'] += 1
                    self.logger.debug(f"发送hidden到 rank {dst_rank}, chunk={msg.chunk_idx}")
                elif msg.msg_type == MessageType.CACHE:
                    self.stats['cache_sent'] += 1
                    self.logger.debug(f"发送cache到 rank {dst_rank}, layer={msg.layer_idx}, chunk={msg.chunk_idx}")
                    
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"P2P发送线程出错: {e}", exc_info=True)
        
        self.logger.info("P2P发送线程退出")
    
    def send_token(self, token: torch.Tensor, dst_rank: int = -1) -> None:
        """发送token"""
        if dst_rank == -1:
            # 广播到所有其他rank
            for other_rank in range(self.world_size):
                if other_rank != self.rank:
                    self._queue_message(MessageType.TOKEN, token, other_rank)
        else:
            self._queue_message(MessageType.TOKEN, token, dst_rank)
    
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """发送hidden state"""
        self._queue_message(MessageType.HIDDEN, hidden, dst_rank, chunk_idx=chunk_idx)
    
    def send_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """异步发送cache"""
        self._queue_message(MessageType.CACHE, kv_cache, dst_rank, 
                           layer_idx=layer_idx, chunk_idx=chunk_idx)
        self.logger.debug(f"Cache入队: layer={layer_idx}, chunk={chunk_idx} -> rank {dst_rank}")
    
    def _queue_message(
        self,
        msg_type: MessageType,
        data: Any,
        dst_rank: int,
        chunk_idx: int = -1,
        layer_idx: int = -1
    ):
        """将消息放入发送队列"""
        msg = Message(
            msg_type=msg_type,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=data,
            chunk_idx=chunk_idx,
            layer_idx=layer_idx,
            seq_id=self._get_next_seq_id()
        )
        self.send_queue.put(msg)


# ==================== Ring通信管理器 ====================

class RingCommManager(ZMQCommManagerBase):
    """
    环形通信管理器
    
    特点：
    - 每个rank只向下一个rank发送消息（rank0→rank1→rank2→rank0）
    - 支持消息聚合：扫描发送队列中目的地相同的消息，合并发送
    - 非阻塞聚合：不刻意等待，有什么发什么
    - 支持消息转发：收到的非本节点消息会转发给下一个节点
    
    通信拓扑（3节点示例）：
        rank0 --> rank1 --> rank2 --> rank0
           ^                           |
           +---------------------------+
    
    环形通信流程（以cache同步为例，3个节点）：
    - rank0 发送 cache0 给 rank1
    - rank1 收到后转发 cache0，同时发送自己的 cache1 给 rank2
    - rank2 收到后转发 cache0+cache1，同时发送 cache2 给 rank0
    - 经过 world_size-1 轮后，所有节点都有所有cache
    
    优势：
    - 减少通信连接数：P2P需要n*(n-1)条连接，Ring只需要n条
    - 减少通信干扰：避免多条链路同时传输
    - 支持聚合发送：减少通信次数
    """
    
    def __init__(self, *args, **kwargs):
        # Ring模式特有的属性（在父类__init__之前设置）
        self.next_rank = None
        self.prev_rank = None
        
        # 转发队列：存储需要转发的消息
        self.forward_queue = ThreadSafeQueue()
        
        super().__init__(*args, **kwargs)
        
        # 设置环形拓扑
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        
        self.logger.info(f"Ring拓扑: prev={self.prev_rank} <- rank{self.rank} -> next={self.next_rank}")
    
    def _setup_sockets(self):
        """设置Ring模式的socket连接（只连接相邻节点）"""
        self.logger.info("设置Ring模式 ZMQ sockets")
        
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        
        # 发送socket：只连接到下一个rank
        send_socket = self.context.socket(zmq.PUSH)
        send_port = self._get_port(self.rank, next_rank)
        addr = self.node_addresses[next_rank]
        send_socket.connect(f"tcp://{addr}:{send_port}")
        send_socket.setsockopt(zmq.SNDHWM, 1000)
        send_socket.setsockopt(zmq.LINGER, 0)
        self.send_sockets[next_rank] = send_socket
        self.logger.debug(f"连接发送socket到 next_rank {next_rank}: tcp://{addr}:{send_port}")
        
        # 接收socket：只接收来自上一个rank
        recv_socket = self.context.socket(zmq.PULL)
        recv_port = self._get_port(prev_rank, self.rank)
        recv_socket.setsockopt(zmq.LINGER, 0)
        recv_socket.bind(f"tcp://*:{recv_port}")
        recv_socket.setsockopt(zmq.RCVHWM, 1000)
        self.recv_sockets[prev_rank] = recv_socket
        self.logger.debug(f"绑定接收socket从 prev_rank {prev_rank}: tcp://*:{recv_port}")
        
        self.logger.info(f"Ring模式 ZMQ sockets设置完成 (prev={prev_rank}, next={next_rank})")
    
    def _process_received_data(self, data: bytes):
        """
        处理接收到的数据
        Ring模式支持批量消息
        """
        try:
            # 尝试批量反序列化
            messages = MessageSerializer.deserialize_batch(data, self.device)
            for msg in messages:
                self._handle_received_message_ring(msg)
        except:
            # 回退到单消息模式
            msg = MessageSerializer.deserialize(data, self.device)
            self._handle_received_message_ring(msg)
    
    def _handle_received_message_ring(self, msg: Message):
        """
        Ring模式处理接收到的消息
        
        1. 如果消息的原始来源是自己，说明已经转了一圈，不再转发
        2. 否则，将消息放入接收队列，同时放入转发队列
        """
        effective_src = msg.get_effective_src()
        
        # 放入接收队列供本地使用
        self.recv_queue.put(msg)
        
        # 更新统计
        if msg.msg_type == MessageType.TOKEN:
            self.stats['token_recv'] += 1
        elif msg.msg_type == MessageType.HIDDEN:
            self.stats['hidden_recv'] += 1
        elif msg.msg_type == MessageType.CACHE:
            self.stats['cache_recv'] += 1
        
        self.logger.debug(f"接收消息 from original_src={effective_src}, type={msg.msg_type.name}")
        
        # 判断是否需要转发（消息未回到原点）
        if effective_src != self.rank:
            # 创建转发消息
            forward_msg = Message(
                msg_type=msg.msg_type,
                src_rank=self.rank,  # 当前发送者
                dst_rank=self.next_rank,
                data=msg.data,
                chunk_idx=msg.chunk_idx,
                layer_idx=msg.layer_idx,
                seq_id=self._get_next_seq_id(),
                original_src=effective_src,  # 保持原始来源
            )
            self.forward_queue.put(forward_msg)
            self.stats['forwarded_messages'] += 1
            self.logger.debug(f"转发消息: original_src={effective_src} -> next={self.next_rank}")
    
    def _send_worker(self):
        """
        Ring模式发送工作线程
        
        特点：
        1. 按优先级处理消息
        2. 聚合相同目的地的消息（非阻塞）
        3. 处理转发队列中的消息
        """
        self.logger.info("Ring发送线程启动")
        
        while self.is_running:
            try:
                messages_to_send = []
                
                # 1. 优先处理转发队列（优先级最高，保证数据流动）
                while True:
                    forward_msg = self.forward_queue.get_by_priority(timeout=0.001)
                    if forward_msg is None:
                        break
                    messages_to_send.append(forward_msg)
                
                # 2. 从发送队列获取消息（按优先级）
                # 聚合所有发往next_rank的消息
                while True:
                    msg = self.send_queue.get_by_priority(timeout=0.001)
                    if msg is None:
                        break
                    # Ring模式：所有消息都发往next_rank
                    msg.dst_rank = self.next_rank
                    messages_to_send.append(msg)
                
                # 3. 如果有消息要发送，批量发送
                if messages_to_send:
                    self._send_aggregated(messages_to_send)
                else:
                    # 没有消息，短暂等待
                    time.sleep(0.01)
                    
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Ring发送线程出错: {e}", exc_info=True)
        
        self.logger.info("Ring发送线程退出")
    
    def _send_aggregated(self, messages: List[Message]):
        """
        聚合发送消息
        
        Args:
            messages: 待发送的消息列表
        """
        if not messages:
            return
        
        # 按优先级排序
        messages.sort(key=lambda m: (m.priority, m.timestamp))
        
        # 批量序列化
        serialized = MessageSerializer.serialize_batch(messages)
        
        # 发送到next_rank
        self.send_sockets[self.next_rank].send(serialized)
        
        # 更新统计
        for msg in messages:
            if msg.msg_type == MessageType.TOKEN:
                self.stats['token_sent'] += 1
            elif msg.msg_type == MessageType.HIDDEN:
                self.stats['hidden_sent'] += 1
            elif msg.msg_type == MessageType.CACHE:
                self.stats['cache_sent'] += 1
        
        self.stats['aggregated_sends'] += 1
        self.logger.debug(f"聚合发送 {len(messages)} 条消息到 rank {self.next_rank}")
    
    def send_token(self, token: torch.Tensor, dst_rank: int = -1) -> None:
        """
        发送token
        
        Ring模式：广播通过环形传递实现
        dst_rank参数用于兼容接口，但实际只发给next_rank
        """
        # 设置original_src为自己，当消息转回来时停止转发
        msg = Message(
            msg_type=MessageType.TOKEN,
            src_rank=self.rank,
            dst_rank=self.next_rank,
            data=token,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank,
        )
        self.send_queue.put(msg)
    
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """发送hidden state"""
        msg = Message(
            msg_type=MessageType.HIDDEN,
            src_rank=self.rank,
            dst_rank=self.next_rank,
            data=hidden,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank,
        )
        self.send_queue.put(msg)
    
    def send_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """
        异步发送cache
        
        Ring模式：dst_rank参数被忽略，所有cache都通过环形传递
        发送一次后，消息会自动在环中传播到所有节点
        """
        msg = Message(
            msg_type=MessageType.CACHE,
            src_rank=self.rank,
            dst_rank=self.next_rank,
            data=kv_cache,
            layer_idx=layer_idx,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank,
        )
        self.send_queue.put(msg)
        self.logger.debug(f"Cache入队(Ring): layer={layer_idx}, chunk={chunk_idx}")


# ==================== 工厂函数 ====================

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
ZMQCommManager = P2PCommManager
