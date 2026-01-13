"""
基于ZMQ的分布式通信管理器 (重构版)
支持无线通信场景，使用优先级队列进行数据传输

核心改进：
1. 统一接收线程：持续监听所有socket，收到消息后根据类型放入对应队列
2. 统一发送线程：按优先级从发送队列取消息并发送
3. 主线程的recv_xxx：从对应的接收队列中取数据（阻塞等待）

特性：
1. 六个队列：3个发送（token, hidden, cache）+ 3个接收（token, hidden, cache）
2. 优先级：token(最高) > hidden state > kv cache(最低)
3. 两种通信模式：pairwise（点对点）和 ring（环形聚合）
4. 线程模型：统一收发线程 + 主线程从队列取数据

数据传输时机：
- cache: 每计算一层就传输一层
- hidden: 当前设备计算完成后发送
- token: prefill阶段结束后同步
"""

import zmq
import torch
import threading
import queue
import time
import pickle
import io
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict


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
    src_rank: int                   # 源设备rank
    dst_rank: int                   # 目标设备rank
    data: Any                       # 数据（tensor或其他）
    chunk_idx: int = -1             # chunk索引（用于SP）
    layer_idx: int = -1             # 层索引（用于cache和PP）
    seq_id: int = 0                 # 序列ID（用于排序和追踪）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    @property
    def priority(self) -> int:
        """获取消息优先级"""
        return MessagePriority(self.msg_type).value


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


class TensorSerializer:
    """Tensor序列化/反序列化工具"""
    
    @staticmethod
    def serialize(tensor: torch.Tensor) -> bytes:
        """将tensor序列化为bytes"""
        buffer = io.BytesIO()
        # 保存tensor的元数据和数据
        torch.save({
            'data': tensor.cpu(),  # 先移到CPU
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
            'timestamp': msg.timestamp
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
            timestamp=msg_dict['timestamp']
        )


class ZMQCommManager:
    """
    基于ZMQ的通信管理器 (重构版)
    
    核心改进：
    - 统一接收线程：持续监听所有socket，收到消息后根据类型放入对应队列
    - 统一发送线程：按优先级从发送队列取消息并发送
    - 主线程的recv_xxx：从对应的接收队列中取数据（阻塞等待）
    
    通信模式：
    - pairwise: 直接点对点发送，任务完成即发送
    - ring: 环形聚合通信，同目的地数据合并发送
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        base_port: int = 29500,
        mode: str = "pairwise",  # "pairwise" or "ring"
        device: str = "cuda",
        node_addresses: Optional[Dict[int, str]] = None,  # rank -> ip地址
    ):
        """
        初始化ZMQ通信管理器
        
        Args:
            rank: 当前设备rank
            world_size: 总设备数
            base_port: 基础端口号
            mode: 通信模式 ("pairwise" 或 "ring")
            device: 设备类型
            node_addresses: 节点地址映射 {rank: ip}，默认全部使用localhost
        """
        self.rank = rank
        self.world_size = world_size
        self.base_port = base_port
        self.mode = mode
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
        
        # 发送socket（到其他每个rank）
        self.send_sockets: Dict[int, zmq.Socket] = {}
        
        # 接收socket（从其他每个rank）
        self.recv_sockets: Dict[int, zmq.Socket] = {}
        
        # 工作线程
        self.send_thread: Optional[threading.Thread] = None  # 统一发送线程
        self.recv_thread: Optional[threading.Thread] = None  # 统一接收线程
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
        }
        
        # 初始化socket连接
        self._setup_sockets()
        
    def _get_next_seq_id(self) -> int:
        """获取下一个序列ID"""
        with self._seq_lock:
            self.seq_counter += 1
            return self.seq_counter
    
    def _get_port(self, sender_rank: int, receiver_rank: int) -> int:
        """计算两个rank之间通信使用的端口"""
        # 每对rank使用唯一端口
        return self.base_port + sender_rank * self.world_size + receiver_rank
    
    def _setup_sockets(self):
        """设置ZMQ socket连接"""
        self.logger.info(f"设置ZMQ sockets (mode={self.mode})")
        
        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue
            
            # 发送socket（PUSH）
            send_socket = self.context.socket(zmq.PUSH)
            send_port = self._get_port(self.rank, other_rank)
            addr = self.node_addresses[other_rank]
            send_socket.connect(f"tcp://{addr}:{send_port}")
            # 设置高水位标记，避免消息积压
            send_socket.setsockopt(zmq.SNDHWM, 1000)
            send_socket.setsockopt(zmq.LINGER, 0)
            self.send_sockets[other_rank] = send_socket
            self.logger.debug(f"连接发送socket到 rank {other_rank}: tcp://{addr}:{send_port}")
            
            # 接收socket（PULL）
            recv_socket = self.context.socket(zmq.PULL)
            recv_port = self._get_port(other_rank, self.rank)
            # 设置端口重用（避免TIME_WAIT状态导致的端口冲突）
            recv_socket.setsockopt(zmq.LINGER, 0)
            recv_socket.bind(f"tcp://*:{recv_port}")
            # 设置高水位标记
            recv_socket.setsockopt(zmq.RCVHWM, 1000)
            self.recv_sockets[other_rank] = recv_socket
            self.logger.debug(f"绑定接收socket从 rank {other_rank}: tcp://*:{recv_port}")
        
        self.logger.info(f"ZMQ sockets设置完成")
    
    def start(self):
        """启动通信管理器"""
        if self.is_running:
            self.logger.warning("通信管理器已经在运行")
            return
        
        self.is_running = True
        
        # 启动统一接收线程（持续监听所有socket）
        self.recv_thread = threading.Thread(
            target=self._unified_recv_worker,
            daemon=True,
            name=f"UnifiedRecv-Rank{self.rank}"
        )
        
        # 启动统一发送线程（按优先级发送）
        self.send_thread = threading.Thread(
            target=self._unified_send_worker, 
            daemon=True,
            name=f"UnifiedSend-Rank{self.rank}"
        )
        
        self.recv_thread.start()
        self.send_thread.start()
        
        self.logger.info("通信管理器已启动（统一收发模式）")
    
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
    
    # ==================== 统一接收线程 ====================
    
    def _unified_recv_worker(self):
        """
        统一接收工作线程
        持续监听所有recv_socket，收到消息后根据类型放入对应的recv_queue
        """
        self.logger.info("统一接收线程启动")
        
        # 创建poller监听所有接收socket
        poller = zmq.Poller()
        for rank, socket in self.recv_sockets.items():
            poller.register(socket, zmq.POLLIN)
        
        while self.is_running:
            try:
                # 等待消息（100ms超时，以便检查is_running）
                socks = dict(poller.poll(100))
                
                for rank, socket in self.recv_sockets.items():
                    if socket in socks:
                        # 接收消息
                        try:
                            data = socket.recv(zmq.NOBLOCK)
                            msg = MessageSerializer.deserialize(data, self.device)
                            
                            # 根据消息类型放入对应队列
                            self.recv_queue.put(msg)
                            
                            # 更新统计
                            if msg.msg_type == MessageType.TOKEN:
                                self.stats['token_recv'] += 1
                                self.logger.debug(f"接收token from rank {msg.src_rank}")
                            elif msg.msg_type == MessageType.HIDDEN:
                                self.stats['hidden_recv'] += 1
                                self.logger.debug(f"接收hidden from rank {msg.src_rank}, chunk={msg.chunk_idx}")
                            elif msg.msg_type == MessageType.CACHE:
                                self.stats['cache_recv'] += 1
                                self.logger.debug(f"接收cache from rank {msg.src_rank}, layer={msg.layer_idx}, chunk={msg.chunk_idx}")
                                
                        except zmq.Again:
                            continue
                        except Exception as e:
                            self.logger.error(f"接收消息出错: {e}", exc_info=True)
                            
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"统一接收线程出错: {e}", exc_info=True)
        
        self.logger.info("统一接收线程退出")
    
    # ==================== 统一发送线程 ====================
    
    def _unified_send_worker(self):
        """
        统一发送工作线程
        按优先级从send_queue取消息并发送
        优先级: TOKEN > HIDDEN > CACHE
        """
        self.logger.info("统一发送线程启动")
        
        while self.is_running:
            try:
                # 按优先级获取消息
                msg = self.send_queue.get_by_priority(timeout=0.1)
                
                if msg is None:
                    continue
                
                # 发送消息
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
                    self.logger.error(f"统一发送线程出错: {e}", exc_info=True)
        
        self.logger.info("统一发送线程退出")
    
    # ==================== Token 通信 ====================
    
    def send_token(self, token: torch.Tensor, dst_rank: int = -1) -> None:
        """
        发送token（放入发送队列，由发送线程异步发送）
        
        Args:
            token: token tensor
            dst_rank: 目标rank，-1表示广播到所有其他rank
        """
        if dst_rank == -1:
            # 广播到所有其他rank
            for other_rank in range(self.world_size):
                if other_rank != self.rank:
                    self._queue_token(token, other_rank)
        else:
            self._queue_token(token, dst_rank)
    
    def _queue_token(self, token: torch.Tensor, dst_rank: int):
        """将token消息放入发送队列"""
        msg = Message(
            msg_type=MessageType.TOKEN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=token,
            seq_id=self._get_next_seq_id()
        )
        self.send_queue.put(msg)
    
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
        pending = []  # 暂存不匹配的消息
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.TOKEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            # 检查src_rank
            if src_rank == -1 or msg.src_rank == src_rank:
                # 找到目标消息，先把暂存的放回去
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return msg.data
            else:
                # 不是期望的源，暂存
                pending.append(msg)
        
        # 超时，把暂存的放回去
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(f"接收token超时")
        return None
    
    def broadcast_token(self, token: torch.Tensor) -> torch.Tensor:
        """
        广播token（所有rank都调用，实现类似torch.distributed.broadcast的效果）
        
        使用最后一个rank作为源，其他rank接收
        
        Args:
            token: 源rank提供的token，其他rank可传None
            
        Returns:
            同步后的token
        """
        src_rank = self.world_size - 1  # 最后一个rank作为源
        
        if self.rank == src_rank:
            # 源rank：发送给所有其他rank
            self.send_token(token, dst_rank=-1)
            return token
        else:
            # 其他rank：接收token
            received = self.recv_token(src_rank=src_rank)
            return received
    
    # ==================== Hidden State 通信 ====================
    
    def send_hidden(
        self, 
        hidden: torch.Tensor, 
        dst_rank: int,
        chunk_idx: int = -1
    ) -> None:
        """
        发送hidden state（放入发送队列，由发送线程异步发送）
        
        Args:
            hidden: hidden state tensor
            dst_rank: 目标rank
            chunk_idx: chunk索引
        """
        msg = Message(
            msg_type=MessageType.HIDDEN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=hidden,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id()
        )
        self.send_queue.put(msg)
    
    def recv_hidden(
        self, 
        src_rank: int, 
        timeout: float = 30.0
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """
        接收hidden state（从接收队列获取，阻塞等待）
        
        Args:
            src_rank: 源rank
            timeout: 超时时间（秒）
            
        Returns:
            (hidden_tensor, chunk_idx) 或 None
        """
        deadline = time.time() + timeout
        pending = []  # 暂存不匹配的消息
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.HIDDEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            # 检查src_rank
            if msg.src_rank == src_rank:
                # 找到目标消息，先把暂存的放回去
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return (msg.data, msg.chunk_idx)
            else:
                # 不是期望的源，暂存
                pending.append(msg)
        
        # 超时，把暂存的放回去
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(f"从rank {src_rank}接收hidden超时")
        return None
    
    # ==================== Cache 通信 ====================
    
    def send_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """
        异步发送cache（放入发送队列）
        
        Args:
            kv_cache: (key_cache, value_cache) tuple
            dst_rank: 目标rank
            layer_idx: 层索引
            chunk_idx: chunk索引
        """
        msg = Message(
            msg_type=MessageType.CACHE,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=kv_cache,
            layer_idx=layer_idx,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id()
        )
        self.send_queue.put(msg)
        self.logger.debug(f"Cache入队: layer={layer_idx}, chunk={chunk_idx} -> rank {dst_rank}")
    
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
            return (msg.data, msg.src_rank, msg.layer_idx, msg.chunk_idx)
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
        
        Args:
            src_rank: 源rank
            layer_idx: 层索引
            chunk_idx: chunk索引
            timeout: 超时时间
            
        Returns:
            kv_cache 或 None
        """
        deadline = time.time() + timeout
        pending = []  # 暂存不匹配的消息
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.CACHE, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            if (msg.src_rank == src_rank and 
                msg.layer_idx == layer_idx and 
                msg.chunk_idx == chunk_idx):
                # 找到目标消息，先把暂存的放回去
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                return msg.data
            else:
                # 不匹配，暂存
                pending.append(msg)
        
        # 超时，把暂存的放回去
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        self.logger.warning(
            f"等待cache超时: src={src_rank}, layer={layer_idx}, chunk={chunk_idx}"
        )
        return None
    
    # ==================== Ring模式的Cache同步（带聚合） ====================
    
    def ring_sync_cache(
        self,
        local_cache: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        chunk_idx: int
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Ring模式的cache同步
        
        环形传递，每个rank收集所有其他rank的cache
        支持聚合：如果有多层cache待发送，可以合并
        
        Args:
            local_cache: 本地的kv cache
            layer_idx: 层索引
            chunk_idx: chunk索引
            
        Returns:
            所有rank的cache字典 {rank: kv_cache}
        """
        all_caches = {self.rank: local_cache}
        
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        
        # 需要world_size-1轮传递
        current_cache_from = self.rank
        
        for step in range(self.world_size - 1):
            # 发送当前持有的cache（来自current_cache_from）
            cache_to_send = all_caches[current_cache_from]
            
            # 直接发送（不经过队列，因为这是同步操作）
            msg = Message(
                msg_type=MessageType.CACHE,
                src_rank=current_cache_from,  # 原始来源
                dst_rank=next_rank,
                data=cache_to_send,
                layer_idx=layer_idx,
                chunk_idx=chunk_idx,
                seq_id=self._get_next_seq_id()
            )
            
            serialized = MessageSerializer.serialize(msg)
            self.send_sockets[next_rank].send(serialized)
            
            # 等待接收来自prev_rank的cache
            received = self.wait_for_cache(
                src_rank=-1,  # 可能来自任何原始源
                layer_idx=layer_idx,
                chunk_idx=chunk_idx,
                timeout=30.0
            )
            
            if received:
                # 这里需要更复杂的逻辑来追踪原始来源
                # 简化处理：按步骤推断来源
                original_src = (current_cache_from - 1 + self.world_size) % self.world_size
                all_caches[original_src] = received
                current_cache_from = original_src
            else:
                self.logger.error(f"Ring同步超时: step={step}")
                break
        
        return all_caches
    
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


# ==================== 工厂函数 ====================

def create_zmq_comm_manager(
    rank: int,
    world_size: int,
    base_port: int = 29500,
    mode: str = "pairwise",
    device: str = "cuda",
    node_addresses: Optional[Dict[int, str]] = None
) -> ZMQCommManager:
    """
    创建ZMQ通信管理器
    
    Args:
        rank: 当前设备rank
        world_size: 总设备数
        base_port: 基础端口号
        mode: 通信模式 ("pairwise" 或 "ring")
        device: 设备类型
        node_addresses: 节点地址映射
        
    Returns:
        ZMQCommManager实例
    """
    manager = ZMQCommManager(
        rank=rank,
        world_size=world_size,
        base_port=base_port,
        mode=mode,
        device=device,
        node_addresses=node_addresses
    )
    return manager
