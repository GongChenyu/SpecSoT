# coding=utf-8
"""
Ring通信管理器

采用环形传播策略：消息沿环形拓扑传播，同时聚合多个消息一起发送
"""

import zmq
import torch
import time
from typing import Dict, List, Tuple, Optional

from .base_comm_manager import ZMQCommManagerBase
from .comm_utils import (
    Message,
    MessageType,
    MessageSerializer,
    AggregatedMessage,
)


class RingCommManager(ZMQCommManagerBase):
    """
    环形通信管理器
    
    采用环形传播策略：消息沿环形拓扑传播，同时聚合多个消息一起发送
    优点：减少总通信量，适合需要全局同步的场景
    缺点：延迟略高（需要多跳）
    
    连接拓扑（环形）：
    - 每个rank只连接到下一个rank（发送）
    - 每个rank只从上一个rank接收
    - rank N-1 连接到 rank 0 形成环
    
    消息传播方式：
    1. 发送给某个rank的消息，如果目标不是下一跳，则先发到下一跳再转发
    2. 广播消息沿环传播，每个节点接收后继续转发
    3. 多个消息可以聚合成一个包一起发送
    """
    
    def __init__(self, *args, **kwargs):
        # Ring模式需要计算下一跳和上一跳
        super().__init__(*args, **kwargs)
    
    def _get_next_rank(self) -> int:
        """获取环中的下一个rank"""
        return (self.rank + 1) % self.world_size
    
    def _get_prev_rank(self) -> int:
        """获取环中的上一个rank"""
        return (self.rank - 1 + self.world_size) % self.world_size
    
    def _setup_sockets(self):
        """设置Ring模式的socket连接"""
        prev_rank = self._get_prev_rank()
        next_rank = self._get_next_rank()
        
        if self.world_size == 1:
            self.logger.info("单节点模式，无需设置Ring连接")
            return
        
        # 接收socket：从上一个rank接收
        port = self._get_port(prev_rank, self.rank)
        recv_socket = self.context.socket(zmq.PULL)
        recv_socket.setsockopt(zmq.RCVHWM, 1000)
        recv_socket.bind(f"tcp://*:{port}")
        self.recv_sockets[prev_rank] = recv_socket
        self.logger.debug(f"Ring: 绑定接收端口 {port} (from rank {prev_rank})")
        
        # 等待所有节点设置好接收端口
        time.sleep(1)
        
        # 发送socket：到下一个rank
        port = self._get_port(self.rank, next_rank)
        addr = self.node_addresses.get(next_rank, "127.0.0.1")
        send_socket = self.context.socket(zmq.PUSH)
        send_socket.setsockopt(zmq.SNDHWM, 1000)
        send_socket.setsockopt(zmq.LINGER, 0)
        send_socket.connect(f"tcp://{addr}:{port}")
        self.send_sockets[next_rank] = send_socket
        self.logger.debug(f"Ring: 连接到 rank {next_rank} 端口 {port}")
        
        self.logger.info(f"Ring模式初始化完成: prev={prev_rank}, next={next_rank}")
    
    def _send_worker(self):
        """Ring模式发送线程：聚合消息并发送到下一跳"""
        self.logger.info("发送线程启动 (Ring模式)")
        next_rank = self._get_next_rank()
        
        if self.world_size == 1:
            self.logger.info("单节点模式，发送线程退出")
            return
        
        while self.is_running:
            try:
                # 收集所有待发送消息
                messages = []
                
                # 获取一条消息（阻塞等待）
                first_msg = self.send_queue.get_by_priority(timeout=0.1)
                if first_msg:
                    messages.append(first_msg)
                    
                    # 非阻塞收集更多消息（聚合）
                    for _ in range(100):  # 最多再收集100条
                        msg = self.send_queue.get_by_priority(timeout=0.001)
                        if msg:
                            messages.append(msg)
                        else:
                            break
                
                if not messages:
                    continue
                
                # 按优先级排序
                messages.sort(key=lambda m: (m.priority, m.timestamp))
                
                if len(messages) > 1:
                    self.stats['aggregated_sends'] += 1
                    self.logger.debug(f"Ring: 聚合发送 {len(messages)} 条消息")
                
                # 批量序列化并发送
                data = MessageSerializer.serialize_batch(messages)
                self.send_sockets[next_rank].send(data)
                
                # 更新统计
                for msg in messages:
                    self._update_send_stats(msg.msg_type)
                    
            except Exception as e:
                self.logger.error(f"发送线程出错: {e}", exc_info=True)
        
        self.logger.info("发送线程退出")
    
    def _update_send_stats(self, msg_type: MessageType):
        """更新发送统计"""
        if msg_type == MessageType.DRAFT_TOKENS:
            self.stats['draft_tokens_sent'] += 1
        elif msg_type == MessageType.HIDDEN:
            self.stats['hidden_sent'] += 1
        elif msg_type == MessageType.EAGLE_INPUT_HIDDEN:
            self.stats['eagle_input_hidden_sent'] += 1
        elif msg_type == MessageType.BASE_CACHE:
            self.stats['base_cache_sent'] += 1
        elif msg_type == MessageType.EAGLE_CACHE:
            self.stats['eagle_cache_sent'] += 1
    
    def _process_received_data(self, data: bytes):
        """Ring模式：处理批量消息并决定是否转发"""
        messages = MessageSerializer.deserialize_batch(data, self.device)
        
        for msg in messages:
            if msg.original_src < 0:
                msg.original_src = msg.src_rank
            
            # 检查是否需要转发
            if msg.dst_rank != self.rank and msg.dst_rank != -1:
                # 需要转发给其他rank
                msg.src_rank = self.rank  # 更新发送者
                self.send_queue.put(msg)
                self.stats['forwarded_messages'] += 1
            elif msg.dst_rank == -1:
                # 广播消息：如果还没转完一圈，继续转发
                original_src = msg.get_effective_src()
                if original_src != self._get_next_rank():
                    # 还没回到起点的下一个节点，继续转发
                    forward_msg = Message(
                        msg_type=msg.msg_type,
                        src_rank=self.rank,
                        dst_rank=-1,
                        data=msg.data,
                        chunk_idx=msg.chunk_idx,
                        layer_idx=msg.layer_idx,
                        seq_id=msg.seq_id,
                        timestamp=msg.timestamp,
                        original_src=original_src,
                    )
                    self.send_queue.put(forward_msg)
                    self.stats['forwarded_messages'] += 1
            
            # 处理这条消息（放入接收队列）
            if msg.dst_rank == self.rank or msg.dst_rank == -1:
                self._handle_received_message(msg)
    
    # ==================== Draft Tokens 实现 ====================
    
    def send_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        dst_rank: int = -1
    ) -> None:
        """Ring模式：发送draft tokens，由转发机制广播"""
        packed_data = {
            'draft_tokens': draft_tokens.cpu(),
            'retrieve_indices': retrieve_indices.cpu(),
            'tree_mask': tree_mask.cpu(),
            'tree_position_ids': tree_position_ids.cpu(),
        }
        
        if dst_rank == -1:
            # 广播：发送到下一跳，标记dst=-1
            msg = Message(
                msg_type=MessageType.DRAFT_TOKENS,
                src_rank=self.rank,
                dst_rank=-1,  # 广播标记
                data=packed_data,
                seq_id=self._get_next_seq_id(),
                original_src=self.rank,
            )
            self.send_queue.put(msg)
        else:
            # 单播：发送到下一跳，由转发机制送达
            msg = Message(
                msg_type=MessageType.DRAFT_TOKENS,
                src_rank=self.rank,
                dst_rank=dst_rank,
                data=packed_data,
                seq_id=self._get_next_seq_id(),
            )
            self.send_queue.put(msg)
    
    # ==================== Hidden State 实现 ====================
    
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """Ring模式：发送hidden，由转发机制送达目标"""
        msg = Message(
            msg_type=MessageType.HIDDEN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=hidden,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    # ==================== Eagle Input Hidden 实现 ====================
    
    def send_eagle_input_hidden(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        dst_rank: int
    ) -> None:
        """Ring模式：发送eagle input hidden，由转发机制送达最后一个rank"""
        msg = Message(
            msg_type=MessageType.EAGLE_INPUT_HIDDEN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=hidden,
            layer_idx=layer_idx,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    # ==================== Base Cache 实现 ====================
    
    def send_base_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """Ring模式：发送base cache，由转发机制送达目标"""
        msg = Message(
            msg_type=MessageType.BASE_CACHE,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=kv_cache,
            layer_idx=layer_idx,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    # ==================== Eagle Cache 实现 ====================
    
    def send_eagle_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        is_incremental: bool = False,
        chunk_idx: int = -1
    ) -> None:
        """Ring模式：发送eagle cache"""
        msg = Message(
            msg_type=MessageType.EAGLE_CACHE,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=kv_cache,
            layer_idx=1 if is_incremental else -1,
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    # ==================== Eagle Stable KV 广播 ====================
    
    def broadcast_eagle_stable_kv(
        self,
        incremental_kv: Tuple[torch.Tensor, torch.Tensor],
        chunk_idx: int,
    ) -> None:
        """
        Ring模式：广播Eagle Layer的增量stable_kv给所有其他rank
        
        通过环形传播，消息会自动转发给所有其他rank
        
        Args:
            incremental_kv: 增量KV cache (new_key, new_value)
            chunk_idx: chunk索引
        """
        self.logger.info(f"广播eagle stable_kv: chunk_idx={chunk_idx}, key shape={incremental_kv[0].shape}")
        
        # Ring模式下，使用广播标记 dst=-1，消息会自动沿环传播
        msg = Message(
            msg_type=MessageType.EAGLE_CACHE,
            src_rank=self.rank,
            dst_rank=-1,  # 广播标记
            data=incremental_kv,
            layer_idx=1,  # 标记为增量
            chunk_idx=chunk_idx,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank,
        )
        self.send_queue.put(msg)
