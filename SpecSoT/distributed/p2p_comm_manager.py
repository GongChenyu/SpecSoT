# coding=utf-8
"""
P2P通信管理器

采用直接发送策略：每个rank直接向目标rank发送消息
"""

import zmq
import torch
import time
from typing import Dict, Tuple, Optional

from .base_comm_manager import ZMQCommManagerBase
from .comm_utils import (
    Message,
    MessageType,
    MessageSerializer,
)


class P2PCommManager(ZMQCommManagerBase):
    """
    点对点通信管理器
    
    采用直接发送策略：每个rank直接向目标rank发送消息
    优点：延迟低，直接送达
    缺点：需要建立更多连接，广播时发送次数较多
    
    连接拓扑：
    - 每个rank建立到所有其他rank的PUSH socket（发送）
    - 每个rank建立从所有其他rank的PULL socket（接收）
    """
    
    def _setup_sockets(self):
        """设置P2P模式的socket连接"""
        # 先设置接收socket（绑定端口）
        for other_rank in range(self.world_size):
            if other_rank != self.rank:
                port = self._get_port(other_rank, self.rank)
                socket = self.context.socket(zmq.PULL)
                socket.setsockopt(zmq.RCVHWM, 1000)
                socket.bind(f"tcp://*:{port}")
                self.recv_sockets[other_rank] = socket
                self.logger.debug(f"绑定接收端口 {port} (from rank {other_rank})")
        
        # 等待所有节点设置好接收端口
        time.sleep(1)
        
        # 再设置发送socket（连接到对端）
        for other_rank in range(self.world_size):
            if other_rank != self.rank:
                port = self._get_port(self.rank, other_rank)
                addr = self.node_addresses.get(other_rank, "127.0.0.1")
                socket = self.context.socket(zmq.PUSH)
                socket.setsockopt(zmq.SNDHWM, 1000)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(f"tcp://{addr}:{port}")
                self.send_sockets[other_rank] = socket
                self.logger.debug(f"连接到 rank {other_rank} 端口 {port}")
        
        self.logger.info(f"P2P模式初始化完成: {len(self.send_sockets)} send, {len(self.recv_sockets)} recv")
    
    def _send_worker(self):
        """P2P模式发送线程：按优先级获取消息并直接发送"""
        self.logger.info("发送线程启动 (P2P模式)")
        
        while self.is_running:
            try:
                msg = self.send_queue.get_by_priority(timeout=0.1)
                if msg is None:
                    continue
                
                # 直接发送到目标
                dst = msg.dst_rank
                if dst in self.send_sockets:
                    data = MessageSerializer.serialize(msg)
                    self.send_sockets[dst].send(data)
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
    
    # ==================== Draft Tokens 实现 ====================
    
    def send_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        dst_rank: int = -1
    ) -> None:
        """P2P模式：直接发送到所有其他rank"""
        packed_data = {
            'draft_tokens': draft_tokens.cpu(),
            'retrieve_indices': retrieve_indices.cpu(),
            'tree_mask': tree_mask.cpu(),
            'tree_position_ids': tree_position_ids.cpu(),
        }
        
        targets = [r for r in range(self.world_size) if r != self.rank] if dst_rank == -1 else [dst_rank]
        
        for target in targets:
            msg = Message(
                msg_type=MessageType.DRAFT_TOKENS,
                src_rank=self.rank,
                dst_rank=target,
                data=packed_data,
                seq_id=self._get_next_seq_id(),
            )
            self.send_queue.put(msg)
    
    # ==================== Hidden State 实现 ====================
    
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """P2P模式：直接发送到目标rank"""
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
        """P2P模式：直接发送到最后一个rank"""
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
        """P2P模式：直接发送cache到目标rank"""
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
        """P2P模式：直接发送eagle cache到目标rank"""
        msg = Message(
            msg_type=MessageType.EAGLE_CACHE,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=kv_cache,
            layer_idx=1 if is_incremental else -1,  # 用layer_idx区分是否增量
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
        P2P模式：广播Eagle Layer的增量stable_kv给所有其他rank
        
        直接向所有其他rank发送增量KV cache
        
        Args:
            incremental_kv: 增量KV cache (new_key, new_value)
            chunk_idx: chunk索引
        """
        self.logger.info(f"广播eagle stable_kv: chunk_idx={chunk_idx}, key shape={incremental_kv[0].shape}")
        
        # 向所有其他rank发送
        for dst_rank in range(self.world_size):
            if dst_rank != self.rank:
                self.send_eagle_cache_async(
                    kv_cache=incremental_kv,
                    dst_rank=dst_rank,
                    is_incremental=True,
                    chunk_idx=chunk_idx,
                )
