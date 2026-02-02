# coding=utf-8
"""
P2P通信管理器

采用直接发送策略：每个rank直接向目标rank发送消息

日志记录说明：
- DEBUG: 详细的发送数据信息（tensor形状、大小）
- INFO: 重要发送事件（初始化、广播）
- WARNING: 发送失败或异常

连接拓扑：
- 每个rank建立到所有其他rank的PUSH socket（发送）
- 每个rank建立从所有其他rank的PULL socket（接收）
"""

import zmq
import torch
import time
from typing import Dict, Tuple, Optional, List, Any

from .base_comm_manager import ZMQCommManagerBase
from ...utils.logging import get_tensor_info
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
        self.logger.debug(f"[INIT] 初始化 P2P 模式 socket 连接...")
        
        # 先设置接收socket（绑定端口）
        for other_rank in range(self.world_size):
            if other_rank != self.rank:
                port = self._get_port(other_rank, self.rank)
                socket = self.context.socket(zmq.PULL)
                socket.setsockopt(zmq.RCVHWM, 1000)
                socket.bind(f"tcp://*:{port}")
                self.recv_sockets[other_rank] = socket
                self.logger.debug(f"[BIND] 绑定接收端口 {port} (接收来自 rank {other_rank} 的消息)")
        
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
                self.logger.debug(f"[CONNECT] 连接到 rank {other_rank} ({addr}:{port})")
        
        self.logger.info(f"[INIT_OK] P2P 模式初始化完成: {len(self.send_sockets)} 发送连接, {len(self.recv_sockets)} 接收连接")
    
    def _send_worker(self):
        """P2P模式发送线程：按优先级获取消息并直接发送"""
        self.logger.debug("[THREAD] 发送线程启动 (P2P模式)")
        
        while self.is_running:
            try:
                msg = self.send_queue.get_by_priority(timeout=0.1)
                if msg is None:
                    continue
                
                # 直接发送到目标
                dst = msg.dst_rank
                if dst in self.send_sockets:
                    send_start = time.time()
                    data = MessageSerializer.serialize(msg)
                    self.send_sockets[dst].send(data)
                    send_time = time.time() - send_start
                    self._update_send_stats(msg.msg_type)
                    
                    # 记录发送日志
                    self._log_send_event(msg, len(data), send_time)
                else:
                    self.logger.warning(f"[SEND_ERR] 目标 rank {dst} 不存在于发送 socket 列表中")
                    
            except Exception as e:
                self.logger.error(f"[THREAD_ERR] 发送线程出错: {e}", exc_info=True)
        
        self.logger.debug("[THREAD] 发送线程退出")
    
    def _log_send_event(self, msg: Message, data_size: int, send_time: float):
        """
        记录发送事件的详细日志
        
        注意：所有通信细节使用DEBUG级别，只有关键事件使用INFO
        """
        size_kb = data_size / 1024
        msg_type_name = MessageType(msg.msg_type).name
        
        if msg.msg_type == MessageType.DRAFT_TOKENS:
            data = msg.data
            if isinstance(data, dict):
                draft_shape = tuple(data.get('draft_tokens', torch.empty(0)).shape)
                # DRAFT_TOKENS是关键事件，使用INFO
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                    f"seq_id={msg.seq_id}, draft_shape={draft_shape}, "
                    f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
                )
            else:
                self.logger.debug(f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | seq_id={msg.seq_id}")
                
        elif msg.msg_type == MessageType.HIDDEN:
            hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
            self.logger.debug(
                f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                f"seq_id={msg.seq_id}, chunk_idx={msg.chunk_idx}, "
                f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
            )
            self.logger.debug(f"  hidden: {hidden_info}")
            
        elif msg.msg_type == MessageType.EAGLE_INPUT_HIDDEN:
            hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
            self.logger.debug(
                f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                f"seq_id={msg.seq_id}, layer_idx={msg.layer_idx}, "
                f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
            )
            self.logger.debug(f"  hidden: {hidden_info}")
            
        elif msg.msg_type == MessageType.BASE_CACHE:
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_shape = tuple(msg.data[0].shape)
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                    f"seq_id={msg.seq_id}, layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, "
                    f"key_shape={key_shape}, size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
                )
            else:
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                    f"layer={msg.layer_idx}, chunk={msg.chunk_idx}"
                )
                
        elif msg.msg_type == MessageType.EAGLE_CACHE:
            is_incremental = (msg.layer_idx != -1)
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_shape = tuple(msg.data[0].shape)
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                    f"seq_id={msg.seq_id}, chunk_idx={msg.chunk_idx}, incremental={is_incremental}, "
                    f"key_shape={key_shape}, size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
                )
            else:
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> rank {msg.dst_rank} | "
                    f"chunk={msg.chunk_idx}, incremental={is_incremental}"
                )

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
        # 分支调度相关
        elif msg_type == MessageType.SCHEDULE_PLAN:
            self.stats['schedule_plan_sent'] += 1
        elif msg_type == MessageType.BRANCH_PROMPT:
            self.stats['branch_prompt_sent'] += 1
        elif msg_type == MessageType.BRANCH_OUTPUT:
            self.stats['branch_output_sent'] += 1
        elif msg_type == MessageType.BRANCH_COMPLETE:
            self.stats['branch_complete_sent'] += 1
    
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
        
        # DRAFT_TOKENS 广播是关键事件，使用INFO
        self.logger.info(
            f"[BROADCAST] DRAFT_TOKENS -> targets={targets} | "
            f"draft_shape={tuple(draft_tokens.shape)}"
        )
        self.logger.debug(
            f"  retrieve_shape={tuple(retrieve_indices.shape)}, "
            f"mask_shape={tuple(tree_mask.shape)}"
        )
        
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
        hidden_info = get_tensor_info(hidden) if torch.is_tensor(hidden) else str(type(hidden))
        self.logger.debug(
            f"[QUEUE] HIDDEN -> rank {dst_rank} | "
            f"chunk_idx={chunk_idx}"
        )
        self.logger.debug(f"  hidden: {hidden_info}")
        
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
        dst_rank: int,
        chunk_idx: int = -1
    ) -> None:
        """P2P模式：直接发送到最后一个rank"""
        hidden_info = get_tensor_info(hidden) if torch.is_tensor(hidden) else str(type(hidden))
        self.logger.debug(
            f"[QUEUE] EAGLE_INPUT_HIDDEN -> rank {dst_rank} | "
            f"layer_idx={layer_idx}, chunk_idx={chunk_idx}"
        )
        self.logger.debug(f"  hidden: {hidden_info}")
        
        msg = Message(
            msg_type=MessageType.EAGLE_INPUT_HIDDEN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=hidden,
            layer_idx=layer_idx,
            chunk_idx=chunk_idx,
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
        key_shape = tuple(kv_cache[0].shape) if kv_cache and len(kv_cache) >= 1 else "N/A"
        self.logger.debug(
            f"[QUEUE] BASE_CACHE -> rank {dst_rank} | "
            f"layer_idx={layer_idx}, chunk_idx={chunk_idx}, key_shape={key_shape}"
        )
        
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
        key_shape = tuple(kv_cache[0].shape) if kv_cache and len(kv_cache) >= 1 else "N/A"
        self.logger.debug(
            f"[QUEUE] EAGLE_CACHE -> rank {dst_rank} | "
            f"chunk_idx={chunk_idx}, incremental={is_incremental}, key_shape={key_shape}"
        )
        
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
    
    def broadcast_eagle_draft_past_key_values(
        self,
        incremental_kv: Tuple[torch.Tensor, torch.Tensor],
        chunk_idx: int,
    ) -> None:
        """
        P2P模式：广播Eagle Layer的增量draft_past_key_values给所有其他rank
        
        直接向所有其他rank发送增量KV cache
        
        Args:
            incremental_kv: 增量KV cache (new_key, new_value)
            chunk_idx: chunk索引
        """
        key_shape = tuple(incremental_kv[0].shape)
        targets = [r for r in range(self.world_size) if r != self.rank]
        
        self.logger.debug(
            f"[BROADCAST] EAGLE_STABLE_KV -> targets={targets} | "
            f"chunk_idx={chunk_idx}, key_shape={key_shape}"
        )
        
        # 向所有其他rank发送
        for dst_rank in targets:
            self.send_eagle_cache_async(
                kv_cache=incremental_kv,
                dst_rank=dst_rank,
                is_incremental=True,
                chunk_idx=chunk_idx,
            )    
    # ==================== 分支调度通信 (最高优先级) ====================
    
    def send_schedule_plan_async(
        self,
        schedule_plan_data: Dict[str, Any],
        dst_rank: int = -1,
    ) -> None:
        """
        P2P模式：异步广播调度计划
        
        Args:
            schedule_plan_data: 序列化的调度计划数据
            dst_rank: 目标rank，-1表示广播
        """
        targets = [r for r in range(self.world_size) if r != self.rank] if dst_rank == -1 else [dst_rank]
        
        self.logger.debug(
            f"[BROADCAST] SCHEDULE_PLAN -> targets={targets}"
        )
        
        for target in targets:
            msg = Message(
                msg_type=MessageType.SCHEDULE_PLAN,
                src_rank=self.rank,
                dst_rank=target,
                data=schedule_plan_data,
                seq_id=self._get_next_seq_id(),
            )
            self.send_queue.put(msg)
    
    def send_branch_prompt_async(
        self,
        branch_data: Dict[str, Any],
        dst_rank: int,
        branch_id: int = -1,
    ) -> None:
        """
        P2P模式：异步发送分支Prompt
        
        Args:
            branch_data: 分支数据
            dst_rank: 目标rank
            branch_id: 分支ID
        """
        self.logger.debug(
            f"[SEND] BRANCH_PROMPT -> rank {dst_rank} | branch_id={branch_id}"
        )
        
        msg = Message(
            msg_type=MessageType.BRANCH_PROMPT,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=branch_data,
            branch_id=branch_id,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    def send_branch_output_async(
        self,
        branch_id: int,
        output_tokens: List[int],
        dst_rank: int = 0,
    ) -> None:
        """
        P2P模式：异步发送分支输出（完成后立即发送）
        
        Args:
            branch_id: 分支ID
            output_tokens: 生成的token列表
            dst_rank: 目标rank
        """
        self.logger.debug(
            f"[SEND] BRANCH_OUTPUT -> rank {dst_rank} | "
            f"branch_id={branch_id}, num_tokens={len(output_tokens)}"
        )
        
        msg = Message(
            msg_type=MessageType.BRANCH_OUTPUT,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=output_tokens,
            branch_id=branch_id,
            seq_id=self._get_next_seq_id(),
        )
        self.send_queue.put(msg)
    
    def send_branch_complete_async(
        self,
        dst_rank: int = -1,
    ) -> None:
        """
        P2P模式：异步广播完成信号
        
        Args:
            dst_rank: 目标rank，-1表示广播
        """
        targets = [r for r in range(self.world_size) if r != self.rank] if dst_rank == -1 else [dst_rank]
        
        self.logger.debug(
            f"[BROADCAST] BRANCH_COMPLETE -> targets={targets}"
        )
        
        for target in targets:
            msg = Message(
                msg_type=MessageType.BRANCH_COMPLETE,
                src_rank=self.rank,
                dst_rank=target,
                data=True,  # 简单的完成标志
                seq_id=self._get_next_seq_id(),
            )
            self.send_queue.put(msg)