# coding=utf-8
"""
Ring通信管理器

采用环形传播策略：消息沿环形拓扑传播，同时聚合多个消息一起发送

日志记录说明：
- DEBUG: 详细的发送/接收数据信息（tensor形状、大小）
- INFO: 重要发送/接收事件、初始化状态
- WARNING: 发送失败、转发异常
- ERROR: 严重错误

连接拓扑（环形）：
- 每个rank只连接到下一个rank（发送）
- 每个rank只从上一个rank接收
- rank N-1 连接到 rank 0 形成环

消息传播方式：
1. 发送给某个rank的消息，如果目标不是下一跳，则先发到下一跳再转发
2. 广播消息沿环传播，每个节点接收后继续转发
3. 多个消息可以聚合成一个包一起发送
"""

import zmq
import torch
import time
from typing import Dict, List, Tuple, Optional, Any

from .base_comm_manager import ZMQCommManagerBase
from ..logging_utils import get_tensor_info
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
    
    Attributes:
        send_sockets: 发送socket（只有到下一个rank的连接）
        recv_sockets: 接收socket（只有从上一个rank的连接）
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
        """
        设置Ring模式的socket连接
        
        Ring模式下每个rank只需要：
        - 1个接收socket：从上一个rank接收
        - 1个发送socket：向下一个rank发送
        """
        prev_rank = self._get_prev_rank()
        next_rank = self._get_next_rank()
        
        self.logger.debug(f"[INIT] 初始化 Ring 模式 socket 连接...")
        
        if self.world_size == 1:
            self.logger.debug("[INIT] 单节点模式，无需设置Ring连接")
            return
        
        # 接收socket：从上一个rank接收
        port = self._get_port(prev_rank, self.rank)
        recv_socket = self.context.socket(zmq.PULL)
        recv_socket.setsockopt(zmq.RCVHWM, 1000)
        recv_socket.bind(f"tcp://*:{port}")
        self.recv_sockets[prev_rank] = recv_socket
        self.logger.debug(f"[BIND] 绑定接收端口 {port} (接收来自 rank {prev_rank} 的消息)")
        
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
        self.logger.debug(f"[CONNECT] 连接到 rank {next_rank} ({addr}:{port})")
        
        self.logger.info(f"[INIT_OK] Ring 模式初始化完成: prev={prev_rank}, next={next_rank}")
    
    def _send_worker(self):
        """
        Ring模式发送线程：聚合消息并发送到下一跳
        
        特点：
        1. 阻塞等待第一条消息
        2. 非阻塞收集更多消息进行聚合
        3. 批量序列化发送，减少网络开销
        """
        self.logger.debug("[THREAD] 发送线程启动 (Ring模式)")
        next_rank = self._get_next_rank()
        
        if self.world_size == 1:
            self.logger.debug("[THREAD] 单节点模式，发送线程退出")
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
                    self.logger.debug(f"[AGGREGATE] 聚合发送 {len(messages)} 条消息")
                
                # 批量序列化并发送
                send_start = time.time()
                data = MessageSerializer.serialize_batch(messages)
                self.send_sockets[next_rank].send(data)
                send_time = time.time() - send_start
                
                # 记录发送日志和更新统计
                for msg in messages:
                    self._update_send_stats(msg.msg_type)
                    self._log_send_event(msg, len(data) // len(messages), send_time / len(messages))
                    
            except Exception as e:
                self.logger.error(f"[THREAD_ERR] 发送线程出错: {e}", exc_info=True)
        
        self.logger.debug("[THREAD] 发送线程退出")
    
    def _log_send_event(self, msg: Message, data_size: int, send_time: float):
        """
        记录发送事件的详细日志
        
        Args:
            msg: 发送的消息
            data_size: 数据大小（字节）
            send_time: 发送耗时（秒）
        """
        size_kb = data_size / 1024
        msg_type_name = MessageType(msg.msg_type).name
        next_rank = self._get_next_rank()
        
        # 确定最终目标
        dst_info = "broadcast" if msg.dst_rank == -1 else f"rank {msg.dst_rank}"
        
        if msg.msg_type == MessageType.DRAFT_TOKENS:
            data = msg.data
            if isinstance(data, dict):
                draft_shape = tuple(data.get('draft_tokens', torch.empty(0)).shape)
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> {dst_info} (via rank {next_rank}) | "
                    f"seq_id={msg.seq_id}, draft_shape={draft_shape}, "
                    f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
                )
            else:
                self.logger.debug(f"[SEND] {msg_type_name} -> {dst_info} | seq_id={msg.seq_id}")
                
        elif msg.msg_type == MessageType.HIDDEN:
            hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
            self.logger.debug(
                f"[SEND] {msg_type_name} -> {dst_info} (via rank {next_rank}) | "
                f"seq_id={msg.seq_id}, chunk_idx={msg.chunk_idx}, "
                f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
            )
            
        elif msg.msg_type == MessageType.EAGLE_INPUT_HIDDEN:
            self.logger.debug(
                f"[SEND] {msg_type_name} -> {dst_info} (via rank {next_rank}) | "
                f"seq_id={msg.seq_id}, layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, "
                f"size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
            )
            
        elif msg.msg_type == MessageType.BASE_CACHE:
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_shape = tuple(msg.data[0].shape)
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> {dst_info} (via rank {next_rank}) | "
                    f"seq_id={msg.seq_id}, layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, "
                    f"key_shape={key_shape}, size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
                )
                
        elif msg.msg_type == MessageType.EAGLE_CACHE:
            is_incremental = (msg.layer_idx != -1)
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_shape = tuple(msg.data[0].shape)
                self.logger.debug(
                    f"[SEND] {msg_type_name} -> {dst_info} (via rank {next_rank}) | "
                    f"seq_id={msg.seq_id}, chunk_idx={msg.chunk_idx}, incremental={is_incremental}, "
                    f"key_shape={key_shape}, size={size_kb:.2f}KB, time={send_time*1000:.2f}ms"
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
    
    def _process_received_data(self, data: bytes):
        """
        Ring模式：处理批量消息并决定是否转发
        
        Ring模式下，消息可能需要转发给其他rank：
        1. 如果消息目标不是本rank，转发给下一跳
        2. 如果是广播消息且还没转完一圈，继续转发
        """
        messages = MessageSerializer.deserialize_batch(data, self.device)
        
        for msg in messages:
            if msg.original_src < 0:
                msg.original_src = msg.src_rank
            
            # 记录接收日志
            self._log_recv_event(msg)
            
            # 检查是否需要转发
            if msg.dst_rank != self.rank and msg.dst_rank != -1:
                # 需要转发给其他rank
                msg.src_rank = self.rank  # 更新发送者
                self.send_queue.put(msg)
                self.stats['forwarded_messages'] += 1
                self.logger.debug(f"[FORWARD] {MessageType(msg.msg_type).name} -> rank {msg.dst_rank}")
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
                    self.logger.debug(f"[FORWARD] 广播 {MessageType(msg.msg_type).name} (orig_src={original_src})")
            
            # 处理这条消息（放入接收队列）
            if msg.dst_rank == self.rank or msg.dst_rank == -1:
                self._handle_received_message(msg)
    
    def _log_recv_event(self, msg: Message):
        """
        记录接收事件的日志
        
        Args:
            msg: 接收到的消息
        """
        msg_type_name = MessageType(msg.msg_type).name
        src = msg.get_effective_src()
        latency = time.time() - msg.timestamp if msg.timestamp > 0 else 0
        
        if msg.msg_type == MessageType.DRAFT_TOKENS:
            if isinstance(msg.data, dict):
                draft_shape = tuple(msg.data.get('draft_tokens', torch.empty(0)).shape)
                self.logger.debug(
                    f"[RECV] {msg_type_name} from rank {src} | "
                    f"seq_id={msg.seq_id}, draft_shape={draft_shape}, latency={latency*1000:.2f}ms"
                )
        elif msg.msg_type == MessageType.HIDDEN:
            self.logger.debug(
                f"[RECV] {msg_type_name} from rank {src} | "
                f"chunk_idx={msg.chunk_idx}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
        elif msg.msg_type == MessageType.EAGLE_INPUT_HIDDEN:
            self.logger.debug(
                f"[RECV] {msg_type_name} from rank {src} | "
                f"layer_idx={msg.layer_idx}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
        elif msg.msg_type == MessageType.BASE_CACHE:
            self.logger.debug(
                f"[RECV] {msg_type_name} from rank {src} | "
                f"layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, latency={latency*1000:.2f}ms"
            )
        elif msg.msg_type == MessageType.EAGLE_CACHE:
            is_incremental = (msg.layer_idx != -1)
            self.logger.debug(
                f"[RECV] {msg_type_name} from rank {src} | "
                f"chunk_idx={msg.chunk_idx}, incremental={is_incremental}, latency={latency*1000:.2f}ms"
            )
    
    # ==================== Draft Tokens 实现 ====================
    
    def send_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        dst_rank: int = -1
    ) -> None:
        """
        Ring模式：发送draft tokens，由转发机制广播
        
        Args:
            draft_tokens: 候选tokens
            retrieve_indices: 检索索引
            tree_mask: 树掩码
            tree_position_ids: 位置编码
            dst_rank: 目标rank，-1表示广播
        """
        packed_data = {
            'draft_tokens': draft_tokens.cpu(),
            'retrieve_indices': retrieve_indices.cpu(),
            'tree_mask': tree_mask.cpu(),
            'tree_position_ids': tree_position_ids.cpu(),
        }
        
        target_info = "broadcast" if dst_rank == -1 else f"rank {dst_rank}"
        self.logger.info(
            f"[BROADCAST] DRAFT_TOKENS -> {target_info} | "
            f"draft_shape={tuple(draft_tokens.shape)}"
        )
        self.logger.debug(
            f"  retrieve_shape={tuple(retrieve_indices.shape)}, "
            f"mask_shape={tuple(tree_mask.shape)}"
        )
        
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
        """
        Ring模式：发送hidden，由转发机制送达目标
        
        Args:
            hidden: hidden state tensor
            dst_rank: 目标rank
            chunk_idx: chunk索引
        """
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
        """
        Ring模式：发送eagle input hidden，由转发机制送达最后一个rank
        
        Args:
            hidden: hidden state tensor
            layer_idx: 层索引
            dst_rank: 目标rank（通常是最后一个rank）
            chunk_idx: chunk索引
        """
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
        """
        Ring模式：发送base cache，由转发机制送达目标
        
        Args:
            kv_cache: (key, value) 元组
            dst_rank: 目标rank
            layer_idx: 层索引
            chunk_idx: chunk索引
        """
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
        """
        Ring模式：发送eagle cache
        
        Args:
            kv_cache: (key, value) 元组
            dst_rank: 目标rank
            is_incremental: 是否为增量更新
            chunk_idx: chunk索引
        """
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
            layer_idx=1 if is_incremental else -1,
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
        Ring模式：广播Eagle Layer的增量draft_past_key_values给所有其他rank
        
        通过环形传播，消息会自动转发给所有其他rank
        
        Args:
            incremental_kv: 增量KV cache (new_key, new_value)
            chunk_idx: chunk索引
        """
        key_shape = tuple(incremental_kv[0].shape)
        self.logger.debug(
            f"[BROADCAST] EAGLE_STABLE_KV -> all ranks | "
            f"chunk_idx={chunk_idx}, key_shape={key_shape}"
        )
        
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
    # ==================== 分支调度通信 (最高优先级) ====================
    
    def send_schedule_plan_async(
        self,
        schedule_plan_data: Dict[str, Any],
        dst_rank: int = -1,
    ) -> None:
        """
        Ring模式：异步广播调度计划
        
        Args:
            schedule_plan_data: 序列化的调度计划数据
            dst_rank: 目标rank，-1表示广播
        """
        self.logger.debug(
            f"[BROADCAST] SCHEDULE_PLAN -> dst_rank={dst_rank}"
        )
        
        msg = Message(
            msg_type=MessageType.SCHEDULE_PLAN,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=schedule_plan_data,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank if dst_rank == -1 else -1,
        )
        self.send_queue.put(msg)
    
    def send_branch_prompt_async(
        self,
        branch_data: Dict[str, Any],
        dst_rank: int,
        branch_id: int = -1,
    ) -> None:
        """
        Ring模式：异步发送分支Prompt
        
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
        Ring模式：异步发送分支输出（完成后立即发送）
        
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
        Ring模式：异步广播完成信号
        
        Args:
            dst_rank: 目标rank，-1表示广播
        """
        self.logger.debug(
            f"[BROADCAST] BRANCH_COMPLETE -> dst_rank={dst_rank}"
        )
        
        msg = Message(
            msg_type=MessageType.BRANCH_COMPLETE,
            src_rank=self.rank,
            dst_rank=dst_rank,
            data=True,
            seq_id=self._get_next_seq_id(),
            original_src=self.rank if dst_rank == -1 else -1,
        )
        self.send_queue.put(msg)