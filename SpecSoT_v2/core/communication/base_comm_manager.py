# coding=utf-8
"""
ZMQ通信管理器基类

定义通用接口和公共功能，具体的发送策略由子类实现

日志记录说明：
- DEBUG: 详细的发送/接收数据信息（tensor形状、大小、类型）- 只写文件
- INFO: 重要事件（启动、停止、关键通信点）- 控制台+文件
- WARNING: 异常情况（超时、队列满）
- ERROR: 错误信息

注意：
- 通信日志控制台只显示WARNING及以上级别，减少刷屏
- 文件记录所有级别，便于调试
- 使用带自动刷新的handler确保日志实时输出
"""

import zmq
import torch
import threading
import queue
import time
import pickle
import io
import logging
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import IntEnum

from .comm_utils import Message, MessageType, MessagePriority, MessageSerializer, TensorSerializer  
from .comm_utils import ThreadSafeQueue, AggregatedMessage

# 使用统一的日志模块
from ...utils.logging import (
    FlushHandler, 
    FlushFileHandler, 
    get_tensor_info,
    get_unified_logger,
)


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
        
        # 设置logger - 使用统一的日志模块
        # 通信日志使用单独的logger，控制台只显示WARNING以上
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_dir, 'logs')
        self.logger = get_unified_logger(
            rank=rank, 
            log_dir=log_dir,
            console_level=logging.WARNING,  # 通信日志控制台只显示警告以上
            file_level=logging.DEBUG,
            name_suffix="-Comm"
        )
        
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
            'draft_tokens_sent': 0,
            'draft_tokens_recv': 0,
            'hidden_sent': 0,
            'hidden_recv': 0,
            'eagle_input_hidden_sent': 0,
            'eagle_input_hidden_recv': 0,
            'base_cache_sent': 0,
            'base_cache_recv': 0,
            'eagle_cache_sent': 0,
            'eagle_cache_recv': 0,
            'aggregated_sends': 0,  # ring模式下聚合发送次数
            'forwarded_messages': 0,  # ring模式下转发消息数
            # 分支调度相关统计
            'schedule_plan_sent': 0,
            'schedule_plan_recv': 0,
            'branch_prompt_sent': 0,
            'branch_prompt_recv': 0,
            'branch_output_sent': 0,
            'branch_output_recv': 0,
            'branch_complete_sent': 0,
            'branch_complete_recv': 0,
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
        self.logger.debug("接收线程启动")
        
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
        
        self.logger.debug("接收线程退出")
    
    def _process_received_data(self, data: bytes):
        """
        处理接收到的原始数据
        子类可以重写此方法来处理批量消息
        """
        msg = MessageSerializer.deserialize(data, self.device)
        self._handle_received_message(msg)
    
    def _handle_received_message(self, msg: Message):
        """
        处理单条接收到的消息，并记录详细日志
        
        注意：通信细节使用DEBUG级别，只有DRAFT_TOKENS使用INFO
        """
        # 放入接收队列
        self.recv_queue.put(msg)
        
        # 获取有效来源rank
        src = msg.get_effective_src()
        recv_time = time.time()
        latency = recv_time - msg.timestamp if msg.timestamp > 0 else 0
        
        # 根据消息类型记录详细信息（大部分使用DEBUG级别）
        if msg.msg_type == MessageType.DRAFT_TOKENS:
            self.stats['draft_tokens_recv'] += 1
            data = msg.data
            if isinstance(data, dict):
                draft_info = get_tensor_info(data.get('draft_tokens'))
                retrieve_info = get_tensor_info(data.get('retrieve_indices'))
                mask_info = get_tensor_info(data.get('tree_mask'))
                pos_info = get_tensor_info(data.get('tree_position_ids'))
                # DRAFT_TOKENS是关键事件，使用INFO
                self.logger.info(
                    f"[RECV] DRAFT_TOKENS from rank {src} | "
                    f"seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
                )
                self.logger.debug(
                    f"  draft_tokens: {draft_info}\n"
                    f"  retrieve_indices: {retrieve_info}\n"
                    f"  tree_mask: {mask_info}\n"
                    f"  tree_position_ids: {pos_info}"
                )
            else:
                self.logger.info(f"[RECV] DRAFT_TOKENS from rank {src} | seq_id={msg.seq_id}")
                
        elif msg.msg_type == MessageType.HIDDEN:
            self.stats['hidden_recv'] += 1
            hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
            self.logger.debug(
                f"[RECV] HIDDEN from rank {src} | "
                f"chunk_idx={msg.chunk_idx}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
            self.logger.debug(f"  hidden_state: {hidden_info}")
            
        elif msg.msg_type == MessageType.EAGLE_INPUT_HIDDEN:
            self.stats['eagle_input_hidden_recv'] += 1
            hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
            self.logger.debug(
                f"[RECV] EAGLE_INPUT_HIDDEN from rank {src} | "
                f"layer_idx={msg.layer_idx}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
            self.logger.debug(f"  eagle_input_hidden: {hidden_info}")
            
        elif msg.msg_type == MessageType.BASE_CACHE:
            self.stats['base_cache_recv'] += 1
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_info = get_tensor_info(msg.data[0])
                value_info = get_tensor_info(msg.data[1])
                self.logger.debug(
                    f"[RECV] BASE_CACHE from rank {src} | "
                    f"layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
                )
                self.logger.debug(f"  key: {key_info}\n  value: {value_info}")
            else:
                self.logger.debug(
                    f"[RECV] BASE_CACHE from rank {src} | layer={msg.layer_idx}, chunk={msg.chunk_idx}"
                )
                
        elif msg.msg_type == MessageType.EAGLE_CACHE:
            self.stats['eagle_cache_recv'] += 1
            is_incremental = (msg.layer_idx != -1)
            if isinstance(msg.data, tuple) and len(msg.data) == 2:
                key_info = get_tensor_info(msg.data[0])
                value_info = get_tensor_info(msg.data[1])
                self.logger.debug(
                    f"[RECV] EAGLE_CACHE from rank {src} | "
                    f"chunk_idx={msg.chunk_idx}, incremental={is_incremental}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
                )
                self.logger.debug(f"  key: {key_info}\n  value: {value_info}")
            else:
                self.logger.debug(
                    f"[RECV] EAGLE_CACHE from rank {src} | chunk={msg.chunk_idx}, incremental={is_incremental}"
                )
        
        # 分支调度相关消息
        elif msg.msg_type == MessageType.SCHEDULE_PLAN:
            self.stats['schedule_plan_recv'] += 1
            self.logger.debug(
                f"[RECV] SCHEDULE_PLAN from rank {src} | "
                f"seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
            
        elif msg.msg_type == MessageType.BRANCH_PROMPT:
            self.stats['branch_prompt_recv'] += 1
            num_branches = len(msg.data) if isinstance(msg.data, list) else 1
            self.logger.debug(
                f"[RECV] BRANCH_PROMPT from rank {src} | "
                f"num_branches={num_branches}, seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
            
        elif msg.msg_type == MessageType.BRANCH_OUTPUT:
            self.stats['branch_output_recv'] += 1
            num_tokens = len(msg.data) if msg.data else 0
            self.logger.debug(
                f"[RECV] BRANCH_OUTPUT from rank {src} | "
                f"branch_id={msg.branch_id}, num_tokens={num_tokens}, "
                f"seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
            
        elif msg.msg_type == MessageType.BRANCH_COMPLETE:
            self.stats['branch_complete_recv'] += 1
            self.logger.debug(
                f"[RECV] BRANCH_COMPLETE from rank {src} | "
                f"seq_id={msg.seq_id}, latency={latency*1000:.2f}ms"
            )
    
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
        
        self.logger.debug(f"通信管理器已启动 ({self.__class__.__name__})")
    
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
        self.logger.debug("通信管理器已停止")
    
    # ==================== Draft Tokens 通信 (最高优先级) ====================
    
    @abstractmethod
    def send_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        dst_rank: int = -1
    ) -> None:
        """
        发送打包的draft tokens数据
        
        Args:
            draft_tokens: 候选tokens [batch, tree_size]
            retrieve_indices: 检索索引
            tree_mask: 树掩码
            tree_position_ids: 位置编码
            dst_rank: 目标rank，-1表示广播
        """
        pass
    
    def recv_draft_tokens(
        self,
        src_rank: int = -1,
        timeout: float = 60.0
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        接收draft tokens数据
        
        Args:
            src_rank: 源rank，-1表示从任意rank接收
            timeout: 超时时间（秒）
            
        Returns:
            (draft_tokens, retrieve_indices, tree_mask, tree_position_ids) 或 None
        """
        start_time = time.time()
        self.logger.debug(f"[WAIT] 等待接收 DRAFT_TOKENS (src_rank={src_rank}, timeout={timeout}s)")
        
        deadline = time.time() + timeout
        pending = []
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.DRAFT_TOKENS, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            effective_src = msg.get_effective_src()
            if src_rank == -1 or effective_src == src_rank:
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                # msg.data是打包的dict
                data = msg.data
                result = (
                    data['draft_tokens'].to(self.device),
                    data['retrieve_indices'].to(self.device),
                    data['tree_mask'].to(self.device),
                    data['tree_position_ids'].to(self.device)
                )
                wait_time = time.time() - start_time
                self.logger.info(
                    f"[RECV_OK] DRAFT_TOKENS 接收成功 | "
                    f"from rank {effective_src}, wait_time={wait_time*1000:.2f}ms, "
                    f"draft_tokens shape={tuple(result[0].shape)}"
                )
                return result
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        wait_time = time.time() - start_time
        self.logger.warning(f"[TIMEOUT] 接收 DRAFT_TOKENS 超时 | wait_time={wait_time:.2f}s")
        return None
    
    # ==================== Hidden State 通信 (第二优先级) ====================
    
    @abstractmethod
    def send_hidden(self, hidden: torch.Tensor, dst_rank: int, chunk_idx: int = -1) -> None:
        """发送hidden state（层间传输）"""
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
        start_time = time.time()
        self.logger.debug(f"[WAIT] 等待接收 HIDDEN (src_rank={src_rank}, timeout={timeout}s)")
        
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
                wait_time = time.time() - start_time
                hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
                self.logger.debug(
                    f"[RECV_OK] HIDDEN 接收成功 | "
                    f"from rank {effective_src}, chunk_idx={msg.chunk_idx}, "
                    f"wait_time={wait_time*1000:.2f}ms"
                )
                self.logger.debug(f"  hidden: {hidden_info}")
                return (msg.data, msg.chunk_idx)
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        wait_time = time.time() - start_time
        self.logger.warning(f"[TIMEOUT] 从 rank {src_rank} 接收 HIDDEN 超时 | wait_time={wait_time:.2f}s")
        return None
    
    # ==================== Eagle Input Hidden 通信 (第三优先级) ====================
    
    @abstractmethod
    def send_eagle_input_hidden(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        dst_rank: int,
        chunk_idx: int = -1
    ) -> None:
        """
        发送eagle layer输入的hidden state
        
        根据层数判断是否需要传输：
        - layer_idx == 2
        - layer_idx == num_layers // 2
        - layer_idx == num_layers - 3
        
        Args:
            hidden: hidden state tensor
            layer_idx: 层索引
            dst_rank: 目标rank（最后一个rank）
            chunk_idx: chunk索引，用于确保接收端正确匹配
        """
        pass
    
    def recv_eagle_input_hidden(
        self,
        layer_idx: int,
        chunk_idx: int = -1,
        timeout: float = 30.0
    ) -> Optional[torch.Tensor]:
        """
        接收eagle layer输入的hidden state
        
        Args:
            layer_idx: 期望的层索引
            chunk_idx: 期望的chunk索引，-1表示不检查chunk_idx
            timeout: 超时时间（秒）
            
        Returns:
            hidden state tensor 或 None
        """
        start_time = time.time()
        self.logger.debug(f"[WAIT] 等待接收 EAGLE_INPUT_HIDDEN (layer_idx={layer_idx}, chunk_idx={chunk_idx}, timeout={timeout}s)")
        
        deadline = time.time() + timeout
        pending = []
        
        while time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.EAGLE_INPUT_HIDDEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            # 检查layer_idx匹配，如果指定了chunk_idx也要检查chunk_idx匹配
            layer_match = (msg.layer_idx == layer_idx)
            chunk_match = (chunk_idx == -1 or msg.chunk_idx == chunk_idx)
            
            if layer_match and chunk_match:
                for pending_msg in pending:
                    self.recv_queue.put(pending_msg)
                wait_time = time.time() - start_time
                hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
                self.logger.debug(
                    f"[RECV_OK] EAGLE_INPUT_HIDDEN 接收成功 | "
                    f"layer_idx={layer_idx}, chunk_idx={msg.chunk_idx}, from rank {msg.get_effective_src()}, "
                    f"wait_time={wait_time*1000:.2f}ms"
                )
                self.logger.debug(f"  hidden: {hidden_info}")
                return msg.data
            else:
                pending.append(msg)
        
        for pending_msg in pending:
            self.recv_queue.put(pending_msg)
        
        wait_time = time.time() - start_time
        self.logger.warning(f"[TIMEOUT] 接收 EAGLE_INPUT_HIDDEN 超时 (layer={layer_idx}, chunk={chunk_idx}) | wait_time={wait_time:.2f}s")
        return None
    
    def recv_all_eagle_input_hidden(
        self,
        eagle_input_layers: List[int],
        my_start_layer: int,
        my_end_layer: int,
        chunk_idx: int = -1,
        timeout: float = 60.0
    ) -> Dict[int, torch.Tensor]:
        """
        接收其他rank发送的eagle input hidden states
        
        只有最后一个rank需要调用此方法
        
        Args:
            eagle_input_layers: 需要收集的层索引列表 [2, num_layers//2, num_layers-3]
            my_start_layer: 本rank负责的起始层
            my_end_layer: 本rank负责的结束层
            chunk_idx: 期望的chunk索引，用于确保接收正确chunk的数据
            timeout: 超时时间（秒）
            
        Returns:
            {layer_idx: hidden_state} 字典
        """
        start_time = time.time()
        
        # 确定需要从其他rank接收的层（不在本rank负责范围内的层）
        expected_layers = []
        for layer_idx in eagle_input_layers:
            if not (my_start_layer <= layer_idx < my_end_layer):
                expected_layers.append(layer_idx)
        
        if not expected_layers:
            self.logger.debug(f"[SKIP] 所有 eagle_input_layers 都在本 rank 范围内，无需接收")
            return {}
        
        self.logger.debug(
            f"[WAIT] 等待接收 ALL_EAGLE_INPUT_HIDDEN | "
            f"expected_layers={expected_layers}, chunk_idx={chunk_idx}, my_range=[{my_start_layer}, {my_end_layer}), timeout={timeout}s"
        )
        
        result = {}
        deadline = time.time() + timeout
        remaining_layers = set(expected_layers)
        
        while remaining_layers and time.time() < deadline:
            remaining = deadline - time.time()
            msg = self.recv_queue.get_by_type(MessageType.EAGLE_INPUT_HIDDEN, timeout=min(0.1, remaining))
            
            if msg is None:
                continue
            
            # 检查layer_idx和chunk_idx是否匹配
            layer_match = (msg.layer_idx in remaining_layers)
            chunk_match = (chunk_idx == -1 or msg.chunk_idx == chunk_idx)
            
            if layer_match and chunk_match:
                result[msg.layer_idx] = msg.data
                remaining_layers.discard(msg.layer_idx)
                hidden_info = get_tensor_info(msg.data) if torch.is_tensor(msg.data) else str(type(msg.data))
                self.logger.debug(
                    f"[RECV_OK] EAGLE_INPUT_HIDDEN | layer_idx={msg.layer_idx}, chunk_idx={msg.chunk_idx}, "
                    f"from rank {msg.get_effective_src()}, remaining={remaining_layers}"
                )
                self.logger.debug(f"  hidden: {hidden_info}")
            else:
                # 不匹配则放回队列（可能是其他chunk的数据）
                self.recv_queue.put(msg)
                if not chunk_match:
                    self.logger.debug(
                        f"[PENDING] EAGLE_INPUT_HIDDEN chunk不匹配 | "
                        f"expected_chunk={chunk_idx}, received_chunk={msg.chunk_idx}, layer={msg.layer_idx}"
                    )
        
        wait_time = time.time() - start_time
        if remaining_layers:
            self.logger.warning(
                f"[TIMEOUT] 接收 EAGLE_INPUT_HIDDEN 不完整 | "
                f"chunk_idx={chunk_idx}, missing_layers={remaining_layers}, wait_time={wait_time:.2f}s"
            )
        else:
            self.logger.debug(
                f"[RECV_OK] ALL_EAGLE_INPUT_HIDDEN 全部接收完成 | "
                f"chunk_idx={chunk_idx}, layers={list(result.keys())}, wait_time={wait_time*1000:.2f}ms"
            )
        
        return result
    
    # ==================== Base Model Cache 通信 (第四优先级) ====================
    
    @abstractmethod
    def send_base_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        layer_idx: int,
        chunk_idx: int = -1
    ) -> None:
        """异步发送base model的cache"""
        pass
    
    def get_received_base_cache(
        self, 
        timeout: float = 0.1
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], int, int, int]]:
        """
        从接收队列获取base cache（非阻塞）
        
        Returns:
            (kv_cache, src_rank, layer_idx, chunk_idx) 或 None
        """
        msg = self.recv_queue.get_by_type(MessageType.BASE_CACHE, timeout=timeout)
        if msg:
            return (msg.data, msg.get_effective_src(), msg.layer_idx, msg.chunk_idx)
        return None
    
    # ==================== Eagle Layer Cache 通信 (第五优先级) ====================
    
    @abstractmethod
    def send_eagle_cache_async(
        self,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_rank: int,
        is_incremental: bool = False,
        chunk_idx: int = -1
    ) -> None:
        """
        异步发送eagle layer的cache
        
        在expand_root中立即调用，不等待draft完成
        
        Args:
            kv_cache: (key, value) 元组
            dst_rank: 目标rank
            is_incremental: 是否为增量更新
            chunk_idx: chunk索引
        """
        pass
    
    def get_received_eagle_cache(
        self, 
        timeout: float = 0.1
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], int, bool, int]]:
        """
        从接收队列获取eagle cache（非阻塞）
        
        Returns:
            (kv_cache, src_rank, is_incremental, chunk_idx) 或 None
        """
        msg = self.recv_queue.get_by_type(MessageType.EAGLE_CACHE, timeout=timeout)
        if msg:
            # layer_idx用于标识是否为增量: -1表示完整, 其他值表示增量
            is_incremental = (msg.layer_idx != -1)
            return (msg.data, msg.get_effective_src(), is_incremental, msg.chunk_idx)
        return None
    
    # ==================== Eagle Stable KV 广播方法 ====================
    
    @abstractmethod
    def broadcast_eagle_draft_past_key_values(
        self,
        incremental_kv: Tuple[torch.Tensor, torch.Tensor],
        chunk_idx: int,
    ) -> None:
        """
        广播Eagle Layer的增量draft_past_key_values给所有其他rank
        
        这是分布式Prefill中用于同步eagle draft_past_key_values的核心方法。
        只有最后一个rank需要调用此方法，将增量KV发送给其他rank。
        
        Args:
            incremental_kv: 增量KV cache (new_key, new_value)
            chunk_idx: chunk索引，用于接收端排序和去重
        """
        pass
    
    # ==================== Cache 同步等待方法 ====================
    
    def wait_for_all_caches(
        self,
        past_key_values: List,
        eagle_layer: Any,
        num_layers: int,
        num_chunks: int,
        cache_received_indicator: torch.Tensor,
        eagle_cache_received_indicator: torch.Tensor,
        start_layer: int,
        end_layer: int,
        is_last_rank: bool,
        timeout: float = 60.0
    ) -> Tuple[int, int]:
        """
        等待接收所有cache（base cache和eagle cache）
        
        在prefill结束之前调用，确保所有cache都接收完毕
        
        Args:
            past_key_values: Base Model的KV Cache列表
            eagle_layer: Eagle Layer模型（用于设置draft_past_key_values）
            num_layers: 模型层数
            num_chunks: chunk数量
            cache_received_indicator: base cache接收状态 [num_chunks, num_layers]
            eagle_cache_received_indicator: eagle cache接收状态 [num_chunks]
            start_layer: 本rank负责的起始层
            end_layer: 本rank负责的结束层
            is_last_rank: 是否是最后一个rank
            timeout: 超时时间（秒）
            
        Returns:
            (remaining_base_caches, remaining_eagle_caches): 未接收完的cache数量
        """
        start_time = time.time()
        
        # 计算需要接收的base cache数量
        expected_base_caches = 0
        layers_to_receive = []
        for chunk_idx in range(num_chunks):
            for layer_idx in range(num_layers):
                if not (start_layer <= layer_idx < end_layer):
                    if cache_received_indicator[chunk_idx, layer_idx] == 0:
                        expected_base_caches += 1
                        layers_to_receive.append((chunk_idx, layer_idx))
        
        # 计算需要接收的eagle cache数量（非最后rank需要接收）
        expected_eagle_caches = 0
        if not is_last_rank:
            for chunk_idx in range(num_chunks):
                if eagle_cache_received_indicator[chunk_idx] == 0:
                    expected_eagle_caches += 1
        
        self.logger.debug(
            f"[WAIT] 等待接收所有 Cache | "
            f"expected_base_caches={expected_base_caches}, expected_eagle_caches={expected_eagle_caches}, "
            f"my_layer_range=[{start_layer}, {end_layer}), timeout={timeout}s"
        )
        self.logger.debug(f"  待接收的 base cache: {layers_to_receive[:20]}{'...' if len(layers_to_receive) > 20 else ''}")
        
        base_cache_received = 0
        eagle_cache_received = 0
        
        # 接收base cache
        while expected_base_caches > 0 and (time.time() - start_time) < timeout:
            result = self.get_received_base_cache(timeout=0.1)
            if result is None:
                continue
            
            kv_cache, src_rank, layer_idx, chunk_idx = result
            key, value = kv_cache
            
            if cache_received_indicator[chunk_idx, layer_idx] == 0:
                past_key_values[layer_idx][0].cat(key)
                past_key_values[layer_idx][1].cat(value)
                cache_received_indicator[chunk_idx, layer_idx] = 1
                expected_base_caches -= 1
                base_cache_received += 1
                
                key_info = get_tensor_info(key)
                self.logger.debug(
                    f"[CACHE_RECV] BASE_CACHE | layer={layer_idx}, chunk={chunk_idx}, "
                    f"from rank={src_rank}, remaining={expected_base_caches}"
                )
                self.logger.debug(f"  key: {key_info}")
        
        # 接收eagle cache（非最后rank）
        if not is_last_rank:
            while expected_eagle_caches > 0 and (time.time() - start_time) < timeout:
                result = self.get_received_eagle_cache(timeout=0.1)
                if result is None:
                    continue
                
                kv_cache, src_rank, is_incremental, chunk_idx = result
                key, value = kv_cache
                
                # 处理eagle cache
                if chunk_idx >= 0 and eagle_cache_received_indicator[chunk_idx] == 0:
                    # 设置Eagle Layer的draft_past_key_values
                    # 检查是否使用 KVCache 类（统一使用 KVCache 类管理）
                    if eagle_layer.kv_cache_initialized and eagle_layer.draft_past_key_values is not None:
                        # 使用 KVCache 类的 cat 方法追加数据
                        key_cache, value_cache = eagle_layer.draft_past_key_values[0]
                        key_cache.cat(key, dim=2)
                        value_cache.cat(value, dim=2)
                        current_kv_length = key_cache.current_length.item()
                    else:
                        # 未初始化时直接设置（不应该发生，因为 prefill 时会初始化）
                        self.logger.warning("Eagle KV Cache 未初始化，使用普通 tensor 格式")
                        if eagle_layer.draft_past_key_values is None:
                            eagle_layer.draft_past_key_values = ((key, value),)
                        else:
                            old_key, old_value = eagle_layer.draft_past_key_values[0]
                            # 兼容 KVCache 类和普通 tensor
                            if hasattr(old_key, 'data'):
                                old_key_tensor = old_key.data[:, :, :old_key.current_length.item(), :]
                                old_value_tensor = old_value.data[:, :, :old_value.current_length.item(), :]
                            else:
                                old_key_tensor = old_key
                                old_value_tensor = old_value
                            new_key = torch.cat([old_key_tensor, key], dim=2)
                            new_value = torch.cat([old_value_tensor, value], dim=2)
                            eagle_layer.draft_past_key_values = ((new_key, new_value),)
                        current_kv_length = eagle_layer.draft_past_key_values[0][0].shape[2]
                    
                    eagle_cache_received_indicator[chunk_idx] = 1
                    expected_eagle_caches -= 1
                    eagle_cache_received += 1
                    
                    key_info = get_tensor_info(key)
                    self.logger.debug(
                        f"[CACHE_RECV] EAGLE_CACHE | chunk={chunk_idx}, incremental={is_incremental}, "
                        f"from rank={src_rank}, kv_length={current_kv_length}, remaining={expected_eagle_caches}"
                    )
                    self.logger.debug(f"  key: {key_info}")
        
        wait_time = time.time() - start_time
        
        # 检查是否有超时
        if expected_base_caches > 0:
            self.logger.warning(
                f"[TIMEOUT] BASE_CACHE 接收不完整 | "
                f"received={base_cache_received}, missing={expected_base_caches}, wait_time={wait_time:.2f}s"
            )
        if expected_eagle_caches > 0:
            self.logger.warning(
                f"[TIMEOUT] EAGLE_CACHE 接收不完整 | "
                f"received={eagle_cache_received}, missing={expected_eagle_caches}, wait_time={wait_time:.2f}s"
            )
        
        if expected_base_caches == 0 and expected_eagle_caches == 0:
            self.logger.info(
                f"[SYNC_OK] 所有 Cache 接收完成 | "
                f"base_cache={base_cache_received}, eagle_cache={eagle_cache_received}, "
                f"wait_time={wait_time*1000:.2f}ms"
            )
        
        return (expected_base_caches, expected_eagle_caches)
    
    # ==================== 分支调度通信 (最高优先级，异步) ====================
    
    @abstractmethod
    def send_schedule_plan_async(
        self,
        schedule_plan_data: Dict[str, Any],
        dst_rank: int = -1,
    ) -> None:
        """
        异步广播调度计划
        
        Args:
            schedule_plan_data: 序列化的调度计划数据
            dst_rank: 目标rank，-1表示广播到所有其他rank
        """
        pass
    
    def recv_schedule_plan(
        self,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        接收调度计划（非阻塞轮询）
        
        Args:
            timeout: 超时时间
            
        Returns:
            调度计划数据或None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.recv_queue.get_by_type(MessageType.SCHEDULE_PLAN, timeout=0.1)
            if msg is not None:
                self.logger.debug(
                    f"[RECV_OK] SCHEDULE_PLAN | from rank {msg.get_effective_src()}"
                )
                return msg.data
        
        self.logger.warning(f"[TIMEOUT] 接收 SCHEDULE_PLAN 超时")
        return None
    
    @abstractmethod
    def send_branch_prompt_async(
        self,
        branch_data: Dict[str, Any],
        dst_rank: int,
        branch_id: int = -1,
    ) -> None:
        """
        异步发送分支Prompt
        
        Args:
            branch_data: 分支数据（包含branch_id, title, predicted_length, prompt_tokens）
            dst_rank: 目标rank
            branch_id: 分支ID
        """
        pass
    
    def recv_branch_prompts(
        self,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        接收分支Prompts（收集所有已到达的）
        
        Args:
            timeout: 超时时间
            
        Returns:
            分支数据列表
        """
        result = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.recv_queue.get_by_type(MessageType.BRANCH_PROMPT, timeout=0.1)
            if msg is not None:
                # msg.data 可能是单个分支或分支列表
                if isinstance(msg.data, list):
                    result.extend(msg.data)
                else:
                    result.append(msg.data)
                self.logger.debug(
                    f"[RECV_OK] BRANCH_PROMPT | {len(result)} branches received"
                )
                # 继续尝试接收更多（非阻塞检查）
                while True:
                    msg2 = self.recv_queue.get_by_type(MessageType.BRANCH_PROMPT, timeout=0.01)
                    if msg2 is None:
                        break
                    if isinstance(msg2.data, list):
                        result.extend(msg2.data)
                    else:
                        result.append(msg2.data)
                break  # 收到第一批后退出
        
        return result
    
    @abstractmethod
    def send_branch_output_async(
        self,
        branch_id: int,
        output_tokens: List[int],
        dst_rank: int = 0,
    ) -> None:
        """
        异步发送分支输出（完成后立即发送）
        
        Args:
            branch_id: 分支ID
            output_tokens: 生成的token列表
            dst_rank: 目标rank（默认发送到master rank 0）
        """
        pass
    
    def recv_branch_output(
        self,
        timeout: float = 0.1,
    ) -> Optional[Tuple[int, List[int]]]:
        """
        接收单个分支输出（非阻塞）
        
        Args:
            timeout: 超时时间
            
        Returns:
            (branch_id, output_tokens) 或 None
        """
        msg = self.recv_queue.get_by_type(MessageType.BRANCH_OUTPUT, timeout=timeout)
        if msg is not None:
            self.logger.debug(
                f"[RECV_OK] BRANCH_OUTPUT | branch_id={msg.branch_id}, "
                f"tokens={len(msg.data) if msg.data else 0}"
            )
            return (msg.branch_id, msg.data)
        return None
    
    def collect_all_branch_outputs(
        self,
        num_branches: int,
        timeout: float = 300.0,
    ) -> Dict[int, List[int]]:
        """
        收集所有分支输出
        
        Args:
            num_branches: 预期分支数量
            timeout: 超时时间
            
        Returns:
            {branch_id: output_tokens} 字典
        """
        result = {}
        start_time = time.time()
        
        while len(result) < num_branches and (time.time() - start_time) < timeout:
            output = self.recv_branch_output(timeout=0.1)
            if output is not None:
                branch_id, tokens = output
                result[branch_id] = tokens
                self.logger.debug(
                    f"[COLLECT] BRANCH_OUTPUT | branch_id={branch_id}, "
                    f"collected={len(result)}/{num_branches}"
                )
        
        if len(result) < num_branches:
            self.logger.warning(
                f"[TIMEOUT] 收集分支输出不完整 | "
                f"collected={len(result)}/{num_branches}"
            )
        else:
            self.logger.info(
                f"[COLLECT_OK] 所有分支输出收集完成 | count={len(result)}"
            )
        
        return result
    
    @abstractmethod
    def send_branch_complete_async(
        self,
        dst_rank: int = -1,
    ) -> None:
        """
        异步广播完成信号
        
        Args:
            dst_rank: 目标rank，-1表示广播
        """
        pass
    
    def recv_complete_signal(
        self,
        timeout: float = 300.0,
    ) -> bool:
        """
        等待接收完成信号
        
        Args:
            timeout: 超时时间
            
        Returns:
            是否收到完成信号
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.recv_queue.get_by_type(MessageType.BRANCH_COMPLETE, timeout=0.1)
            if msg is not None:
                self.logger.debug(f"[RECV_OK] BRANCH_COMPLETE signal")
                return True
        
        self.logger.warning(f"[TIMEOUT] 等待完成信号超时")
        return False
    
    def broadcast_parallel_complete_signal(self) -> None:
        """
        广播并行阶段完成信号
        
        当 skeleton 解析为 direct/error 模式时调用此方法，
        通知所有 worker 不需要执行并行任务。
        """
        self.send_branch_complete_async(dst_rank=-1)
    
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

