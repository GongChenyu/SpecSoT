"""
KV Cache同步管理器
支持两种同步策略：
1. Pairwise (两两通信): 所有设备两两通信，完成全局同步
2. Ring (环形通信): 设备按环形拓扑传递cache
"""

import torch
import torch.distributed as dist
import threading
import queue
import time
from typing import List, Tuple, Optional
import logging


class CacheSyncManager:
    """
    KV Cache同步管理器
    使用独立线程进行异步cache同步，避免阻塞主推理流程
    """
    
    def __init__(
        self, 
        rank: int, 
        world_size: int, 
        strategy: str = "pairwise",
        device: str = "cuda",
        backend: str = "nccl"
    ):
        """
        Args:
            rank: 当前设备rank
            world_size: 总设备数
            strategy: 同步策略 ("pairwise" 或 "ring")
            device: 设备类型
            backend: 通信后端 ("nccl" 或 "gloo")
        """
        self.rank = rank
        self.world_size = world_size
        self.strategy = strategy
        self.device = device
        self.backend = backend
        
        self.logger = logging.getLogger(f"CacheSyncManager-Rank{rank}")
        
        # 同步队列：用于主线程和同步线程之间传递数据
        self.sync_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 同步线程
        self.sync_thread = None
        self.is_running = False
        
        # 统计信息
        self.sync_count = 0
        self.total_sync_time = 0.0
        
    def start_sync_thread(self):
        """启动同步线程"""
        if self.sync_thread is not None and self.sync_thread.is_alive():
            self.logger.warning("同步线程已经在运行")
            return
            
        self.is_running = True
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        self.logger.info(f"同步线程已启动 (策略: {self.strategy})")
        
    def stop_sync_thread(self):
        """停止同步线程"""
        if self.sync_thread is None:
            return
            
        self.is_running = False
        self.sync_queue.put(None)  # 发送停止信号
        self.sync_thread.join(timeout=5.0)
        self.logger.info("同步线程已停止")
        
    def _sync_worker(self):
        """同步线程的工作函数"""
        self.logger.info("同步线程开始工作")
        
        while self.is_running:
            try:
                # 从队列获取待同步的cache
                item = self.sync_queue.get(timeout=1.0)
                
                if item is None:  # 停止信号
                    break
                    
                layer_idx, kv_cache, sync_id = item
                
                # 执行同步
                start_time = time.time()
                
                if self.strategy == "pairwise":
                    synced_cache = self._pairwise_sync(layer_idx, kv_cache)
                elif self.strategy == "ring":
                    synced_cache = self._ring_sync(layer_idx, kv_cache)
                else:
                    raise ValueError(f"未知的同步策略: {self.strategy}")
                
                sync_time = time.time() - start_time
                self.total_sync_time += sync_time
                self.sync_count += 1
                
                self.logger.debug(
                    f"Layer {layer_idx} 同步完成 (耗时: {sync_time*1000:.2f}ms)"
                )
                
                # 将结果放入结果队列
                self.result_queue.put((sync_id, synced_cache))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"同步过程出错: {e}", exc_info=True)
                
        self.logger.info("同步线程退出")
        
    def async_sync_cache(
        self, 
        layer_idx: int, 
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        sync_id: int
    ):
        """
        异步提交cache同步请求
        
        Args:
            layer_idx: 层索引
            kv_cache: (key_cache, value_cache) tuple
            sync_id: 同步请求ID
        """
        self.sync_queue.put((layer_idx, kv_cache, sync_id))
        
    def wait_for_sync(self, sync_id: int, timeout: float = 30.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        等待特定的同步请求完成
        
        Args:
            sync_id: 同步请求ID
            timeout: 超时时间（秒）
            
        Returns:
            同步后的kv_cache
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_id, synced_cache = self.result_queue.get(timeout=1.0)
                if result_id == sync_id:
                    return synced_cache
                else:
                    # 不是我们要的结果，放回队列
                    self.result_queue.put((result_id, synced_cache))
            except queue.Empty:
                continue
                
        raise TimeoutError(f"等待同步请求 {sync_id} 超时")
        
    def _pairwise_sync(
        self, 
        layer_idx: int, 
        kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        两两通信策略：使用点对点通信，无需全局同步
        
        通信模式（以3个设备为例）：
        - 每个rank与其他所有rank进行P2P通信
        - 使用非阻塞通信避免死锁
        - 不需要所有rank同时到达
        
        每个设备最终获得所有设备的完整cache
        """
        key_cache, value_cache = kv_cache
        
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} 开始P2P同步 (backend={self.backend}), cache shape: K={key_cache.shape}, V={value_cache.shape}")
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} Key contiguous: {key_cache.is_contiguous()}, Value contiguous: {value_cache.is_contiguous()}")
        
        # Gloo后端需要在CPU上通信
        if self.backend == 'gloo':
            key_cache_comm = key_cache.cpu()
            value_cache_comm = value_cache.cpu()
        else:
            key_cache_comm = key_cache
            value_cache_comm = value_cache
        
        # 存储所有rank的cache（包括自己的）
        all_key_caches = [None] * self.world_size
        all_value_caches = [None] * self.world_size
        
        # 先存储自己的cache（保持在GPU上）
        all_key_caches[self.rank] = key_cache
        all_value_caches[self.rank] = value_cache
        
        # 准备发送和接收的请求列表
        send_requests = []
        recv_requests = []
        recv_buffers = []  # 用于gloo后端保存接收缓冲区
        
        # 与其他所有rank进行P2P通信
        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue
            
            # 发送自己的cache给other_rank（非阻塞）
            send_req_k = dist.isend(key_cache_comm, dst=other_rank)
            send_req_v = dist.isend(value_cache_comm, dst=other_rank)
            send_requests.extend([send_req_k, send_req_v])
            
            # 准备接收缓冲区
            if self.backend == 'gloo':
                # Gloo: 在CPU上接收
                recv_key = torch.zeros_like(key_cache_comm)
                recv_value = torch.zeros_like(value_cache_comm)
            else:
                # NCCL: 在GPU上接收
                recv_key = torch.zeros_like(key_cache)
                recv_value = torch.zeros_like(value_cache)
            
            # 从other_rank接收cache（非阻塞）
            recv_req_k = dist.irecv(recv_key, src=other_rank)
            recv_req_v = dist.irecv(recv_value, src=other_rank)
            recv_requests.extend([recv_req_k, recv_req_v])
            
            # 保存接收缓冲区引用
            recv_buffers.append((other_rank, recv_key, recv_value))
        
        # 等待所有发送完成
        for req in send_requests:
            req.wait()
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} 发送完成")
        
        # 等待所有接收完成
        for req in recv_requests:
            req.wait()
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} 接收完成")
        
        # 将接收到的数据移到GPU（如果使用gloo）并存储
        for other_rank, recv_key, recv_value in recv_buffers:
            if self.backend == 'gloo':
                all_key_caches[other_rank] = recv_key.to(key_cache.device)
                all_value_caches[other_rank] = recv_value.to(value_cache.device)
            else:
                all_key_caches[other_rank] = recv_key
                all_value_caches[other_rank] = recv_value
        
        # 按rank顺序拼接所有cache
        merged_key = torch.cat(all_key_caches, dim=2)  # dim=2是seq_len维度
        merged_value = torch.cat(all_value_caches, dim=2)
        
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} 拼接后 shape: K={merged_key.shape}, V={merged_value.shape}")
        
        return (merged_key, merged_value)
        
    def _ring_sync(
        self, 
        layer_idx: int, 
        kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        环形通信策略：使用P2P点对点通信实现环形传递
        
        通信流程（以3个设备为例）：
        初始: Rank0有cache0, Rank1有cache1, Rank2有cache2
        
        使用P2P通信，每轮环形传递一个cache：
        Step 1: 每个rank向下一个rank发送，从上一个rank接收
        Step 2: 重复world_size-1轮
        
        优势：不需要全局同步，各rank可以异步执行
        """
        key_cache, value_cache = kv_cache
        
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} 开始Ring P2P同步 (backend={self.backend})")
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} Key contiguous: {key_cache.is_contiguous()}, Value contiguous: {value_cache.is_contiguous()}")
        
        # 使用字典存储来自不同rank的cache，key是原始rank编号
        all_keys = {self.rank: key_cache}
        all_values = {self.rank: value_cache}
        
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        
        # 追踪当前持有的cache来源（用于传递）
        current_cache_from_rank = self.rank
        
        # 进行world_size-1轮传递，每轮传递一个cache
        for step in range(self.world_size - 1):
            # 发送当前持有的cache（这是从current_cache_from_rank来的）
            send_keys = all_keys[current_cache_from_rank].contiguous()
            send_values = all_values[current_cache_from_rank].contiguous()
            
            # Gloo后端需要在CPU上通信
            if self.backend == 'gloo':
                send_keys_comm = send_keys.cpu()
                send_values_comm = send_values.cpu()
                recv_keys = torch.empty_like(key_cache).cpu().contiguous()
                recv_values = torch.empty_like(value_cache).cpu().contiguous()
            else:
                send_keys_comm = send_keys
                send_values_comm = send_values
                recv_keys = torch.empty_like(key_cache).contiguous()
                recv_values = torch.empty_like(value_cache).contiguous()
            
            # 使用P2P通信同时发送和接收（非阻塞）
            recv_req_k = dist.irecv(recv_keys, src=prev_rank)
            send_req_k = dist.isend(send_keys_comm, dst=next_rank)
            recv_req_v = dist.irecv(recv_values, src=prev_rank)
            send_req_v = dist.isend(send_values_comm, dst=next_rank)
            
            # 等待通信完成
            recv_req_k.wait()
            recv_req_v.wait()
            send_req_k.wait()
            send_req_v.wait()
            
            # 更新：接收到的cache来自prev_rank传递的cache
            received_from_rank = (current_cache_from_rank - 1 + self.world_size) % self.world_size
            
            # Gloo后端需要移回GPU
            if self.backend == 'gloo':
                all_keys[received_from_rank] = recv_keys.to(key_cache.device)
                all_values[received_from_rank] = recv_values.to(value_cache.device)
            else:
                all_keys[received_from_rank] = recv_keys
                all_values[received_from_rank] = recv_values
            
            # 下一轮要传递的cache来源更新
            current_cache_from_rank = received_from_rank
            
        # 按rank顺序拼接所有cache
        sorted_ranks = sorted(all_keys.keys())
        merged_key = torch.cat([all_keys[r] for r in sorted_ranks], dim=2)
        merged_value = torch.cat([all_values[r] for r in sorted_ranks], dim=2)
        
        self.logger.debug(f"  [Layer {layer_idx}] Rank {self.rank} Ring同步完成")
        
        return (merged_key, merged_value)
        
    def sync_all_layers_sync(
        self, 
        all_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        同步所有层的cache（同步版本，阻塞直到完成）
        
        Args:
            all_kv_caches: 所有层的KV cache列表
            
        Returns:
            同步后的所有层KV cache
        """
        start_time = time.time()
        self.logger.info(f"开始同步 {len(all_kv_caches)} 层的cache (策略: {self.strategy})")
        
        synced_caches = []
        
        for layer_idx, kv_cache in enumerate(all_kv_caches):
            if self.strategy == "pairwise":
                synced_cache = self._pairwise_sync(layer_idx, kv_cache)
            elif self.strategy == "ring":
                synced_cache = self._ring_sync(layer_idx, kv_cache)
            else:
                raise ValueError(f"未知的同步策略: {self.strategy}")
                
            synced_caches.append(synced_cache)
            
        sync_time = time.time() - start_time
        self.logger.info(f"所有层cache同步完成 (耗时: {sync_time:.3f}s)")
        
        return synced_caches
        
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'sync_count': self.sync_count,
            'total_sync_time': self.total_sync_time,
            'avg_sync_time': self.total_sync_time / self.sync_count if self.sync_count > 0 else 0
        }
        
    def reset_stats(self):
        """重置统计信息"""
        self.sync_count = 0
        self.total_sync_time = 0.0


class StreamingSyncManager(CacheSyncManager):
    """
    流式同步管理器：支持计算和通信重叠
    每计算完一层就立即开始同步该层的cache
    """
    
    def __init__(self, rank: int, world_size: int, strategy: str = "pairwise", device: str = "cuda", backend: str = "nccl"):
        super().__init__(rank, world_size, strategy, device, backend)
        
        # 用于追踪每层的同步状态
        self.layer_sync_status = {}  # {layer_idx: sync_id}
        self.next_sync_id = 0
        
    def submit_layer_sync(self, layer_idx: int, kv_cache: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        提交单层cache进行异步同步
        
        Returns:
            sync_id: 用于后续查询同步结果
        """
        sync_id = self.next_sync_id
        self.next_sync_id += 1
        
        self.async_sync_cache(layer_idx, kv_cache, sync_id)
        self.layer_sync_status[layer_idx] = sync_id
        
        return sync_id
        
    def wait_all_layers(self, num_layers: int, timeout: float = 60.0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        等待所有层的同步完成
        
        Args:
            num_layers: 总层数
            timeout: 超时时间
            
        Returns:
            按层顺序排列的同步后cache列表
        """
        synced_caches = [None] * num_layers
        
        for layer_idx in range(num_layers):
            if layer_idx not in self.layer_sync_status:
                raise ValueError(f"Layer {layer_idx} 未提交同步请求")
                
            sync_id = self.layer_sync_status[layer_idx]
            synced_cache = self.wait_for_sync(sync_id, timeout)
            synced_caches[layer_idx] = synced_cache
            
        return synced_caches
        
    def clear_status(self):
        """清空同步状态"""
        self.layer_sync_status.clear()


# 便捷函数
def create_sync_manager(
    rank: int, 
    world_size: int, 
    strategy: str = "pairwise",
    streaming: bool = False,
    backend: str = "nccl"
) -> CacheSyncManager:
    """
    创建cache同步管理器
    
    Args:
        rank: 当前设备rank
        world_size: 总设备数
        strategy: 同步策略 ("pairwise" 或 "ring")
        streaming: 是否使用流式同步
        backend: 通信后端 ("nccl" 或 "gloo")
        
    Returns:
        CacheSyncManager实例
    """
    if streaming:
        return StreamingSyncManager(rank, world_size, strategy, backend=backend)
    else:
        return CacheSyncManager(rank, world_size, strategy, backend=backend)
