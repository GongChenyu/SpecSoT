# coding=utf-8
"""
分布式推理配置类

该模块提供分布式推理的配置管理，包括：
- 分布式启用状态
- 层拆分策略
- 通信配置
- 设备映射

使用示例：
    >>> config = DistributedConfig.from_args(
    ...     enabled=True,
    ...     layer_splits=[14, 28],  # 3台设备拆分36层模型
    ...     rank=0,
    ...     world_size=3
    ... )
    >>> start, end = config.get_layer_range(num_layers=36)
    >>> print(f"Rank 0 负责层: {start}-{end}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DistributedConfig:
    """
    分布式推理配置
    
    Attributes:
        enabled: 是否启用分布式推理
        rank: 当前设备的rank (0, 1, 2, ...)
        world_size: 总设备数
        layer_splits: 层拆分点列表，如 [14, 28] 表示:
                     - rank 0: 层 0-13 (共14层)
                     - rank 1: 层 14-27 (共14层)
                     - rank 2: 层 28-end + eagle layer
        base_port: ZMQ通信基础端口
        comm_mode: 通信模式 ("p2p" 或 "ring")
        node_addresses: 节点地址映射 {rank: ip}
        startup_delay: 启动延迟（秒），等待其他节点就绪
        chunk_size: Sequence Parallel的chunk大小
    """
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    layer_splits: List[int] = field(default_factory=list)
    base_port: int = 45000
    comm_mode: str = "ring"
    node_addresses: Optional[Dict[int, str]] = None
    startup_delay: float = 3.0
    chunk_size: int = 128
    
    def __post_init__(self):
        """验证配置有效性"""
        if self.enabled:
            if self.world_size < 2:
                raise ValueError(f"分布式模式下world_size必须>=2, 当前: {self.world_size}")
            if self.rank >= self.world_size:
                raise ValueError(f"rank ({self.rank}) 必须小于 world_size ({self.world_size})")
            if len(self.layer_splits) != self.world_size - 1:
                raise ValueError(
                    f"layer_splits长度必须为world_size-1, "
                    f"期望 {self.world_size - 1}, 实际 {len(self.layer_splits)}"
                )
            # 验证拆分点递增
            for i in range(1, len(self.layer_splits)):
                if self.layer_splits[i] <= self.layer_splits[i-1]:
                    raise ValueError(f"layer_splits必须严格递增: {self.layer_splits}")
    
    @classmethod
    def from_args(
        cls,
        enabled: bool = False,
        rank: int = 0,
        world_size: int = 1,
        layer_splits: Optional[List[int]] = None,
        base_port: int = 45000,
        comm_mode: str = "ring",
        node_addresses: Optional[Dict[int, str]] = None,
        startup_delay: float = 2.0,
        chunk_size: int = 128,
    ) -> "DistributedConfig":
        """
        从参数创建配置
        
        Args:
            enabled: 是否启用分布式
            rank: 当前rank
            world_size: 总设备数
            layer_splits: 层拆分点，如 [14, 28]
            base_port: 基础端口
            comm_mode: 通信模式
            node_addresses: 节点地址
            startup_delay: 启动延迟
            chunk_size: chunk大小
            
        Returns:
            DistributedConfig实例
        """
        return cls(
            enabled=enabled,
            rank=rank,
            world_size=world_size,
            layer_splits=layer_splits or [],
            base_port=base_port,
            comm_mode=comm_mode,
            node_addresses=node_addresses,
            startup_delay=startup_delay,
            chunk_size=chunk_size,
        )
    
    @classmethod
    def from_layer_splits_str(
        cls,
        layer_splits_str: str,
        rank: int,
        world_size: int,
        **kwargs
    ) -> "DistributedConfig":
        """
        从字符串解析拆分策略
        
        Args:
            layer_splits_str: 拆分策略字符串，如 "14,28"
            rank: 当前rank
            world_size: 总设备数
            **kwargs: 其他配置参数
            
        Returns:
            DistributedConfig实例
        """
        if layer_splits_str:
            layer_splits = [int(x.strip()) for x in layer_splits_str.split(",")]
        else:
            layer_splits = []
        
        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            layer_splits=layer_splits,
            **kwargs
        )
    
    def get_layer_range(self, num_layers: int) -> tuple:
        """
        获取当前rank负责的层范围
        
        Args:
            num_layers: 模型总层数
            
        Returns:
            (start_layer, end_layer): 负责的层范围 [start, end)
            
        Example:
            >>> config = DistributedConfig(enabled=True, rank=1, world_size=3, layer_splits=[14, 28])
            >>> start, end = config.get_layer_range(36)
            >>> print(f"Rank 1: 层 {start}-{end-1}")  # 14-27
        """
        if not self.enabled:
            return 0, num_layers
        
        if self.rank == 0:
            start = 0
            end = self.layer_splits[0]
        elif self.rank == self.world_size - 1:
            start = self.layer_splits[-1]
            end = num_layers
        else:
            start = self.layer_splits[self.rank - 1]
            end = self.layer_splits[self.rank]
        
        return start, end
    
    def is_first_rank(self) -> bool:
        """是否为第一个rank（负责embedding）"""
        return self.rank == 0
    
    def is_last_rank(self) -> bool:
        """是否为最后一个rank（负责eagle layer和lm_head）"""
        return self.rank == self.world_size - 1
    
    def get_prev_rank(self) -> Optional[int]:
        """获取上一个rank"""
        if self.rank == 0:
            return None
        return self.rank - 1
    
    def get_next_rank(self) -> Optional[int]:
        """获取下一个rank"""
        if self.rank == self.world_size - 1:
            return None
        return self.rank + 1
    
    def get_owner_rank(self, layer_idx: int, num_layers: int) -> int:
        """
        获取负责指定层的rank
        
        Args:
            layer_idx: 层索引
            num_layers: 总层数
            
        Returns:
            负责该层的rank
        """
        if not self.enabled:
            return 0
        
        for rank in range(self.world_size):
            start, end = self._get_range_for_rank(rank, num_layers)
            if start <= layer_idx < end:
                return rank
        
        return self.world_size - 1
    
    def _get_range_for_rank(self, rank: int, num_layers: int) -> tuple:
        """获取指定rank的层范围"""
        if rank == 0:
            return 0, self.layer_splits[0]
        elif rank == self.world_size - 1:
            return self.layer_splits[-1], num_layers
        else:
            return self.layer_splits[rank - 1], self.layer_splits[rank]
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "world_size": self.world_size,
            "layer_splits": self.layer_splits,
            "base_port": self.base_port,
            "comm_mode": self.comm_mode,
            "node_addresses": self.node_addresses,
            "startup_delay": self.startup_delay,
            "chunk_size": self.chunk_size,
        }
    
    def __repr__(self) -> str:
        if not self.enabled:
            return "DistributedConfig(enabled=False)"
        return (
            f"DistributedConfig("
            f"rank={self.rank}/{self.world_size-1}, "
            f"layer_splits={self.layer_splits}, "
            f"comm_mode={self.comm_mode})"
        )
