# coding=utf-8
"""
设备配置模块

提供异构设备支持的配置管理：
- DeviceProfile: 设备能力描述（算力、显存等）
- DeviceConfig: 单个设备配置
- parse_devices(): 解析设备列表
- load_device_profiles(): 从配置文件加载设备参数

使用示例：
    >>> configs = parse_devices("127.0.0.1#3,127.0.0.1#4")
    >>> print(configs[0].ip, configs[0].gpu_id)
    
    >>> profiles = load_device_profiles("devices.json")
    >>> print(profiles[0].compute_capability)
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DeviceProfile:
    """
    设备能力描述
    
    用于异构设备调度，描述每个设备的算力和显存特性。
    
    Attributes:
        device_id: 设备 ID（全局唯一）
        ip: 设备 IP 地址
        gpu_id: 本地 GPU 索引
        compute_capability: 算力指标 (TFLOPS)，用于估算计算时间
        memory_capacity: 显存容量 (GB)
        memory_bandwidth: 显存带宽 (GB/s)，用于估算访存时间
        roofline_inflection: Roofline 拐点（算术强度），用于判断计算/访存密集
        max_parallel: 最大并行分支数（受显存限制）
        metadata: 其他元数据
    """
    device_id: int
    ip: str
    gpu_id: int
    compute_capability: float = 100.0  # TFLOPS
    memory_capacity: float = 24.0  # GB
    memory_bandwidth: float = 900.0  # GB/s
    roofline_inflection: float = 100.0  # ops/byte
    max_parallel: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def estimate_compute_time(self, flops: float) -> float:
        """
        估算计算时间
        
        Args:
            flops: 浮点运算数 (FLOPs)
            
        Returns:
            估算时间 (秒)
        """
        return flops / (self.compute_capability * 1e12)
    
    def estimate_memory_time(self, bytes_accessed: float) -> float:
        """
        估算访存时间
        
        Args:
            bytes_accessed: 访问字节数
            
        Returns:
            估算时间 (秒)
        """
        return bytes_accessed / (self.memory_bandwidth * 1e9)
    
    def estimate_total_time(self, flops: float, bytes_accessed: float) -> float:
        """
        使用 Roofline 模型估算总时间
        
        Args:
            flops: 浮点运算数
            bytes_accessed: 访问字节数
            
        Returns:
            估算时间 (秒)
        """
        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else float('inf')
        
        if arithmetic_intensity < self.roofline_inflection:
            # 访存密集
            return self.estimate_memory_time(bytes_accessed)
        else:
            # 计算密集
            return self.estimate_compute_time(flops)
    
    def can_fit_cache(self, cache_size_gb: float) -> bool:
        """
        检查显存是否能容纳指定大小的 KV Cache
        
        Args:
            cache_size_gb: KV Cache 大小 (GB)
            
        Returns:
            是否能容纳
        """
        # 保留一定余量用于模型权重和激活值
        available = self.memory_capacity * 0.7
        return cache_size_gb <= available


@dataclass
class DeviceConfig:
    """
    单个设备配置（简化版，用于分布式通信）
    
    Attributes:
        ip: 设备 IP 地址
        gpu_id: GPU 索引
        rank: 分布式 rank
    """
    ip: str
    gpu_id: int
    rank: int


def parse_devices(devices_str: str) -> List[DeviceConfig]:
    """
    解析设备列表字符串
    
    Args:
        devices_str: 格式 "ip#gpu_id,ip#gpu_id,..."
                     例如: "127.0.0.1#3,127.0.0.1#4,127.0.0.1#5"
    
    Returns:
        设备配置列表
        
    Example:
        >>> configs = parse_devices("127.0.0.1#3,127.0.0.1#4")
        >>> print(configs[0].ip, configs[0].gpu_id, configs[0].rank)
        127.0.0.1 3 0
    """
    configs = []
    for i, item in enumerate(devices_str.split(",")):
        item = item.strip()
        if not item:
            continue
        parts = item.split("#")
        if len(parts) != 2:
            raise ValueError(f"设备格式错误: '{item}'，应为 'ip#gpu_id'")
        ip = parts[0].strip()
        try:
            gpu_id = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"GPU ID 必须是整数: '{parts[1]}'")
        configs.append(DeviceConfig(ip=ip, gpu_id=gpu_id, rank=i))
    return configs


def load_device_profiles(config_path: str) -> List[DeviceProfile]:
    """
    从 JSON 配置文件加载设备参数
    
    配置文件格式：
    {
        "devices": [
            {
                "device_id": 0,
                "ip": "127.0.0.1",
                "gpu_id": 0,
                "compute_capability": 150.0,
                "memory_capacity": 24.0,
                "memory_bandwidth": 900.0,
                "max_parallel": 8
            },
            ...
        ]
    }
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        设备配置列表
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"设备配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    profiles = []
    for device_data in config.get("devices", []):
        profile = DeviceProfile(
            device_id=device_data.get("device_id", len(profiles)),
            ip=device_data.get("ip", "127.0.0.1"),
            gpu_id=device_data.get("gpu_id", 0),
            compute_capability=device_data.get("compute_capability", 100.0),
            memory_capacity=device_data.get("memory_capacity", 24.0),
            memory_bandwidth=device_data.get("memory_bandwidth", 900.0),
            roofline_inflection=device_data.get("roofline_inflection", 100.0),
            max_parallel=device_data.get("max_parallel", 8),
            metadata=device_data.get("metadata", {}),
        )
        profiles.append(profile)
    
    return profiles


def create_default_profiles(devices_str: str) -> List[DeviceProfile]:
    """
    从设备字符串创建默认的设备配置（使用默认参数）
    
    Args:
        devices_str: 设备字符串，如 "127.0.0.1#0,127.0.0.1#1"
        
    Returns:
        设备配置列表
    """
    device_configs = parse_devices(devices_str)
    profiles = []
    
    for config in device_configs:
        profile = DeviceProfile(
            device_id=config.rank,
            ip=config.ip,
            gpu_id=config.gpu_id,
        )
        profiles.append(profile)
    
    return profiles


class DeviceConfigValidator:
    """设备配置验证器"""

    @staticmethod
    def validate(
        devices_str: str,
        enable_parallel: bool,
        distributed: bool,
    ) -> tuple:
        """
        验证设备配置

        根据设计文档的规则验证配置：
        1. 如果 enable_parallel=False，则 distributed 必须为 False
        2. 如果 distributed=True，设备数必须 >= 2
        3. world_size 从设备列表自动计算

        Args:
            devices_str: 设备字符串，如 "127.0.0.1#0,127.0.0.1#1"
            enable_parallel: 是否启用并行解码
            distributed: 是否启用分布式模式

        Returns:
            (is_valid, error_message): 验证结果和错误信息
        """
        try:
            device_list = parse_devices(devices_str)
            world_size = len(device_list)
        except ValueError as e:
            return False, f"设备字符串解析失败: {e}"

        if world_size == 0:
            return False, "设备列表不能为空"

        if not enable_parallel and distributed:
            return False, "enable_parallel=False 时不支持分布式模式"

        if distributed and world_size < 2:
            return False, f"分布式模式需要至少 2 个设备，当前: {world_size}"

        return True, ""

    @staticmethod
    def validate_profiles(profiles: List[DeviceProfile]) -> bool:
        """
        验证设备配置的有效性
        
        Args:
            profiles: 设备配置列表
            
        Returns:
            是否有效
            
        Raises:
            ValueError: 配置无效时抛出
        """
        if not profiles:
            raise ValueError("设备列表不能为空")
        
        # 检查 device_id 唯一性
        device_ids = [p.device_id for p in profiles]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("device_id 必须唯一")
        
        # 检查参数有效性
        for p in profiles:
            if p.compute_capability <= 0:
                raise ValueError(f"设备 {p.device_id}: compute_capability 必须为正数")
            if p.memory_capacity <= 0:
                raise ValueError(f"设备 {p.device_id}: memory_capacity 必须为正数")
            if p.memory_bandwidth <= 0:
                raise ValueError(f"设备 {p.device_id}: memory_bandwidth 必须为正数")
            if p.max_parallel <= 0:
                raise ValueError(f"设备 {p.device_id}: max_parallel 必须为正数")
        
        return True
