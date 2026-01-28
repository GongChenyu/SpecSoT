# coding=utf-8
"""
SpecSoT Engine 工具模块

提供引擎所需的静态工具函数：
- 设备配置解析
- 端口清理
- 类型转换
"""

import subprocess
from typing import List
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """单个设备配置"""
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


def cleanup_ports(base_port: int, world_size: int, logger=None):
    """
    清理被占用的端口
    
    Args:
        base_port: 基础端口号
        world_size: 进程数量
        logger: 可选的日志记录器
    """
    ports_to_clean = set()
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                port = base_port + sender * world_size + receiver
                ports_to_clean.add(port)
    
    for port in ports_to_clean:
        try:
            subprocess.run(
                f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true",
                shell=True,
                capture_output=True,
                text=True
            )
        except Exception:
            pass


def str2bool(v) -> bool:
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
