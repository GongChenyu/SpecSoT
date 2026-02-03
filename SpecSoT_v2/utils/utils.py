# coding=utf-8
"""
SpecSoT 工具模块

提供通用工具函数：
- Tensor 操作工具
- 设备配置解析
- 端口清理
- 随机种子设置
"""

import random
import torch
import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    """
    准备采样 logits 处理器列表（温度、top-p、top-k 等）
    
    Args:
        temperature: 采样温度 (0 表示 greedy)
        repetition_penalty: 重复惩罚系数
        top_p: nucleus sampling 阈值
        top_k: top-k sampling 数量
        
    Returns:
        LogitsProcessorList: 处理器列表
    """
    processor_list = LogitsProcessorList()
    
    if temperature > 1e-5:
        if temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
            
    return processor_list


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


def stack_with_left_padding(
    tensor_list: List[torch.Tensor],
    pad_id: int,
    device: torch.device,
    return_mask: bool = False,
) -> torch.Tensor:
    """
    将不等长的 Tensor 列表堆叠为 Batch，使用左填充
    
    支持 1D (tokens) 和 2D (hidden states) 输入。
    
    Args:
        tensor_list: Tensor 列表
        pad_id: 填充值
        device: 目标设备
        return_mask: 是否返回填充掩码
        
    Returns:
        padded_tensor: 填充后的 Tensor [batch, max_len, ...]
        padding_mask (可选): 填充掩码 [batch, max_len]
    """
    if not tensor_list:
        return None

    batch_size = len(tensor_list)
    max_len = max(t.size(0) for t in tensor_list)
    trailing_dims = list(tensor_list[0].shape[1:])
    target_shape = [batch_size, max_len] + trailing_dims

    padded_tensor = torch.full(
        target_shape, pad_id, dtype=tensor_list[0].dtype, device=device
    )

    if return_mask:
        padding_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, t in enumerate(tensor_list):
        length = t.size(0)
        start_idx = max_len - length
        padded_tensor[i, start_idx:] = t
        if return_mask:
            padding_mask[i, start_idx:] = 1

    if return_mask:
        return padded_tensor, padding_mask
    return padded_tensor


def set_random_seed(seed: int):
    """
    设置随机种子（确保可复现性）
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)






