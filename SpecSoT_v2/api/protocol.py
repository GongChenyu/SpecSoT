# coding=utf-8
"""
API 请求/响应协议定义

定义 SpecSoT API 的数据结构：
- GenerateRequest: 生成请求
- GenerateResponse: 生成响应
- HealthResponse: 健康检查响应
- ModelInfo: 模型信息
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class GenerateMode(Enum):
    """生成模式"""
    EAGLE = "eagle"      # 纯 EAGLE 投机解码
    SPECSOT = "specsot"  # SpecSoT 骨架并行


@dataclass
class GenerateRequest:
    """
    生成请求

    Attributes:
        prompt: 用户输入的提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度 (0 表示 greedy)
        top_p: nucleus sampling 阈值
        enable_parallel: 是否启用骨架并行 (SpecSoT 模式)
        use_semantic_constraint: 是否使用 FSM 语义约束
    """
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    enable_parallel: bool = True
    use_semantic_constraint: bool = True


@dataclass
class GenerateResponse:
    """
    生成响应

    Attributes:
        text: 生成的文本
        tokens: 生成的 token 数
        time_ms: 推理耗时 (毫秒)
        mode: 使用的生成模式
        stats: 详细统计信息
    """
    text: str
    tokens: int
    time_ms: float
    mode: str = "unknown"
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResponse:
    """
    健康检查响应

    Attributes:
        status: 服务状态 ("ok" / "error")
        model_loaded: 模型是否已加载
        gpu_memory_mb: GPU 显存使用量 (MB)
    """
    status: str
    model_loaded: bool
    gpu_memory_mb: float = 0.0


@dataclass
class ModelInfo:
    """
    模型信息

    Attributes:
        base_model: Base Model 路径
        eagle_model: Eagle Model 路径
        use_eagle3: 是否使用 Eagle3
        max_seq_len: 最大序列长度
    """
    base_model: str
    eagle_model: str
    use_eagle3: bool
    max_seq_len: int
