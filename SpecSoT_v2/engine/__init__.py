# coding=utf-8
"""
SpecSoT Engine 模块

提供核心推理引擎和生成器。

使用方式：
    from SpecSoT_v2.engine import SpecSoTGenerator, InferenceEngine

    # 生成器
    generator = SpecSoTGenerator.from_pretrained(...)
    result = generator.generate(prompt)
"""

from .generator import SpecSoTGenerator
from .inference import InferenceEngine
from .evaluator import (
    evaluate_single,
    evaluate_parallel,
    greedy_sampling,
    rejection_sampling,
    logits_sampling,
)

__all__ = [
    'SpecSoTGenerator',
    'InferenceEngine',
    # Evaluator
    'evaluate_single',
    'evaluate_parallel',
    'greedy_sampling',
    'rejection_sampling',
    'logits_sampling',
]
