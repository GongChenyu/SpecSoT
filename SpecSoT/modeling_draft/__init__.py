# coding=utf-8
"""
Eagle Modeling Draft Module

该模块包含 Eagle 草稿模型的所有组件：
- Eagle3: EAGLE3 单层 Decoder 架构（用于 Qwen3、LLaMA 3.1 等新模型）
- Eagle2: EAGLE2 多层 Decoder Stack 架构（用于 Vicuna 等旧模型）
- Drafter: Draft Tree 生成器（独立于 Eagle Layer 的生成逻辑）
- EagleBase: Eagle Layer 基类（公共接口和共享实现）
- EConfig: Eagle 模型配置类

使用示例：
    >>> from SpecSoT.modeling_draft import Eagle3, Eagle2, Drafter
    >>> eagle_layer = Eagle3.from_pretrained(...)
    >>> drafter = Drafter(eagle_layer)
    >>> draft_tokens, retrieve_indices, tree_mask, tree_position_ids = drafter.generate_draft_tree(hidden_states, input_ids)
"""

from .configs import EConfig
from .eagle3 import Eagle3
from .eagle2 import Eagle2
from .drafter import Drafter
from .eagle_base import (
    EagleBase,
    _make_causal_mask,
    _expand_mask,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaMLP,
)

__all__ = [
    # Config
    "EConfig",
    
    # Main Classes
    "Eagle3",
    "Eagle2", 
    "Drafter",
    "EagleBase",
    
    # Utilities
    "_make_causal_mask",
    "_expand_mask",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "rotate_half",
    "LlamaRotaryEmbedding",
    "LlamaRMSNorm",
    "LlamaMLP",
]
