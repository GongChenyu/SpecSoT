# coding=utf-8
"""
SpecSoT 模型层

提供 Base Model 和 Draft Model：
- modeling: Base Model (LLaMA, Qwen2, Qwen3, Mixtral)
- modeling_draft: Draft Model (Eagle2, Eagle3)
"""

from .base_model import (
    LlamaForCausalLMKV as KVLlamaForCausalLM,
    Qwen2ForCausalLMKV as KVQwen2ForCausalLM,
    Qwen3ForCausalLMKV as KVQwen3ForCausalLM,
    MixtralForCausalLMKV as KVMixtralForCausalLM,
    att_time_recoding,
    ffn_time_recoding,
)

from .draft_model import (
    Eagle2,
    Eagle3,
    Drafter,
    EConfig,
    EagleBase,
)

__all__ = [
    # Base Models
    "KVLlamaForCausalLM",
    "KVQwen2ForCausalLM", 
    "KVQwen3ForCausalLM",
    "KVMixtralForCausalLM",
    # Time Recording
    "att_time_recoding",
    "ffn_time_recoding",
    # Draft Models
    "Eagle2",
    "Eagle3",
    "Drafter",
    "EConfig",
    "EagleBase",
]
