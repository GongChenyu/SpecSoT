# coding=utf-8
"""
SpecSoT Modeling 模块

包含各种基础模型的 KV Cache 适配实现：
- LLaMA
- Qwen2
- Qwen3
- Mixtral

以及计算指标记录：
- global_recorder: 记录 base model 的 attention/ffn 时间
"""

from .modeling_llama_kv import LlamaForCausalLM as LlamaForCausalLMKV
from .modeling_qwen2_kv import Qwen2ForCausalLM as Qwen2ForCausalLMKV
from .modeling_qwen3_kv import Qwen3ForCausalLM as Qwen3ForCausalLMKV
from .modeling_mixtral_kv import MixtralForCausalLM as MixtralForCausalLMKV
from .global_recorder import att_time_recoding, ffn_time_recoding

__all__ = [
    "LlamaForCausalLMKV",
    "Qwen2ForCausalLMKV",
    "Qwen3ForCausalLMKV",
    "MixtralForCausalLMKV",
    "att_time_recoding",
    "ffn_time_recoding",
]
