# coding=utf-8
"""
SpecSoT Module

包含 SpecSoT 模型的核心组件：
- SpecSoTModel: 主模型类（Speculative Decoding + Skeleton-of-Thought）
- EagleLayer: 轻量级草稿层
- SemanticLogitsProcessor: 语义约束处理器

使用示例：
    >>> from SpecSoT import SpecSoTModel
    >>> model = SpecSoTModel.from_pretrained(
    ...     base_model_path="path/to/base_model",
    ...     ea_model_path="path/to/eagle_model",
    ... )
    >>> output_ids, stats = model.generate(task_prompt="...")
"""

# 主模型
from .specsot_model import SpecSoTModel

# Eagle Layer
from .eagle_layer import EagleLayer

# Logits Processors
from .logits_processor import SemanticLogitsProcessor

# 工具函数
from .utils import (
    prepare_logits_processor,
    build_parallel_prefill_mask,
    evaluate_single,
    evaluate_parallel,
    stack_with_left_padding,
    parse_skeleton,
    prepare_parallel_branches,
    prepare_skeleton_input,
    create_skeleton_logits_processor,
    check_stop_conditions,
    check_stop_conditions_parallel,
    merge_outputs,
)

# KV Cache
from .kv_cache import initialize_past_key_values

# Configurations
from .configs import EConfig

# Distributed Support
from .distributed import (
    DistributedConfig,
    DistributedPrefillManager,
    create_zmq_comm_manager,
    ZMQCommManagerBase,
)

# Prompts
from .prompts import base_prompt, skeleton_trigger_zh, parallel_trigger_zh

__all__ = [
    # 主模型
    "SpecSoTModel",
    
    # Eagle Layer
    "EagleLayer",
    
    # Processors
    "SemanticLogitsProcessor",
    
    # Utils
    "prepare_logits_processor",
    "build_parallel_prefill_mask",
    "evaluate_single",
    "evaluate_parallel",
    "reset_tree_mode",
    "stack_with_left_padding",
    "initialize_past_key_values",
    "parse_skeleton",
    "check_stop_conditions",
    "merge_outputs",
    
    # Config
    "EConfig",
    
    # Distributed
    "DistributedConfig",
    "DistributedPrefillManager",
    "create_zmq_comm_manager",
    "ZMQCommManagerBase",
    
    # Prompts
    "base_prompt",
    "skeleton_trigger_zh",
    "parallel_trigger_zh",
]
