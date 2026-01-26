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
from .eagle_layer3 import EagleLayer3
from .eagle_layer2 import EagleLayer2  # EAGLE2 支持 (用于 Vicuna 等模型)

# Logits Processors
from .logits_processor import SemanticLogitsProcessor, VocabScanner

# 工具函数
from .utils import (
    prepare_logits_processor,
    build_parallel_prefill_mask,
    evaluate_single,
    evaluate_parallel,
    stack_with_left_padding,
    prepare_skeleton_input,
    check_stop_conditions,
    check_stop_conditions_parallel,
    merge_outputs,
    set_random_seed,
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

# 日志工具
from .logging_utils import (
    FlushingStreamHandler,
    FlushingFileHandler,
    get_unified_logger,
    get_comm_logger,
    get_prefill_logger,
    get_tensor_info,
    format_tensor_brief,
    format_timing,
    format_message_info,
    log_phase_start,
    log_phase_end,
    log_progress,
    cleanup_loggers,
)

# Prompts
from .prompts import base_prompt_zh, skeleton_trigger_zh, parallel_trigger_zh
from .prompts import base_prompt_en, skeleton_trigger_en, parallel_trigger_en

__all__ = [
    # 主模型
    "SpecSoTModel",
    
    # Eagle Layer
    "EagleLayer3",
    "EagleLayer2",  # EAGLE2 支持
    
    # Processors
    "SemanticLogitsProcessor",
    "VocabScanner",
    
    # Utils
    "prepare_logits_processor",
    "build_parallel_prefill_mask",
    "evaluate_single",
    "evaluate_parallel",
    "reset_tree_mode",
    "stack_with_left_padding",
    "initialize_past_key_values",
    "parse_skeleton",
    "parse_skeleton_str",  # 字符串解析方式
    "check_stop_conditions",
    "merge_outputs",
    "set_random_seed",
    
    # Config
    "EConfig",
    
    # Distributed
    "DistributedConfig",
    "DistributedPrefillManager",
    "create_zmq_comm_manager",
    "ZMQCommManagerBase",
    
    # Logging
    "FlushingStreamHandler",
    "FlushingFileHandler",
    "get_unified_logger",
    "get_comm_logger",
    "get_prefill_logger",
    "get_tensor_info",
    "format_tensor_brief",
    "format_timing",
    "format_message_info",
    "log_phase_start",
    "log_phase_end",
    "log_progress",
    "cleanup_loggers",
    
    # Prompts
    "base_prompt_zh",
    "skeleton_trigger_zh",
    "parallel_trigger_zh",
    "base_prompt_en",
    "skeleton_trigger_en",
    "parallel_trigger_en",
]
