# coding=utf-8
"""
SpecSoT Module (v5.0 重构版本)

包含 SpecSoT 模型的核心组件：

架构层次：
- config: 配置层（设备配置、系统配置、分布式配置）
- engine: 引擎层（Master/Worker、SpecSoTGenerator、InferenceEngine）
- core: 核心组件层（Drafter、StateManager、KVCache、Evaluator、Scheduling、Distributed）
- models: 模型层（Base Model、Draft Model）
- processing: 处理层（Prompt、Logits）
- utils: 工具层

使用示例：
    >>> from SpecSoT_v2 import SpecSoTGenerator
    >>> model = SpecSoTGenerator.from_pretrained(
    ...     base_model_path="path/to/base_model",
    ...     ea_model_path="path/to/eagle_model",
    ... )
    >>> output_ids, stats = model.generate(task_prompt="...")
"""

# =====================================================================
# 配置层
# =====================================================================
from .config import (
    DeviceProfile,
    DeviceConfig,
    parse_devices,
    DistributedConfig,
    SystemConfig,
)

# =====================================================================
# 核心组件层
# =====================================================================
from .core import (
    KVCache,
    initialize_past_key_values,
    initialize_eagle_past_key_values,
    Drafter,
    BranchStateManager,
)

# =====================================================================
# 引擎层
# =====================================================================
from .engine import (
    SpecSoTGenerator,
    InferenceEngine,
    evaluate_single,
    evaluate_parallel,
)

# =====================================================================
# 模型层
# =====================================================================
from .models.draft_model import Eagle3, Eagle2, EConfig


# =====================================================================
# 处理层
# =====================================================================
from .processing import (
    SemanticLogitsProcessor,
    VocabScanner,
    prepare_skeleton_input,
    prepare_parallel_inputs,
    parse_skeleton_output,
)

# =====================================================================
# 工具层
# =====================================================================
from .utils import (
    GPUMemoryMonitor,
    set_random_seed,
    stack_with_left_padding,
)
from .utils.logging import (
    FlushHandler,
    FlushFileHandler,
    get_logger,
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
    cleanup,
    cleanup_loggers,
)

__all__ = [
    # 配置层
    "DeviceProfile",
    "DeviceConfig",
    "parse_devices",
    "DistributedConfig",
    "SystemConfig",
    
    # 核心组件层
    "KVCache",
    "initialize_past_key_values",
    "initialize_eagle_past_key_values",
    "Drafter",
    "BranchStateManager",
    "evaluate_single",
    "evaluate_parallel",
    
    # 引擎层
    "SpecSoTGenerator",
    "InferenceEngine",
    "MasterEngine",
    "WorkerEngine",
    
    # 模型层
    "Eagle3",
    "Eagle2",
    "EConfig",
    
    # 处理层
    "SemanticLogitsProcessor",
    "VocabScanner",
    "prepare_skeleton_input",
    "prepare_parallel_inputs",
    "parse_skeleton_output",
    
    # 工具层
    "GPUMemoryMonitor",
    "set_random_seed",
    "stack_with_left_padding",
    
    # 日志工具
    "FlushHandler",
    "FlushFileHandler",
    "get_logger",
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
    "cleanup",
    "cleanup_loggers",
]
