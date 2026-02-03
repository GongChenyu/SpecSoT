# coding=utf-8
"""
SpecSoT 核心组件层

提供推理核心组件：
- KVCache: KV Cache 封装类
- Drafter: Draft Tree 生成器
- BranchStateManager: 统一 BIM 状态管理
- InferenceEngine: 推理引擎
- eval_utils: 评估工具
"""

from .kv_cache import KVCache, initialize_past_key_values, initialize_eagle_past_key_values
from .drafter import Drafter
from .state_manager import BranchStateManager, BranchState

__all__ = [
    # KV Cache
    "KVCache",
    "initialize_past_key_values", 
    "initialize_eagle_past_key_values",
    # Drafter
    "Drafter",
    # State Manager
    "BranchStateManager",
    "AlignmentManager",
    "BranchState",
]
