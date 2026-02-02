# coding=utf-8
"""
SpecSoT 处理层

提供 Prompt 模板和 Logits 处理：
- prompts: Prompt 模板和解析
- logits_processor: Logits 处理器
"""

from .prompts import (
    prepare_skeleton_input,
    prepare_parallel_branches,
    parse_skeleton_output,
    skeleton_prompt,
    parallel_prompt,
    system_prompt,
)
from .logits_processor import (
    SemanticLogitsProcessor,
    VocabScanner,
)

__all__ = [
    # Prompts
    "prepare_skeleton_input",
    "prepare_parallel_branches", 
    "parse_skeleton_output",
    "skeleton_prompt",
    "parallel_prompt",
    "system_prompt",
    # Logits Processor
    "SemanticLogitsProcessor",
    "VocabScanner",
]
