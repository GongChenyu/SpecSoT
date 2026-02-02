# coding=utf-8
"""
SpecSoT 流水线数据结构

定义各阶段的输入输出数据结构：
- SkeletonPrefillResult: 骨架 Prefill 结果
- SkeletonDecodeResult: 骨架解码结果
- SkeletonParseResult: 骨架解析结果
- ScheduleResult: 调度结果
- ParallelPrefillResult: 并行 Prefill 结果
- ParallelDecodeResult: 并行解码结果
- GenerateResult: 最终生成结果
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch


@dataclass
class SkeletonPrefillResult:
    """
    骨架 Prefill 阶段结果
    
    Attributes:
        draft_tokens: 候选 token 树 [batch, tree_size]
        retrieve_indices: 检索索引 [batch, num_leaves, depth]
        tree_mask: 树 attention mask
        tree_position_ids: 树 position ids
        input_ids: 完整输入序列
        task_input_ids: 任务部分的 input_ids
        input_len: 输入长度
        base_prompt_len: 基础 prompt 长度
        max_kv_len: KV Cache 最大长度
        hidden_states: 隐藏状态（可选）
    """
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    input_ids: torch.Tensor
    task_input_ids: torch.Tensor
    input_len: int
    base_prompt_len: int
    max_kv_len: int
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class SkeletonDecodeResult:
    """
    骨架解码阶段结果
    
    Attributes:
        skeleton_ids: 生成的骨架 token IDs
        skeleton_text: 骨架文本
        decode_time: 解码时间 (秒)
        total_tokens: 生成的总 token 数
        accept_rate: 平均接受率
    """
    skeleton_ids: torch.Tensor
    skeleton_text: str
    decode_time: float
    total_tokens: int = 0
    accept_rate: float = 0.0


@dataclass
class SkeletonParseResult:
    """
    骨架解析阶段结果
    
    Attributes:
        mode: 解析模式 ('plan', 'direct', 'error')
        tasks: 解析出的任务列表
        clean_branches: 清理后的分支 token 列表
        instruction_len: 各分支的 instruction 长度
        error_msg: 错误信息（如果解析失败）
    """
    mode: str  # 'plan', 'direct', 'error'
    tasks: Optional[List[Dict]] = None
    clean_branches: Optional[List[List[int]]] = None
    instruction_len: Optional[Dict[int, int]] = None
    error_msg: Optional[str] = None
    
    @property
    def is_parallel(self) -> bool:
        """是否为并行模式"""
        return self.mode == 'plan' and self.tasks is not None
    
    @property
    def num_branches(self) -> int:
        """分支数量"""
        if self.tasks:
            return len(self.tasks)
        return 0


@dataclass
class ScheduleResult:
    """
    调度阶段结果
    
    Attributes:
        schedule_plan: 调度计划
        my_branch_ids: 当前设备负责的分支 ID 列表
        branch_infos: 分支信息列表
        device_assignments: 设备分配 {branch_id: device_id}
    """
    schedule_plan: Optional[Any] = None
    my_branch_ids: List[int] = field(default_factory=list)
    branch_infos: List[Any] = field(default_factory=list)
    device_assignments: Dict[int, int] = field(default_factory=dict)


@dataclass
class ParallelPrefillResult:
    """
    并行 Prefill 阶段结果
    
    Attributes:
        input_ids: 拉平的多分支输入 [1, total_len]
        draft_tokens: 候选 token 树 [num_branches, tree_size]
        retrieve_indices: 检索索引
        tree_mask: 树 attention mask
        tree_position_ids: 树 position ids
        tips_indices: 各分支 tip 位置
        branch_begins: 各分支在序列中的起始位置
        branch_lengths: 各分支长度
        branch_index_map: BIM
        position_ids: Position IDs
    """
    input_ids: torch.Tensor
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    tips_indices: torch.Tensor
    branch_begins: List[int]
    branch_lengths: List[int]
    branch_index_map: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


@dataclass
class ParallelDecodeResult:
    """
    并行解码阶段结果
    
    Attributes:
        branch_outputs: 各分支的输出 {branch_id: token_ids}
        stats: 统计信息
        total_tokens: 生成的总 token 数
        decode_time: 解码时间
    """
    branch_outputs: Dict[int, List[int]]
    stats: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    decode_time: float = 0.0


@dataclass
class GenerateResult:
    """
    最终生成结果
    
    Attributes:
        output_text: 完整输出文本
        skeleton_text: 骨架文本
        branch_outputs: 各分支输出
        total_tokens: 总 token 数
        total_time: 总时间
        stats: 详细统计
    """
    output_text: str
    skeleton_text: str = ""
    branch_outputs: Dict[int, str] = field(default_factory=dict)
    total_tokens: int = 0
    total_time: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_per_second(self) -> float:
        """每秒生成 token 数"""
        if self.total_time > 0:
            return self.total_tokens / self.total_time
        return 0.0


@dataclass
class BranchDecodeState:
    """
    单分支解码状态（用于 Continuous Batching）
    
    Attributes:
        branch_id: 分支 ID
        input_ids: 当前输入
        draft_tokens: 候选 token 树
        retrieve_indices: 检索索引
        tree_mask: 树 mask
        tree_position_ids: 树 position ids
        generated_tokens: 已生成的 tokens
        is_finished: 是否完成
        finish_reason: 完成原因
    """
    branch_id: int
    input_ids: torch.Tensor
    draft_tokens: Optional[torch.Tensor] = None
    retrieve_indices: Optional[torch.Tensor] = None
    tree_mask: Optional[torch.Tensor] = None
    tree_position_ids: Optional[torch.Tensor] = None
    generated_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = ""
    
    @property
    def num_generated(self) -> int:
        """已生成 token 数"""
        return len(self.generated_tokens)


@dataclass
class ContinuousDecodingState:
    """
    Continuous Decoding 状态（Batching 模式）
    
    Attributes:
        cache_length: 统一的 cache 物理长度
        valid_lengths: 各分支的有效长度
        input_lengths: 本次输入长度
        is_prefill: 各分支是否为 prefill
        padding_mask: 输入的 padding mask
        position_ids: 各分支独立的 position ids
    """
    cache_length: int
    valid_lengths: List[int]
    input_lengths: List[int]
    is_prefill: List[bool]
    padding_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
