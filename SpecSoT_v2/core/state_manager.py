# coding=utf-8
"""
统一 BIM 状态管理器

实现双模式推理的统一状态管理：
- BIM 模式 (In-One-Sequence): batch_size=1，通过 BIM 索引管理多分支
- Batching 模式: batch_size=num_branches，标准批量推理

核心类：
- BranchState: 单分支状态
- BranchStateManager: 统一状态管理器
- AlignmentManager: Batching 模式对齐管理器

设计原则：
1. 预分配固定内存，避免运行时分配
2. 原地操作，避免不必要的内存复制
3. 统一接口，支持双模式切换
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import torch
import torch.nn as nn
from .kv_cache import initialize_past_key_values, initialize_eagle_past_key_values, KVCache
from ..utils.utils import stack_with_left_padding


class InferenceMode(Enum):
    """推理模式"""
    BIM = "bim"           # In-One-Sequence 模式
    BATCHING = "batching"  # 批量推理模式


@dataclass
class BranchState:
    """
    单分支状态
    
    Attributes:
        branch_id: 分支 ID
        start_pos: 在序列中的起始位置（BIM 模式）
        current_len: 当前已生成长度
        cache_len: KV Cache 长度
        is_active: 是否活跃
        is_finished: 是否完成
        output_ids: 已生成的 token IDs
    """
    branch_id: int
    start_pos: int = 0
    current_len: int = 0
    cache_len: int = 0
    is_active: bool = True
    is_finished: bool = False
    output_ids: List[int] = field(default_factory=list)
    
    def get_tip_pos(self) -> int:
        """获取分支 tip 位置"""
        return self.start_pos + self.current_len - 1
    
    def get_next_pos(self) -> int:
        """获取下一个写入位置"""
        return self.start_pos + self.current_len


class BranchStateManager:
    """
    统一 BIM 状态管理器
    
    管理 BIM、Position IDs、分支状态等，支持双模式切换。
    
    BIM (Branch Index Map) 含义：
    - -2: 空位置（未使用）
    - -1: 共享 prefix
    - >= 0: 分支 ID
    
    Attributes:
        use_bim_mode: 是否使用 BIM 模式
        max_seq_len: 最大序列长度
        max_branches: 最大分支数
        device: 设备
        dtype: 数据类型
    """
    
    # BIM 特殊值
    BIM_EMPTY = -2
    BIM_PREFIX = -1
    
    def __init__(
        self,
        max_seq_len: int = 4096,
        max_branches: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        use_bim_mode: bool = True,
    ):
        """
        初始化状态管理器
        
        Args:
            max_seq_len: 最大序列长度
            max_branches: 最大分支数
            device: 设备
            dtype: 数据类型
            use_bim_mode: 是否使用 BIM 模式
        """
        self.max_seq_len = max_seq_len
        self.max_branches = max_branches
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.use_bim_mode = use_bim_mode
        
        # 模式
        self.mode = InferenceMode.BIM if use_bim_mode else InferenceMode.BATCHING

        # ===== 通用状态 =====
        # 分支状态字典
        self.branch_states: Dict[int, BranchState] = {}
        # 活跃分支列表
        self.active_branches: List[int] = []
        # 共享 prefix 长度
        self.prefix_len: int = 0

        
        if self.use_bim_mode:
            # ===== BIM 模式状态 =====
            # BIM: Branch Index Map
            self.bim = torch.full(self.max_seq_len, self.BIM_EMPTY, dtype=torch.long, device=self.device)
            # Position IDs
            self.position_ids = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.device)
            # 当前总序列长度（BIM 模式）
            self.current_seq_len: int = 0
        else:
            # ===== Batching 模式状态 =====
            # Batching 模式下的 padding mask [num_branches, max_len]，1=有效，0=padding
            self.padding_mask: Optional[torch.Tensor] = None
            # Batching 模式下的 position_ids [num_branches, max_len]
            self.batching_position_ids: Optional[torch.Tensor] = None
            # Batching 模式下各分支长度列表
            self.branch_lengths: List[int] = []
    
    def reset(self):
        """重置所有状态"""
        self.branch_states.clear()
        self.active_branches.clear()
        self.prefix_len = 0
        self.current_seq_len = 0
        
        if self.use_bim_mode:
            self.bim.fill_(self.BIM_EMPTY)
            self.position_ids.fill_(0)
        else:
            # Batching 模式状态重置
            self.padding_mask = None
            self.batching_position_ids = None
            self.branch_lengths = []

    # =========================================================================
    # Skeleton -> Parallel 阶段转换
    # =========================================================================

    def init_parallel_state(
        self,
        base_model: nn.Module,
        eagle_layer: nn.Module,
        prefix_len: int,
        num_branches: int,
        max_new_tokens: int,
        old_past_key_values: List,
        old_past_key_values_data: List[torch.Tensor],
        old_current_length_data: torch.Tensor,
    ) -> Tuple[List, List, torch.Tensor]:
        """
        初始化并行阶段状态（Skeleton -> Parallel 转换）
        
        同时处理 Base Model 和 Eagle Layer 的 KV Cache：
        - Base Model: 根据模式复用/扩展 KV cache
        - Eagle Layer: 复用 prefix cache 并扩展到各分支
        
        根据不同模式执行不同的初始化：
        - BIM 模式: 重置 cache 长度到 prefix_len，复用原有内存
        - Batching 模式: 创建新的 batch KV cache，复制 prefix cache
        
        Args:
            base_model: Base Model（用于获取配置）
            eagle_layer: Eagle Layer（用于处理 draft cache）
            prefix_len: 共享 prefix 长度
            num_branches: 分支数量
            max_new_tokens: 最大生成 token 数
            old_past_key_values: 原有的 KV cache（skeleton 阶段的）
            old_past_key_values_data: 原有的 KV cache 数据列表（连续内存块）
            old_current_length_data: 原有的 current_length_data
            
        Returns:
            past_key_values: 新的 KV cache 列表
            past_key_values_data: 新的 KV cache 数据列表
            current_length_data: 新的 current_length_data
        """
        # 重置状态管理器
        self.reset()
        self.prefix_len = prefix_len
        self.current_seq_len = prefix_len
        
        max_kv_len = old_past_key_values_data[0].shape[3]  # seq_len 维度
        
        if self.use_bim_mode:
            # =================================================================
            # BIM 模式：复用原有内存，只重置长度
            # =================================================================
            
            # 1. Base Model Cache: 重置长度
            old_current_length_data.fill_(prefix_len)
            
            # 2. Eagle Layer Cache: 重置长度
            eagle_layer.draft_current_length_data.fill_(prefix_len)
            
            # 3. 设置 BIM 状态
            self.bim[:prefix_len] = self.BIM_PREFIX
            self.position_ids[:prefix_len] = torch.arange(prefix_len, device=self.device)
            
            return old_past_key_values, old_past_key_values_data, old_current_length_data
        
        else:
            # =================================================================
            # Batching 模式：创建新 cache，复制 prefix 到各分支
            # =================================================================
            
            # -----------------------------------------------------------------
            # 1. Base Model Cache: 创建新 cache 并复制 prefix
            # -----------------------------------------------------------------
            new_past_key_values, new_data_list, new_length_data = initialize_past_key_values(
                model=base_model, max_length=max_kv_len, batch_size=num_branches
            )
            
            # 直接操作连续内存块：复制 prefix 并扩展到 num_branches
            # old_data 形状: [num_layers*2, 1, num_heads, seq_len, head_dim]
            # new_data 形状: [num_layers*2, num_branches, num_heads, seq_len, head_dim]
            for old_data, new_data in zip(old_past_key_values_data, new_data_list):
                prefix_data = old_data[:, :, :, :prefix_len, :]
                new_data[:, :, :, :prefix_len, :] = prefix_data.expand(-1, num_branches, -1, -1, -1)
            
            new_length_data.fill_(prefix_len)
            
            # -----------------------------------------------------------------
            # 2. Eagle Layer Cache: 创建新 cache 并复制 prefix
            # -----------------------------------------------------------------
            old_draft_data = eagle_layer.draft_past_key_values_data  # 保存引用
            
            # 使用统一函数初始化新的 KV Cache
            new_draft_kv, new_draft_data, new_draft_length = initialize_eagle_past_key_values(
                eagle_layer, max_length=max_kv_len, batch_size=num_branches
            )
            
            # 直接操作连续内存块（与 Base Model 一致）
            # old_draft_data 形状: [num_layers*2, 1, num_heads, seq_len, head_dim] 或 List
            # new_draft_data 形状: [num_layers*2, num_branches, num_heads, seq_len, head_dim] 或 List
            if isinstance(old_draft_data, list):
                for old_data, new_data in zip(old_draft_data, new_draft_data):
                    prefix_data = old_data[:, :, :, :prefix_len, :]
                    new_data[:, :, :, :prefix_len, :] = prefix_data.expand(-1, num_branches, -1, -1, -1)
            else:
                # 单个 tensor 的情况
                prefix_data = old_draft_data[:, :, :, :prefix_len, :]
                new_draft_data[:, :, :, :prefix_len, :] = prefix_data.expand(-1, num_branches, -1, -1, -1)
            
            new_draft_length.fill_(prefix_len)
            
            # 设置新的 KV Cache 到 eagle_layer
            eagle_layer.reset_state()
            eagle_layer.draft_past_key_values = new_draft_kv
            eagle_layer.draft_past_key_values_data = new_draft_data
            eagle_layer.draft_current_length_data = new_draft_length
            
            return new_past_key_values, new_data_list, new_length_data
 
    
    # =========================================================================
    # BIM 模式接口
    # =========================================================================
    
    def add_branch(
        self, 
        branch_id: int, 
        branch_len: int,
        start_position: Optional[int] = None,
    ) -> BranchState:
        """
        添加新分支
        
        Args:
            branch_id: 分支 ID
            branch_len: 分支 prompt 长度
            start_position: 分支起始位置（BIM 模式）
            
        Returns:
            新创建的分支状态
        """
        if branch_id in self.branch_states:
            raise ValueError(f"分支 {branch_id} 已存在")
        
        if self.use_bim_mode:
            # BIM 模式：在序列末尾追加
            start_pos = start_position if start_position is not None else self.current_seq_len
            end_pos = start_pos + branch_len
            
            if end_pos > self.max_seq_len:
                raise RuntimeError(f"序列长度超出限制: {end_pos} > {self.max_seq_len}")
            
            # 设置 BIM
            self.bim[start_pos:end_pos] = branch_id
            
            # 设置 Position IDs（从 prefix_len 开始）
            self.position_ids[start_pos:end_pos] = torch.arange(
                self.prefix_len, 
                self.prefix_len + branch_len, 
                device=self.device
            )
            
            # 更新当前序列长度
            self.current_seq_len = max(self.current_seq_len, end_pos)
            
            # 创建分支状态
            state = BranchState(
                branch_id=branch_id,
                start_pos=start_pos,
                current_len=branch_len,
                cache_len=self.prefix_len + branch_len,
            )
        else:
            # Batching 模式
            state = BranchState(
                branch_id=branch_id,
                start_pos=0,  # Batching 模式不使用
                current_len=branch_len,
                cache_len=self.prefix_len + branch_len,
            )
        
        self.branch_states[branch_id] = state
        self.active_branches.append(branch_id)
        
        return state
    
    def extend_branch(self, branch_id: int, num_tokens: int):
        """
        扩展分支（添加新生成的 tokens）
        
        Args:
            branch_id: 分支 ID
            num_tokens: 新增 token 数
        """
        if branch_id not in self.branch_states:
            raise ValueError(f"分支 {branch_id} 不存在")
        
        state = self.branch_states[branch_id]
        
        if self.use_bim_mode:
            # 更新 BIM 和 Position
            old_end = state.start_pos + state.current_len
            new_end = old_end + num_tokens
            
            if new_end > self.max_seq_len:
                raise RuntimeError(f"序列长度超出限制: {new_end} > {self.max_seq_len}")
            
            self.bim[old_end:new_end] = branch_id
            
            # Position 继续递增
            for i in range(num_tokens):
                self.position_ids[old_end + i] = self.prefix_len + state.current_len + i
            
            self.current_seq_len = max(self.current_seq_len, new_end)
        
        # 更新状态
        state.current_len += num_tokens
        state.cache_len += num_tokens
    
    def finish_branch(self, branch_id: int):
        """标记分支完成"""
        if branch_id in self.branch_states:
            self.branch_states[branch_id].is_finished = True
            self.branch_states[branch_id].is_active = False
            if branch_id in self.active_branches:
                self.active_branches.remove(branch_id)

    def update_after_verify(
        self,
        branch_id: int,
        accept_length: int,
        accepted_tokens: torch.Tensor,
    ):
        """
        Verify 后更新分支状态

        在投机解码的 verify 阶段后调用，更新分支的状态和输出。

        Args:
            branch_id: 分支 ID
            accept_length: 接受的 token 数
            accepted_tokens: 接受的 token IDs [accept_length]
        """
        if branch_id not in self.branch_states:
            raise ValueError(f"分支 {branch_id} 不存在")

        state = self.branch_states[branch_id]

        if accept_length > 0:
            # 更新输出 token 列表
            if isinstance(accepted_tokens, torch.Tensor):
                tokens_list = accepted_tokens[:accept_length].tolist()
            else:
                tokens_list = list(accepted_tokens[:accept_length])
            state.output_ids.extend(tokens_list)

            # 扩展分支
            self.extend_branch(branch_id, accept_length)

    def flatten_branches_to_sequence(
        self,
        tip_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将所有活跃分支的 tip tokens 拉平成单序列

        用于 BIM 模式下的 Draft Tree expand_root 阶段。

        Args:
            tip_token_ids: 各分支 tip 的 token IDs [num_branches]
                          如果为 None，需要外部提供

        Returns:
            input_ids: [1, num_branches] 拉平的 token 序列
            bim: [num_branches] 各 token 对应的分支 ID
            position_ids: [1, num_branches] 各 token 的 position
        """
        num_branches = len(self.active_branches)

        if num_branches == 0:
            raise ValueError("没有活跃分支")

        # 构造 BIM: 各 token 对应的分支 ID
        bim = torch.tensor(
            self.active_branches,
            dtype=torch.long,
            device=self.device
        )

        # 构造 Position IDs: 各分支当前的 position
        positions = []
        for branch_id in self.active_branches:
            state = self.branch_states[branch_id]
            # position = prefix_len + current_len (下一个位置)
            positions.append(self.prefix_len + state.current_len)

        position_ids = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, num_branches]

        # 如果提供了 tip_token_ids，直接使用
        if tip_token_ids is not None:
            input_ids = tip_token_ids.view(1, -1)  # [1, num_branches]
        else:
            # 返回 None，由调用方提供
            input_ids = None

        return input_ids, bim, position_ids

    def get_branch_tip_positions(self) -> Dict[int, int]:
        """获取所有活跃分支的 tip 位置"""
        tips = {}
        for branch_id in self.active_branches:
            state = self.branch_states[branch_id]
            tips[branch_id] = state.get_tip_pos()
        return tips
    
    def get_branch_tips_tensor(self) -> torch.Tensor:
        """获取所有活跃分支的 tip 位置（tensor 形式）"""
        tips = []
        for branch_id in self.active_branches:
            state = self.branch_states[branch_id]
            tips.append(state.get_tip_pos())
        return torch.tensor(tips, dtype=torch.long, device=self.device)

    # =========================================================================
    # 双模式统一接口
    # =========================================================================

    def flatten_branches(
        self,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        BIM 模式：将多分支拉平成单序列

        Args:
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表

        Returns:
            flat_tokens: [1, total_len] 拉平的 token 序列
            branch_index_map: [total_capacity] BIM 索引
            position_ids: [1, total_len] 位置编码
            branch_begins: 各分支起始位置列表
            branch_lengths: 各分支长度列表
        """
        if not self.use_bim_mode:
            raise ValueError("flatten_branches 仅用于 BIM 模式")

        flat_tokens_list = []
        branch_begins = []
        branch_lengths = []
        pos_ids_list = []

        current_offset = 0
        for branch_id, prompt in zip(branch_ids, branch_prompts):
            curr_len = len(prompt)
            branch_begins.append(current_offset)
            flat_tokens_list.extend(prompt)
            branch_lengths.append(curr_len)

            # 更新 BIM
            start_pos = self.prefix_len + current_offset
            end_pos = start_pos + curr_len
            self.bim[start_pos:end_pos] = branch_id

            # Position IDs: 每个分支从 prefix_len 开始独立计数
            curr_pos = list(range(self.prefix_len, self.prefix_len + curr_len))
            pos_ids_list.extend(curr_pos)

            # 更新 Position IDs
            self.position_ids[start_pos:end_pos] = torch.tensor(
                curr_pos, device=self.device, dtype=torch.long
            )

            current_offset += curr_len

        # 更新序列长度
        self.current_seq_len = self.prefix_len + current_offset

        # 构建输出 Tensor
        flat_tokens = torch.tensor(
            [flat_tokens_list], device=self.device, dtype=torch.long
        )
        position_ids = torch.tensor(
            pos_ids_list, device=self.device, dtype=torch.long
        ).unsqueeze(0)

        return flat_tokens, self.bim, position_ids, branch_begins, branch_lengths

    def align_branches(
        self,
        branch_prompts: List[List[int]],
        pad_token_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batching 模式：左填充对齐各分支

        Args:
            branch_prompts: 各分支的 prompt token 列表
            pad_token_id: padding token ID

        Returns:
            padded_input: [num_branches, max_len] 对齐后的输入
            position_ids: [num_branches, max_len] 位置编码
            padding_mask: [num_branches, max_len] padding 掩码 (1=有效, 0=padding)
        """
        if self.use_bim_mode:
            raise ValueError("align_branches 仅用于 Batching 模式")

        branch_lengths = [len(p) for p in branch_prompts]

        # 左填充对齐（使用工具函数）
        prompt_tensors = [torch.tensor(p, dtype=torch.long, device=self.device) for p in branch_prompts]
        padded_input, padding_mask = stack_with_left_padding(
            prompt_tensors, pad_token_id, self.device, return_mask=True
        )

        # position_ids 需要单独处理（每个分支从 prefix_len 开始）
        position_tensors = [
            torch.arange(self.prefix_len, self.prefix_len + len(p), dtype=torch.long, device=self.device)
            for p in branch_prompts
        ]
        position_ids = stack_with_left_padding(position_tensors, 0, self.device)

        # 保存到状态管理器
        self.padding_mask = padding_mask
        self.batching_position_ids = position_ids
        self.branch_lengths = branch_lengths

        return padded_input, position_ids, padding_mask

    # =========================================================================
    # 并行 Prefill 准备方法
    # =========================================================================

    def prepare_parallel_prefill_bim(
        self,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        BIM 模式：准备并行 Prefill 所需的所有数据

        包括：
        1. 重置 KV Cache 长度到 prefix_len（外部执行）
        2. 构建拉平序列和 BIM
        3. 构建 position_ids
        4. 构建 attention_mask
        5. 计算 tips_indices

        Args:
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            max_new_tokens: 最大生成 token 数（用于 BIM 容量计算）

        Returns:
            Dict 包含:
                - input_ids: [1, branches_total_len] 拉平的分支 tokens
                - branch_index_map: [total_capacity] BIM 索引
                - position_ids: [1, branches_total_len] 位置编码
                - attention_mask: [1, 1, branches_len, prefix_len + branches_len]
                - tips_indices: [num_branches] 各分支 tip 在拉平序列中的位置
                - branch_begins: 各分支在拉平序列中的起始位置
                - branch_lengths: 各分支长度
        """
        if not self.use_bim_mode:
            raise ValueError("prepare_parallel_prefill_bim 仅用于 BIM 模式")

        # 1. 复用 flatten_branches 构建拉平序列和 BIM
        input_ids, _, position_ids, branch_begins, branch_lengths = self.flatten_branches(
            branch_prompts, branch_ids
        )

        # 2. 计算 tips_indices（各分支 tip 在拉平序列中的位置）
        tips_indices = torch.tensor(
            [begin + length - 1 for begin, length in zip(branch_begins, branch_lengths)],
            device=self.device
        )

        # 3. 构建扩展容量的 BIM（预留 max_new_tokens 空间）
        branches_len = sum(branch_lengths)
        total_capacity = self.prefix_len + branches_len + max_new_tokens + 128
        branch_index_map = torch.full(
            (total_capacity,), self.BIM_EMPTY, dtype=torch.long, device=self.device
        )
        # 复制已有的 BIM 数据
        branch_index_map[:self.current_seq_len] = self.bim[:self.current_seq_len]

        # 4. 构建 attention mask
        attention_mask = self.build_bim_prefill_mask(
            branch_index_map, self.prefix_len, branches_len
        )

        # 5. 注册分支状态
        for branch_id, begin, length in zip(branch_ids, branch_begins, branch_lengths):
            state = BranchState(
                branch_id=branch_id,
                start_pos=self.prefix_len + begin,
                current_len=length,
                cache_len=self.prefix_len + length,
            )
            self.branch_states[branch_id] = state
            self.active_branches.append(branch_id)

        return {
            'input_ids': input_ids,
            'branch_index_map': branch_index_map,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'tips_indices': tips_indices,
            'branch_begins': branch_begins,
            'branch_lengths': branch_lengths,
        }

    def prepare_parallel_prefill_batching(
        self,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        pad_token_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Batching 模式：准备并行 Prefill 所需的所有数据

        包括：
        1. 左填充对齐各分支
        2. 构建 position_ids
        3. 构建 padding_mask
        4. 构建 attention_mask (causal + padding)

        注意：KV Cache 的扩展（复制 prefix 到各分支）需要外部执行

        Args:
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            pad_token_id: padding token ID

        Returns:
            Dict 包含:
                - input_ids: [num_branches, max_len] 对齐后的输入
                - position_ids: [num_branches, max_len] 位置编码
                - padding_mask: [num_branches, max_len] padding 掩码 (1=有效, 0=padding)
                - attention_mask: [num_branches, 1, max_len, prefix_len + max_len]
                - branch_lengths: 各分支长度
        """
        if self.use_bim_mode:
            raise ValueError("prepare_parallel_prefill_batching 仅用于 Batching 模式")

        # 1. 左填充对齐（复用 align_branches）
        padded_input, position_ids, padding_mask = self.align_branches(branch_prompts, pad_token_id)
        
        max_branch_len = padded_input.shape[1]
        branch_lengths = self.branch_lengths  # align_branches 已设置

        # 2. 构建 attention mask
        attention_mask = self.build_batching_prefill_mask(
            padding_mask, branch_lengths, max_branch_len
        )

        # 3. 注册分支状态
        for branch_id, length in zip(branch_ids, branch_lengths):
            state = BranchState(
                branch_id=branch_id,
                start_pos=0,
                current_len=length,
                cache_len=self.prefix_len + length,
            )
            self.branch_states[branch_id] = state
            self.active_branches.append(branch_id)

        return {
            'input_ids': padded_input,
            'position_ids': position_ids,
            'padding_mask': padding_mask,
            'attention_mask': attention_mask,
            'branch_lengths': branch_lengths,
        }

    def build_bim_prefill_mask(
        self,
        branch_index_map: torch.Tensor,
        prefix_len: int,
        branches_len: int,
    ) -> torch.Tensor:
        """
        构建 BIM 模式并行 Prefill 的 attention mask

        确保：
        1. 所有分支都能看到共享的 prefix
        2. 每个分支只能看到自己的内容
        3. 遵循因果约束

        Args:
            branch_index_map: BIM 索引 [total_capacity]
            prefix_len: prefix 长度
            branches_len: 分支总长度

        Returns:
            attention_mask: [1, 1, branches_len, prefix_len + branches_len]
        """
        total_len = prefix_len + branches_len

        total_ids = branch_index_map[:total_len]
        branch_ids = branch_index_map[prefix_len:total_len]

        # 初始化为全部遮蔽
        mask = torch.full(
            (1, 1, branches_len, total_len),
            torch.finfo(self.dtype).min, device=self.device
        )

        # 1. Prefix 全部可见 (BIM == -1)
        is_prefix = (total_ids == self.BIM_PREFIX).unsqueeze(0)
        mask.masked_fill_(is_prefix, 0)

        # 2. 同分支可见 + 因果约束
        branch_ids_view = branch_ids.unsqueeze(1)  # [branches_len, 1]
        total_ids_view = total_ids.unsqueeze(0)    # [1, total_len]
        block_mask = (branch_ids_view == total_ids_view)

        branch_idx = torch.arange(prefix_len, total_len, device=self.device).unsqueeze(1)
        total_idx = torch.arange(total_len, device=self.device).unsqueeze(0)
        causal_mask = (total_idx <= branch_idx)

        valid_mask = block_mask & causal_mask
        mask.masked_fill_(valid_mask, 0)

        return mask

    def build_batching_prefill_mask(
        self,
        padding_mask: torch.Tensor,
        branch_lengths: List[int],
        max_branch_len: int,
    ) -> torch.Tensor:
        """
        构建 Batching 模式并行 Prefill 的 attention mask

        确保：
        1. 可以看到 prefix（KV Cache）
        2. 可以看到自己的有效 tokens（非 padding）
        3. 遵循因果约束

        Args:
            padding_mask: [num_branches, max_branch_len] padding 掩码 (1=有效, 0=padding)
            branch_lengths: 各分支长度
            max_branch_len: 最大分支长度

        Returns:
            attention_mask: [num_branches, 1, max_branch_len, prefix_len + max_branch_len]
        """
        num_branches = len(branch_lengths)
        total_key_len = self.prefix_len + max_branch_len

        mask = torch.full(
            (num_branches, 1, max_branch_len, total_key_len),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )

        for i, length in enumerate(branch_lengths):
            pad_len = max_branch_len - length

            # 可以看到 prefix（KV Cache 中的部分）
            mask[i, :, :, :self.prefix_len] = 0

            # 可以看到自己分支的有效 tokens（因果）
            for q in range(max_branch_len):
                if padding_mask[i, q] == 1:  # 有效位置
                    # 可以看到 prefix_len 到 prefix_len + q 的位置（只看有效部分）
                    valid_start = self.prefix_len + pad_len
                    valid_end = self.prefix_len + q + 1
                    mask[i, :, q, valid_start:valid_end] = 0

        return mask
    
    # =========================================================================
    # Attention Mask 构建
    # =========================================================================
    
    def build_attention_mask(
        self,
        query_len: int,
        key_len: Optional[int] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        构建 attention mask
        
        Args:
            query_len: query 长度
            key_len: key 长度（包含 KV Cache）
            batch_size: batch 大小
            
        Returns:
            attention_mask: [batch_size, 1, query_len, key_len]
        """
        if self.use_bim_mode:
            return self._build_bim_attention_mask(query_len, key_len)
        else:
            return self._build_batching_attention_mask(query_len, key_len, batch_size)
    
    def _build_bim_attention_mask(
        self,
        query_len: int,
        key_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        构建 BIM 模式的 attention mask
        
        规则：
        1. Prefix (BIM == -1) 对所有位置可见
        2. 同分支可见 (BIM 相同)
        3. 遵循因果约束（当前位置只能看之前的位置）
        """
        if key_len is None:
            key_len = self.current_seq_len
        
        # 获取 BIM
        bim = self.bim[:key_len]
        
        # query 对应的 BIM（通常是序列末尾的 query_len 个位置）
        query_start = key_len - query_len
        query_bim = bim[query_start:]
        
        # 初始化为全部遮蔽
        mask = torch.full(
            (1, 1, query_len, key_len),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )
        
        # 1. Prefix 全部可见
        is_prefix = (bim == self.BIM_PREFIX).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = mask.masked_fill(is_prefix, 0)
        
        # 2. 同分支可见
        query_bim_view = query_bim.view(1, 1, query_len, 1)
        key_bim_view = bim.view(1, 1, 1, key_len)
        is_same_branch = (query_bim_view == key_bim_view) & (query_bim_view >= 0)
        
        # 3. 因果约束
        query_idx = torch.arange(query_start, key_len, device=self.device).view(1, 1, query_len, 1)
        key_idx = torch.arange(key_len, device=self.device).view(1, 1, 1, key_len)
        causal_mask = (key_idx <= query_idx)
        
        # 组合
        valid_mask = is_same_branch & causal_mask
        mask = mask.masked_fill(valid_mask, 0)
        
        return mask
    
    def _build_batching_attention_mask(
        self,
        query_len: int,
        key_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        构建 Batching 模式的 attention mask
        
        标准因果掩码 + padding 掩码
        """
        # 基础因果掩码
        mask = torch.full(
            (batch_size, 1, query_len, key_len),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )
        
        # 因果约束
        query_idx = torch.arange(query_len, device=self.device).view(1, 1, query_len, 1)
        key_idx = torch.arange(key_len, device=self.device).view(1, 1, 1, key_len)
        
        # 假设 query 从 key_len - query_len 开始
        query_offset = key_len - query_len
        causal_mask = (key_idx <= query_idx + query_offset)
        mask = mask.masked_fill(causal_mask, 0)
        
        # padding 掩码（使用 padding_mask 而不是 padding_lengths）
        if self.padding_mask is not None and self.padding_mask.shape[0] >= batch_size:
            # padding_mask: [num_branches, seq_len], 1=有效, 0=padding
            # 需要将 padding 位置在 attention mask 中遮蔽
            for i in range(batch_size):
                # 找到 padding 结束的位置
                valid_positions = self.padding_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    first_valid = valid_positions[0].item()
                    if first_valid > 0:
                        # 遮蔽 KV cache 中对应的 padding 部分
                        # key 中的 padding 位置 = prefix_len + [0, first_valid)
                        pad_start = self.prefix_len
                        pad_end = self.prefix_len + first_valid
                        mask[i, :, :, pad_start:pad_end] = torch.finfo(self.dtype).min
        
        return mask
    
    # =========================================================================
    # Continuous Batching (混合 Prefill + Decode) Mask 构建
    # =========================================================================

    def build_continuous_decode_mask(
        self,
        history_bim: torch.Tensor,
        combined_bim_tensor: torch.Tensor,
        current_length: int,
        num_old_branches: int,
        num_nodes: int,
        tree_mask: Optional[torch.Tensor],
        new_prompt_lengths: List[int],
    ) -> torch.Tensor:
        """
        构建 Continuous Batching 并行解码阶段的混合 Attention Mask
        
        用于处理 PD (Parallel Decoding) 混合状态，即同时包含：
        - 老分支的 draft tokens（使用 tree_mask）
        - 新分支的 prompt tokens（使用 causal mask）
        
        该掩码确保：
        1. 所有分支都能看到共享的 prefix (BIM == -1)
        2. 每个分支只能看到自己的历史和当前输入
        3. 老分支内部遵循 tree_mask
        4. 新分支内部遵循 causal mask
        
        Args:
            history_bim: 历史 BIM (Branch Index Map) [current_length]
            combined_bim_tensor: 当前输入的 BIM [total_input_len]
            current_length: 历史 KV Cache 的长度
            num_old_branches: 老分支数量
            num_nodes: 每个老分支的 draft tokens 数量
            tree_mask: 老分支的 tree mask [num_old, 1, num_nodes, num_nodes] (可选)
            new_prompt_lengths: 新分支的 prompt 长度列表
            
        Returns:
            combined_mask: [1, 1, total_input_len, current_length + total_input_len]
        """
        total_input_len = combined_bim_tensor.shape[0]
        
        # =====================================================================
        # 1. 构建 Cross Mask (输入 -> 历史)
        # =====================================================================
        cross_mask = torch.full(
            (1, 1, total_input_len, current_length),
            torch.finfo(self.dtype).min, device=self.device
        )
        
        # Prefix 全部可见 (BIM == -1)
        is_prefix = (history_bim == self.BIM_PREFIX).view(1, 1, 1, -1)
        cross_mask.masked_fill_(is_prefix, 0)
        
        # 同分支可见
        input_ids_view = combined_bim_tensor.view(1, 1, -1, 1)
        hist_ids_view = history_bim.view(1, 1, 1, -1)
        is_same_branch = (input_ids_view == hist_ids_view)
        cross_mask.masked_fill_(is_same_branch, 0)
        
        # =====================================================================
        # 2. 构建 Input Block Mask (输入 -> 输入)
        # =====================================================================
        input_block_mask = torch.full(
            (total_input_len, total_input_len),
            torch.finfo(self.dtype).min, device=self.device
        )
        
        # 老分支的 tree mask（块对角结构）
        if num_old_branches > 0 and tree_mask is not None:
            converted_tree_mask = torch.where(
                tree_mask == 1, 0.0, torch.finfo(self.dtype).min
            )
            for i in range(num_old_branches):
                st, ed = i * num_nodes, (i + 1) * num_nodes
                input_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :]
        
        # 新分支的 causal mask
        if len(new_prompt_lengths) > 0:
            old_total_len = num_old_branches * num_nodes if (num_old_branches > 0 and tree_mask is not None) else 0
            
            # 每个新分支内部使用 causal mask
            offset = old_total_len
            for prompt_len in new_prompt_lengths:
                for j in range(prompt_len):
                    for k in range(j + 1):
                        input_block_mask[offset + j, offset + k] = 0
                offset += prompt_len
        
        input_block_mask = input_block_mask.unsqueeze(0).unsqueeze(0)
        
        # =====================================================================
        # 3. 合并 Cross Mask 和 Input Block Mask
        # =====================================================================
        combined_mask = torch.cat([cross_mask, input_block_mask], dim=-1)
        
        return combined_mask

    def prepare_continuous_decode_inputs(
        self,
        old_branch_ids: List[int],
        new_branch_ids: List[int],
        new_prompt_tokens: Dict[int, List[int]],
        prefix_len: int,
    ) -> Dict[str, Any]:
        """
        准备 Continuous Batching 的输入数据
        
        同时处理：
        - 老分支的 decode 输入
        - 新分支的 prefill 输入
        
        Args:
            old_branch_ids: 老分支 ID 列表
            new_branch_ids: 新分支 ID 列表
            new_prompt_tokens: 新分支的 prompt tokens {branch_id: [token_ids]}
            prefix_len: 共享 prefix 长度
            
        Returns:
            Dict 包含:
                - new_prompt_lengths: 新分支 prompt 长度列表
                - new_branch_bim: 新分支的 BIM 列表
                - combined_bim: 整体 BIM tensor
        """
        new_prompt_lengths = []
        new_branch_bim = []
        
        for bid in new_branch_ids:
            prompt = new_prompt_tokens[bid]
            prompt_len = len(prompt)
            new_prompt_lengths.append(prompt_len)
            new_branch_bim.extend([bid] * prompt_len)
        
        return {
            'new_prompt_lengths': new_prompt_lengths,
            'new_branch_bim': new_branch_bim,
        }

    # =========================================================================
    # 工具方法
    # =========================================================================
    
    def get_active_branch_count(self) -> int:
        """获取活跃分支数量"""
        return len(self.active_branches)
    
    def get_total_generated_tokens(self) -> int:
        """获取所有分支生成的总 token 数"""
        total = 0
        for state in self.branch_states.values():
            total += len(state.output_ids)
        return total
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "mode": self.mode.value,
            "prefix_len": self.prefix_len,
            "current_seq_len": self.current_seq_len,
            "num_branches": len(self.branch_states),
            "active_branches": list(self.active_branches),
            "branch_lens": {
                bid: state.current_len 
                for bid, state in self.branch_states.items()
            },
        }

