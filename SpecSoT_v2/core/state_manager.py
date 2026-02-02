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
        
        # ===== BIM 模式状态 =====
        # BIM: Branch Index Map
        self.bim: Optional[torch.Tensor] = None
        # Position IDs
        self.position_ids: Optional[torch.Tensor] = None
        # 预分配的 BIM 缓冲区
        self._bim_buffer: Optional[torch.Tensor] = None
        self._position_buffer: Optional[torch.Tensor] = None
        
        # ===== 通用状态 =====
        # 分支状态字典
        self.branch_states: Dict[int, BranchState] = {}
        # 活跃分支列表
        self.active_branches: List[int] = []
        # 共享 prefix 长度
        self.prefix_len: int = 0
        # 当前总序列长度（BIM 模式）
        self.current_seq_len: int = 0
        
        # ===== Batching 模式状态 =====
        # 各分支的有效长度（Batching 模式下所有分支 cache 长度相同，但有效长度不同）
        self.valid_lengths: Optional[torch.Tensor] = None
        # 各分支的 padding 长度
        self.padding_lengths: Optional[torch.Tensor] = None
        
        # 初始化缓冲区
        self._init_buffers()
    
    def _init_buffers(self):
        """初始化预分配缓冲区"""
        if self.use_bim_mode:
            # BIM 缓冲区: [max_seq_len]
            self._bim_buffer = torch.full(
                (self.max_seq_len,), 
                self.BIM_EMPTY, 
                dtype=torch.long, 
                device=self.device
            )
            # Position 缓冲区: [max_seq_len]
            self._position_buffer = torch.zeros(
                self.max_seq_len, 
                dtype=torch.long, 
                device=self.device
            )
        else:
            # Batching 模式: 有效长度缓冲区
            self.valid_lengths = torch.zeros(
                self.max_branches, 
                dtype=torch.long, 
                device=self.device
            )
            self.padding_lengths = torch.zeros(
                self.max_branches, 
                dtype=torch.long, 
                device=self.device
            )
    
    def reset(self):
        """重置所有状态"""
        self.branch_states.clear()
        self.active_branches.clear()
        self.prefix_len = 0
        self.current_seq_len = 0
        
        if self.use_bim_mode:
            self._bim_buffer.fill_(self.BIM_EMPTY)
            self._position_buffer.fill_(0)
            self.bim = None
            self.position_ids = None
        else:
            self.valid_lengths.fill_(0)
            self.padding_lengths.fill_(0)
    
    # =========================================================================
    # BIM 模式接口
    # =========================================================================
    
    def init_prefix(self, prefix_len: int):
        """
        初始化共享 prefix
        
        Args:
            prefix_len: prefix 长度
        """
        self.prefix_len = prefix_len
        self.current_seq_len = prefix_len
        
        if self.use_bim_mode:
            # 设置 prefix 区域的 BIM
            self._bim_buffer[:prefix_len] = self.BIM_PREFIX
            # 设置 prefix 的 position ids
            self._position_buffer[:prefix_len] = torch.arange(
                prefix_len, device=self.device
            )
            # 更新视图
            self.bim = self._bim_buffer[:prefix_len]
            self.position_ids = self._position_buffer[:prefix_len]
    
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
            self._bim_buffer[start_pos:end_pos] = branch_id
            
            # 设置 Position IDs（从 prefix_len 开始）
            self._position_buffer[start_pos:end_pos] = torch.arange(
                self.prefix_len, 
                self.prefix_len + branch_len, 
                device=self.device
            )
            
            # 更新当前序列长度
            self.current_seq_len = max(self.current_seq_len, end_pos)
            
            # 更新视图
            self.bim = self._bim_buffer[:self.current_seq_len]
            self.position_ids = self._position_buffer[:self.current_seq_len]
            
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
            self.valid_lengths[len(self.active_branches)] = state.cache_len
        
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
            
            self._bim_buffer[old_end:new_end] = branch_id
            
            # Position 继续递增
            for i in range(num_tokens):
                self._position_buffer[old_end + i] = self.prefix_len + state.current_len + i
            
            self.current_seq_len = max(self.current_seq_len, new_end)
            
            # 更新视图
            self.bim = self._bim_buffer[:self.current_seq_len]
            self.position_ids = self._position_buffer[:self.current_seq_len]
        
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
            self._bim_buffer[start_pos:end_pos] = branch_id

            # Position IDs: 每个分支从 prefix_len 开始独立计数
            curr_pos = list(range(self.prefix_len, self.prefix_len + curr_len))
            pos_ids_list.extend(curr_pos)

            # 更新 Position buffer
            self._position_buffer[start_pos:end_pos] = torch.tensor(
                curr_pos, device=self.device, dtype=torch.long
            )

            current_offset += curr_len

        # 更新序列长度
        self.current_seq_len = self.prefix_len + current_offset
        self.bim = self._bim_buffer[:self.current_seq_len]
        self.position_ids = self._position_buffer[:self.current_seq_len]

        # 构建输出 Tensor
        flat_tokens = torch.tensor(
            [flat_tokens_list], device=self.device, dtype=torch.long
        )
        position_ids = torch.tensor(
            pos_ids_list, device=self.device, dtype=torch.long
        ).unsqueeze(0)

        return flat_tokens, self._bim_buffer, position_ids, branch_begins, branch_lengths

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

        num_branches = len(branch_prompts)
        branch_lengths = [len(p) for p in branch_prompts]
        max_len = max(branch_lengths)

        # 分配输出 tensor
        padded_input = torch.full(
            (num_branches, max_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.zeros(
            (num_branches, max_len),
            dtype=torch.long,
            device=self.device,
        )
        padding_mask = torch.zeros(
            (num_branches, max_len),
            dtype=torch.long,
            device=self.device,
        )

        # 左填充对齐
        for i, (prompt, length) in enumerate(zip(branch_prompts, branch_lengths)):
            pad_len = max_len - length
            padded_input[i, pad_len:] = torch.tensor(
                prompt, device=self.device, dtype=torch.long
            )
            position_ids[i, pad_len:] = torch.arange(
                self.prefix_len, self.prefix_len + length, device=self.device
            )
            padding_mask[i, pad_len:] = 1

            # 更新 padding_lengths
            self.padding_lengths[i] = pad_len
            self.valid_lengths[i] = self.prefix_len + length

        return padded_input, position_ids, padding_mask
    
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
        bim = self.bim[:key_len] if self.bim is not None else self._bim_buffer[:key_len]
        
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
        
        # padding 掩码（如果有）
        if self.padding_lengths is not None:
            for i in range(min(batch_size, len(self.active_branches))):
                pad_len = self.padding_lengths[i].item()
                if pad_len > 0:
                    mask[i, :, :, :pad_len] = torch.finfo(self.dtype).min
        
        return mask
    
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


class AlignmentManager:
    """
    Batching 模式对齐管理器
    
    处理不同长度输入的对齐：
    - 左填充对齐
    - Position IDs 计算
    - Attention Mask 构建
    """
    
    def __init__(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        pad_token_id: int = 0,
    ):
        """
        初始化对齐管理器
        
        Args:
            device: 设备
            dtype: 数据类型
            pad_token_id: padding token ID
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pad_token_id = pad_token_id
    
    def align_inputs(
        self,
        input_ids_list: List[torch.Tensor],
        cache_lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对齐不同长度的输入
        
        Args:
            input_ids_list: 各分支的 input_ids 列表
            cache_lengths: 各分支的 KV Cache 长度
            
        Returns:
            aligned_input_ids: [batch_size, max_len]
            position_ids: [batch_size, max_len]
            padding_mask: [batch_size, max_len] (1=有效, 0=padding)
        """
        batch_size = len(input_ids_list)
        input_lengths = [ids.shape[-1] for ids in input_ids_list]
        max_len = max(input_lengths)
        
        # 分配输出 tensor
        aligned_input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=self.device,
        )
        padding_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=self.device,
        )
        
        # 左填充对齐
        for i, (ids, input_len, cache_len) in enumerate(
            zip(input_ids_list, input_lengths, cache_lengths)
        ):
            pad_len = max_len - input_len
            
            # 复制 input_ids（右对齐）
            aligned_input_ids[i, pad_len:] = ids.view(-1)
            
            # Position IDs（从 cache_len 开始）
            position_ids[i, pad_len:] = torch.arange(
                cache_len, cache_len + input_len, device=self.device
            )
            
            # Padding mask
            padding_mask[i, pad_len:] = 1
        
        return aligned_input_ids, position_ids, padding_mask
    
    def build_aligned_attention_mask(
        self,
        padding_mask: torch.Tensor,
        cache_lengths: List[int],
        query_len: int,
    ) -> torch.Tensor:
        """
        构建对齐后的 attention mask
        
        Args:
            padding_mask: [batch_size, query_len]
            cache_lengths: 各分支的 KV Cache 长度
            query_len: query 长度
            
        Returns:
            attention_mask: [batch_size, 1, query_len, max_key_len]
        """
        batch_size = padding_mask.shape[0]
        max_cache_len = max(cache_lengths)
        max_key_len = max_cache_len + query_len
        
        # 初始化
        mask = torch.full(
            (batch_size, 1, query_len, max_key_len),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )
        
        for i, cache_len in enumerate(cache_lengths):
            # 因果掩码
            for q in range(query_len):
                # 可以看到 cache 中的所有位置
                mask[i, 0, q, :cache_len] = 0
                # 可以看到当前位置及之前的输入
                if padding_mask[i, q] == 1:
                    # 找到这个 query 位置对应的有效范围
                    valid_start = (padding_mask[i, :q+1] == 0).sum().item()
                    mask[i, 0, q, cache_len:cache_len + q + 1 - valid_start] = 0
        
        return mask
    
    def align_caches_for_new_branch(
        self,
        prefix_cache: Any,  # KVCache
        target_length: int,
        prefix_len: int,
    ) -> Any:
        """
        为新分支创建等长的 KV Cache
        
        Batching 模式下，所有分支的 cache 长度必须一致。
        新分支的 cache = prefix + padding
        
        Args:
            prefix_cache: 共享的 prefix cache
            target_length: 目标 cache 长度（老分支的 cache 长度）
            prefix_len: prefix 长度
            
        Returns:
            新分支的 KV Cache
        """
        # 这个方法需要 KVCache 的具体实现
        # 这里只提供接口定义
        raise NotImplementedError("需要 KVCache 类的具体实现")


def build_bim_for_parallel_branches(
    prefix_len: int,
    branch_lengths: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为并行分支构建 BIM 和 Position IDs
    
    Args:
        prefix_len: prefix 长度
        branch_lengths: 各分支长度列表
        device: 设备
        
    Returns:
        bim: Branch Index Map [total_len]
        position_ids: Position IDs [total_len]
    """
    total_len = prefix_len + sum(branch_lengths)
    
    bim = torch.empty(total_len, dtype=torch.long, device=device)
    position_ids = torch.empty(total_len, dtype=torch.long, device=device)
    
    # Prefix
    bim[:prefix_len] = BranchStateManager.BIM_PREFIX
    position_ids[:prefix_len] = torch.arange(prefix_len, device=device)
    
    # Branches
    current_pos = prefix_len
    for branch_id, branch_len in enumerate(branch_lengths):
        bim[current_pos:current_pos + branch_len] = branch_id
        position_ids[current_pos:current_pos + branch_len] = torch.arange(
            prefix_len, prefix_len + branch_len, device=device
        )
        current_pos += branch_len
    
    return bim, position_ids
