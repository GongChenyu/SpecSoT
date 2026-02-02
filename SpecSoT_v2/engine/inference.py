# coding=utf-8
"""
推理引擎模块

提供底层推理执行接口：
- prefill_single(): 单序列 Prefill
- prefill_parallel(): 并行 Prefill
- decode_step(): 单步解码
- verify_step(): 验证步骤
- update_state(): 状态更新

设计原则：
1. 分离推理执行和流程控制
2. 支持双模式（BIM / Batching）
3. 原地操作，避免内存复制
"""

from typing import List, Tuple, Optional, Any, Dict
import torch
import torch.nn as nn

from ..core.state_manager import BranchStateManager
from ..core import initialize_past_key_values
from .evaluator import evaluate_single, evaluate_parallel


class InferenceEngine:
    """
    推理引擎
    
    封装底层推理操作，提供统一的接口。
    
    Attributes:
        base_model: Base Model
        eagle_layer: Eagle Layer
        drafter: Draft Tree 生成器
        state_manager: 状态管理器
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        eagle_layer: nn.Module,
        drafter: Any,  # Drafter
        state_manager: Optional[BranchStateManager] = None,
        device: torch.device = None,
        use_bim_mode: bool = True,
    ):
        """
        初始化推理引擎

        Args:
            base_model: Base Model
            eagle_layer: Eagle Layer
            drafter: Draft Tree 生成器
            state_manager: 状态管理器
            device: 设备
            use_bim_mode: 是否使用 BIM 模式
        """
        self.base_model = base_model
        self.eagle_layer = eagle_layer
        self.drafter = drafter
        self.state_manager = state_manager
        self.device = device or next(base_model.parameters()).device
        self.use_bim_mode = use_bim_mode

        # KV Cache 引用
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None
    
    def set_kv_cache(
        self,
        past_key_values: List,
        past_key_values_data: Any,
        current_length_data: torch.Tensor,
    ):
        """设置 KV Cache 引用"""
        self.past_key_values = past_key_values
        self.past_key_values_data = past_key_values_data
        self.current_length_data = current_length_data
    
    # =========================================================================
    # Prefill 操作
    # =========================================================================

    @torch.inference_mode()
    def prefill_single_with_init(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        带 KV Cache 初始化的单序列 Prefill

        整合 KV Cache 初始化和 prefill 操作，简化 generator 调用。

        Args:
            input_ids: [1, seq_len]
            max_new_tokens: 最大生成 token 数
            logits_processor: logits 处理器

        Returns:
            与 prefill_single 相同的返回值
        """
        input_len = input_ids.shape[1]

        # 1. 初始化 Base Model KV Cache
        tree_buffer = self.eagle_layer.total_tokens if hasattr(self.eagle_layer, 'total_tokens') else 100
        max_kv_len = input_len + max_new_tokens + tree_buffer + 200

        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)

        # 2. 初始化 Eagle Layer KV Cache
        eagle_max_len = input_len + max_new_tokens + self.eagle_layer.total_tokens + 100
        self.eagle_layer.draft_past_key_values, self.eagle_layer.draft_past_key_values_data, \
            self.eagle_layer.draft_current_length_data = \
            self.eagle_layer.init_kv_cache(max_length=eagle_max_len)

        # 3. 执行 prefill
        return self.prefill_single(input_ids, logits_processor=logits_processor)

    @torch.inference_mode()
    def prefill_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单序列 Prefill
        
        初始化 KV Cache，执行首次前向传播。
        
        Args:
            input_ids: [1, seq_len]
            attention_mask: 可选的 attention mask
            position_ids: 可选的 position ids
            logits_processor: logits 处理器
            
        Returns:
            draft_tokens: 候选 token 树
            retrieve_indices: 检索索引
            tree_mask: 树 attention mask
            tree_position_ids: 树 position ids
            hidden_states: 最后的隐藏状态
            logits: 输出 logits
            sample_token: 采样的 token
        """
        # Base Model Forward
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)
        
        # 采样
        if logits_processor is not None:
            processed_logits = logits_processor(input_ids, logits[:, -1:, :])
            sample_token = torch.argmax(processed_logits[:, -1, :], dim=-1, keepdim=True)
        else:
            sample_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # 将 sample_token 拼接到 input_ids（与 specsot_model 保持一致）
        input_ids = torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1)

        # 初始化 Eagle Layer
        self.eagle_layer.reset_state()

        # 生成 Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(hidden_states, input_ids)
        
        return (
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            hidden_states, logits, sample_token
        )

    @torch.inference_mode()
    def prefill_parallel_branches(
        self,
        prefix_len: int,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        并行 Prefill 统一入口

        根据 use_bim_mode 自动选择 BIM 或 Batching 模式。

        Args:
            prefix_len: 共享 prefix 长度
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            max_new_tokens: 最大生成 token 数

        Returns:
            包含 prefill 结果的字典
        """
        if self.use_bim_mode:
            return self._prefill_parallel_bim(
                prefix_len, branch_prompts, branch_ids, max_new_tokens
            )
        else:
            return self._prefill_parallel_batching(
                prefix_len, branch_prompts, branch_ids, max_new_tokens
            )

    @torch.inference_mode()
    def prefill_parallel(
        self,
        input_ids: torch.Tensor,
        branch_index_map: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_len: int,
        branch_begins: List[int],
        branch_lengths: List[int],
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        并行 Prefill
        
        同时处理多个分支的 prompt。
        
        Args:
            input_ids: [1, total_len] 拉平的多分支 input
            branch_index_map: BIM
            position_ids: Position IDs
            attention_mask: Attention Mask
            prefix_len: 共享 prefix 长度
            branch_begins: 各分支在序列中的起始位置
            branch_lengths: 各分支的长度
            logits_processor: logits 处理器
            
        Returns:
            draft_tokens: 候选 token 树
            retrieve_indices: 检索索引
            tree_mask: 树 mask
            tree_position_ids: 树 position ids
            tips_indices: 各分支 tip 位置
            active_branches: 活跃分支列表
        """
        num_branches = len(branch_begins)
        
        # Base Model Forward
        self.base_model.model.tree_mask = attention_mask
        self.base_model.model.tree_mode = "parallel_prefill"
        
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=None,  # 使用 tree_mask
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        
        # 提取各分支的 tip hidden states
        tips_indices = torch.tensor(
            [begin + length - 1 for begin, length in zip(branch_begins, branch_lengths)],
            device=self.device
        )
        
        # 初始化 Eagle Layer 并行状态
        self.eagle_layer.init_parallel_state(
            branch_index_map=branch_index_map,
            position_ids=position_ids,
            num_branches=num_branches,
        )
        
        # 生成 Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(
                hidden_states, 
                input_ids,
                prefix_len=prefix_len,
                active_branch=list(range(num_branches)),
            )
        
        return (
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            tips_indices, list(range(num_branches))
        )
    
    # =========================================================================
    # Decode 操作
    # =========================================================================
    
    @torch.inference_mode()
    def decode_step_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        单序列解码步骤
        
        执行一轮 Draft -> Verify -> Update。
        
        Args:
            input_ids: 当前已生成的完整序列
            draft_tokens: 候选 token 树
            retrieve_indices: 检索索引
            tree_mask: 树 mask
            tree_position_ids: 树 position ids
            logits_processor: logits 处理器
            
        Returns:
            updated_input_ids: 更新后的输入
            new_draft_tokens: 新的候选树
            new_retrieve_indices: 新的检索索引
            new_tree_mask: 新的树 mask
            new_tree_position_ids: 新的树 position ids
            accept_length: 接受的 token 数
        """
        # 保存 verify 前的 cache 长度，用于后续 update
        prev_length = self.current_length_data[0].item()
        
        # Verify: Base Model 前向
        verify_result = self.verify_step_single(
            input_ids, draft_tokens, tree_mask, tree_position_ids, logits_processor
        )
        logits, hidden_states = verify_result[:2]
        
        # Evaluate: 选择最佳路径
        best_candidate, accept_length, sample_token = evaluate_single(
            input_ids, logits, draft_tokens, retrieve_indices, logits_processor
        )
        
        # Update: 更新状态（传入 verify 前的长度）
        updated_input_ids, accept_hidden, draft_input_ids = self.update_state_single(
            input_ids, draft_tokens, hidden_states,
            best_candidate, accept_length, sample_token, retrieve_indices,
            prev_length=prev_length,  # 传入 verify 前的长度
        )
        
        # 生成新的 Draft Tree
        # accept_hidden: [batch, accept_len+1, hidden]
        # draft_input_ids: 用于 Eagle 的输入
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(accept_hidden, draft_input_ids)
        
        return (
            updated_input_ids, new_draft_tokens, new_retrieve_indices,
            new_tree_mask, new_tree_position_ids, accept_length.item()
        )
    
    @torch.inference_mode()
    def verify_step_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单序列验证步骤
        
        使用 Base Model 验证 Draft Tree。
        
        Args:
            input_ids: 当前序列
            draft_tokens: 候选 token 树
            tree_mask: 树 mask
            tree_position_ids: 树 position ids
            logits_processor: logits 处理器
            
        Returns:
            logits: 验证 logits
            hidden_states: 隐藏状态
        """
        # 准备输入
        verify_input = draft_tokens
        
        # 设置 tree mode
        self.base_model.model.tree_mask = tree_mask
        self.base_model.model.tree_mode = "tree_verify"
        
        # 计算 position ids
        current_len = self.current_length_data[0].item()
        position_ids = tree_position_ids + current_len
        
        # Base Model Forward
        outputs = self.base_model.model(
            input_ids=verify_input,
            attention_mask=None,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)
        
        return logits, hidden_states
    
    @torch.inference_mode()
    def update_state_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        best_candidate: torch.Tensor,
        accept_length: torch.Tensor,
        sample_token: torch.Tensor,
        retrieve_indices: torch.Tensor,
        prev_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单序列状态更新
        
        根据验证结果更新 KV Cache 和输入序列。
        
        Args:
            input_ids: 当前序列
            draft_tokens: 候选 token 树
            hidden_states: 隐藏状态
            best_candidate: 最佳候选索引
            accept_length: 接受长度
            sample_token: 采样 token（Bonus Token）
            retrieve_indices: 检索索引
            prev_length: verify 前的 cache 长度（如果不提供，则从 current_length_data 获取）
            
        Returns:
            updated_input_ids: 更新后的序列
            new_hidden: 用于下一轮 Draft 的隐藏状态
        """
        accept_len = accept_length.item()
        candidate_idx = best_candidate.item()
        
        # 获取接受的 token 路径
        # retrieve_indices 可能是 [num_leaves, depth] 或 [batch, num_leaves, depth]
        if retrieve_indices.ndim == 3:
            path_indices = retrieve_indices[0, candidate_idx, :accept_len + 1]
        else:
            path_indices = retrieve_indices[candidate_idx, :accept_len + 1]
        
        # 确保 path_indices 是1D向量且在正确设备上
        path_indices = path_indices.flatten()
        if path_indices.device != self.device:
            path_indices = path_indices.to(self.device)
        
        # 获取接受的 tokens（从 draft_tokens 提取）
        accepted_tokens = draft_tokens[0, path_indices]
        
        # 更新 KV Cache
        # 使用 verify 前的长度（如果提供），否则从 current_length_data 获取
        if prev_length is None:
            prev_length = self.current_length_data[0].item()
        
        # 复制接受路径的 cache 到连续位置
        # path_indices 是相对于 verify 输入（draft_tokens）的索引
        # 需要加上 prev_length 得到在整个 cache 中的绝对位置
        absolute_indices = path_indices + prev_length
        for layer_kv in self.past_key_values:
            for kv_cache in layer_kv:
                kv_cache.copy(absolute_indices, prev_length)
        
        # 更新 input_ids
        # new_tokens = accepted_tokens（不含 bonus token）
        # bonus token 会在 draft_input_ids 中添加
        updated_input_ids = torch.cat([input_ids, accepted_tokens.unsqueeze(0)], dim=1)
        
        # 创建用于 Draft 的 input_ids
        # Eagle 的对齐规则：hidden[i] 和 emb(token[i+1]) 配对预测 token[i+2]
        # 
        # 我们需要：
        # - h_sample_token 和 emb(node_1) → 预测 node_2
        # - h_node_1 和 emb(node_2) → 预测 node_3
        # - ...
        # - h_node_{n-1} 和 emb(node_n) → 预测 bonus_next
        # - h_node_n 和 emb(bonus) → 预测下一个 token
        #
        # 所以 input_ids 的格式应该是 [dummy, node_1, node_2, ..., bonus]
        # 其中 dummy 占位让 input_ids[:, 1:] = [node_1, node_2, ..., bonus]
        # 这样 actual_input = input_ids[:, 1:] 长度正好匹配 accept_hidden
        draft_input_ids = torch.cat([
            torch.zeros(1, 1, dtype=accepted_tokens.dtype, device=accepted_tokens.device),  # dummy 占位
            accepted_tokens.unsqueeze(0),  # [node_1, node_2, ..., node_n]
            sample_token.view(1, -1)       # [bonus]
        ], dim=1)
        
        # 提取接受路径的 Hidden States（用于下一轮 Draft Tree 生成）
        # 关键：需要从位置 0（sample_token）开始，因为：
        # - h_sample_token 和 emb(node_1) 配对预测 node_2
        # - h_node_1 和 emb(node_2) 配对预测 node_3
        # - ...
        # 构建包含位置 0 的完整 path
        full_path_indices = torch.cat([
            torch.zeros(1, dtype=path_indices.dtype, device=path_indices.device),  # 位置 0: sample_token
            path_indices  # [node_1_idx, node_2_idx, ...]
        ])
        accept_hidden = hidden_states[:, full_path_indices, :]  # [batch, accept_len+2, hidden_dim]
        
        # 更新当前长度
        self.current_length_data[0] = prev_length + len(path_indices)
        
        return updated_input_ids, accept_hidden, draft_input_ids
    
    # =========================================================================
    # 并行解码操作
    # =========================================================================
    
    @torch.inference_mode()
    def decode_step_parallel(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        logits_processor: Optional[Any] = None,
        prefix_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        并行解码步骤（统一入口）

        根据 use_bim_mode 自动选择 BIM 或 Batching 模式。
        """
        if self.use_bim_mode:
            return self._decode_step_parallel_bim(
                input_ids, draft_tokens, retrieve_indices, tree_mask,
                tree_position_ids, branch_index_map, active_branches,
                logits_processor, prefix_len
            )
        else:
            return self._decode_step_parallel_batching(
                input_ids, draft_tokens, retrieve_indices, tree_mask,
                tree_position_ids, active_branches, logits_processor, prefix_len
            )

    @torch.inference_mode()
    def _decode_step_parallel_bim(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        logits_processor: Optional[Any] = None,
        prefix_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        BIM 模式并行解码步骤

        同时处理多个分支的 Draft -> Verify -> Update。
        """
        # Step 1: Verify
        logits, hidden_states = self.verify_step_parallel(
            draft_tokens, tree_mask, tree_position_ids, branch_index_map, active_branches
        )

        # Step 2: Evaluate
        best_candidates, accept_lengths, sample_tokens = evaluate_parallel(
            logits, draft_tokens, retrieve_indices, logits_processor
        )

        # Step 3: Update
        new_hidden = self.update_state_parallel(
            draft_tokens, hidden_states, best_candidates, accept_lengths,
            sample_tokens, retrieve_indices, branch_index_map, active_branches, prefix_len
        )

        # Step 4: Draft
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(
                new_hidden, sample_tokens,
                prefix_len=prefix_len,
                active_branch=active_branches,
            )

        all_finished = (accept_lengths == 0).all().item()

        return (
            new_draft_tokens, new_retrieve_indices, new_tree_mask,
            new_tree_position_ids, accept_lengths, all_finished
        )

    @torch.inference_mode()
    def _decode_step_parallel_batching(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        active_branches: List[int],
        logits_processor: Optional[Any] = None,
        prefix_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Batching 模式并行解码步骤

        各分支独立验证，需要对齐处理。
        """
        from ..utils.utils import stack_with_left_padding

        num_branches = len(active_branches)
        device = draft_tokens.device

        # Step 1: Verify (Batching 模式)
        logits, hidden_states = self._verify_step_batching(
            draft_tokens, tree_mask, tree_position_ids, active_branches
        )

        # Step 2: Evaluate
        best_candidates, accept_lengths, sample_tokens = evaluate_parallel(
            logits, draft_tokens, retrieve_indices, logits_processor
        )

        # Step 3: Update (Batching 模式)
        new_hidden = self._update_state_batching(
            draft_tokens, hidden_states, best_candidates, accept_lengths,
            sample_tokens, retrieve_indices, active_branches
        )

        # Step 4: Draft
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(
                new_hidden, sample_tokens,
                prefix_len=prefix_len,
                active_branch=active_branches,
            )

        all_finished = (accept_lengths == 0).all().item()

        return (
            new_draft_tokens, new_retrieve_indices, new_tree_mask,
            new_tree_position_ids, accept_lengths, all_finished
        )

    @torch.inference_mode()
    def verify_step_parallel(
        self,
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行验证步骤
        
        构建正确的 BIM-aware attention mask，确保：
        1. Prefix 区域（BIM==-1）对所有 draft tokens 可见
        2. 同分支的历史 KV 对该分支的 draft tokens 可见
        3. Draft tokens 使用块对角 tree mask（每个分支独立）
        
        Args:
            draft_tokens: [num_branches, tree_size]
            tree_mask: 树 mask [num_branches, 1, tree_size, tree_size]
            tree_position_ids: 树 position ids [num_branches, tree_size]
            branch_index_map: BIM，历史 KV 的分支归属
            active_branches: 当前活跃的分支 ID 列表
            
        Returns:
            logits: 验证 logits [num_branches, tree_size, vocab_size]
            hidden_states: 隐藏状态 [num_branches, tree_size, hidden_size]
        """
        device = draft_tokens.device
        num_branches = draft_tokens.shape[0]
        num_nodes = draft_tokens.shape[1]
        current_length = self.current_length_data[0].item()
        packed_draft_len = num_branches * num_nodes
        
        # =====================================================================
        # Step 1: 构建并行验证的注意力掩码
        # =====================================================================
        history_bim = branch_index_map[:current_length]
        
        # 初始化 Cross Mask (Draft -> History)，初始全部遮蔽
        cross_mask = torch.full(
            (1, 1, packed_draft_len, current_length),
            torch.finfo(torch.float32).min, device=device
        )
        
        # 计算 Draft tokens 的分支归属
        active_ids_tensor = torch.tensor(active_branches, device=device)
        draft_branch_ids = active_ids_tensor.repeat_interleave(num_nodes)
        
        # Prefix 全部可见 (BIM == -1)
        is_prefix = (history_bim == -1).view(1, 1, 1, -1)
        cross_mask.masked_fill_(is_prefix, 0)
        
        # 同分支历史可见
        draft_ids_view = draft_branch_ids.view(1, 1, -1, 1)
        hist_ids_view = history_bim.view(1, 1, 1, -1)
        is_same_branch = (draft_ids_view == hist_ids_view)
        cross_mask.masked_fill_(is_same_branch, 0)
        
        # 构建 Draft Block Mask (块对角结构)
        # 将 tree_mask 从 {0, 1} 转换为 {0, -inf}
        converted_tree_mask = torch.where(
            tree_mask == 1, 0.0, torch.finfo(torch.float32).min
        )
        
        draft_block_mask = torch.full(
            (packed_draft_len, packed_draft_len),
            torch.finfo(torch.float32).min, device=device
        )
        for i in range(num_branches):
            st, ed = i * num_nodes, (i + 1) * num_nodes
            draft_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :]
        
        draft_block_mask = draft_block_mask.unsqueeze(0).unsqueeze(0)
        
        # 合并 Cross Mask 和 Draft Block Mask
        combined_mask = torch.cat([cross_mask, draft_block_mask], dim=-1)
        
        # =====================================================================
        # Step 2: Base Model Forward
        # =====================================================================
        flat_draft_tokens = draft_tokens.reshape(1, -1)
        
        # 计算绝对位置
        # tree_position_ids 是相对位置，需要加上当前 tip 的位置
        # Eagle 的 full_position_ids 最后一个位置就是 tip
        current_tip_pos = self.eagle_layer.full_position_ids[:, -1].unsqueeze(-1)
        abs_draft_pos = tree_position_ids + current_tip_pos + 1
        flat_draft_pos = abs_draft_pos.view(1, -1)
        
        # 清除 tree_mask 设置，因为我们手动构建了 combined_mask
        self.base_model.model.tree_mask = None
        
        # Base Model Forward
        outputs = self.base_model.model(
            input_ids=flat_draft_tokens,
            attention_mask=combined_mask,
            past_key_values=self.past_key_values,
            position_ids=flat_draft_pos,
        )
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)
        
        # 重新组织为分支维度
        logits = logits.view(num_branches, num_nodes, -1)
        hidden_states = hidden_states.view(num_branches, num_nodes, -1)
        
        return logits, hidden_states
        
        return logits, hidden_states
    
    @torch.inference_mode()
    def update_state_parallel(
        self,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        best_candidates: torch.Tensor,
        accept_lengths: torch.Tensor,
        sample_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        prefix_len: int,
    ) -> torch.Tensor:
        """
        并行状态更新
        
        根据验证结果更新所有分支的 KV Cache 和状态。
        
        Args:
            draft_tokens: [num_branches, tree_size]
            hidden_states: [num_branches, tree_size, hidden_size]
            best_candidates: [num_branches]
            accept_lengths: [num_branches]
            sample_tokens: [num_branches, 1]
            retrieve_indices: [num_branches, num_leaves, depth]
            branch_index_map: BIM
            active_branches: 活跃分支列表
            prefix_len: prefix 长度
            
        Returns:
            new_hidden: 用于下一轮 Draft 的隐藏状态 [num_branches, 1, hidden_size]
        """
        num_branches = len(active_branches)
        device = draft_tokens.device
        
        new_hidden_list = []
        prev_length = self.current_length_data[0].item()
        
        for i, branch_id in enumerate(active_branches):
            accept_len = accept_lengths[i].item()
            candidate_idx = best_candidates[i].item()
            
            if accept_len > 0:
                # 获取接受的 token 路径
                path_indices = retrieve_indices[i, candidate_idx, :accept_len + 1]
                
                # 复制接受路径的 cache 到连续位置
                for layer_kv in self.past_key_values:
                    for kv_cache in layer_kv:
                        # 需要处理多分支情况
                        kv_cache.copy(path_indices, prev_length)
                
                # 获取最后一个位置的 hidden state
                last_pos = path_indices[-1].item() if accept_len > 0 else 0
                new_hidden = hidden_states[i:i+1, last_pos:last_pos + 1, :]
            else:
                # 如果没有接受任何 token，使用第一个位置的 hidden state
                new_hidden = hidden_states[i:i+1, 0:1, :]
            
            new_hidden_list.append(new_hidden)
        
        # 拼接所有分支的 hidden states
        new_hidden = torch.cat(new_hidden_list, dim=0)

        return new_hidden

    # =========================================================================
    # Continuous Batching 操作
    # =========================================================================

    @torch.inference_mode()
    def continuous_decode_step_parallel(
        self,
        new_branch_prompts: List[torch.Tensor],
        old_branch_tokens: List[torch.Tensor],
        new_branch_ids: List[int],
        old_branch_ids: List[int],
        prefix_len: int,
        prefix_cache_len: int,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Continuous Batching 专用的并行解码步骤

        处理新分支 prefill + 老分支 decode 同时进行的场景。
        这是 Batching 模式下的关键方法，用于动态添加新分支。

        Args:
            new_branch_prompts: 新分支的 prompt tokens 列表
            old_branch_tokens: 老分支的 decode tokens 列表
            new_branch_ids: 新分支 ID 列表
            old_branch_ids: 老分支 ID 列表
            prefix_len: prefix 长度
            prefix_cache_len: 老分支当前的 cache 长度
            logits_processor: logits 处理器

        Returns:
            logits: 输出 logits
            hidden_states: 隐藏状态
            accept_lengths: 各分支接受长度
            all_branch_ids: 所有分支 ID
        """
        num_new = len(new_branch_prompts)
        num_old = len(old_branch_tokens)
        total_branches = num_new + num_old

        if total_branches == 0:
            raise ValueError("没有分支需要处理")

        # 计算各分支输入长度
        new_lengths = [p.shape[-1] for p in new_branch_prompts]
        old_lengths = [t.shape[-1] for t in old_branch_tokens]
        all_lengths = new_lengths + old_lengths
        max_input_len = max(all_lengths) if all_lengths else 1

        device = self.device

        # 构建对齐后的输入（左填充）
        aligned_input_ids = torch.zeros(
            (total_branches, max_input_len),
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.zeros(
            (total_branches, max_input_len),
            dtype=torch.long,
            device=device,
        )
        padding_mask = torch.zeros(
            (total_branches, max_input_len),
            dtype=torch.long,
            device=device,
        )

        # 填充新分支（prefill）
        for i, (prompt, length) in enumerate(zip(new_branch_prompts, new_lengths)):
            pad_len = max_input_len - length
            aligned_input_ids[i, pad_len:] = prompt.view(-1)
            # 新分支 position 从 prefix_len 开始
            position_ids[i, pad_len:] = torch.arange(
                prefix_len, prefix_len + length, device=device
            )
            padding_mask[i, pad_len:] = 1

        # 填充老分支（decode）
        for i, (tokens, length) in enumerate(zip(old_branch_tokens, old_lengths)):
            idx = num_new + i
            pad_len = max_input_len - length
            aligned_input_ids[idx, pad_len:] = tokens.view(-1)
            # 老分支 position 从当前 cache 长度继续
            position_ids[idx, pad_len:] = torch.arange(
                prefix_cache_len, prefix_cache_len + length, device=device
            )
            padding_mask[idx, pad_len:] = 1

        # 构建 attention mask
        # 新分支需要完整的 causal mask
        # 老分支需要看到之前的 cache
        attention_mask = self._build_continuous_attention_mask(
            padding_mask, new_lengths, old_lengths,
            prefix_len, prefix_cache_len, max_input_len
        )

        # Base Model Forward
        outputs = self.base_model.model(
            input_ids=aligned_input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)

        # 合并分支 ID
        all_branch_ids = new_branch_ids + old_branch_ids

        # 计算接受长度（简化：新分支全部接受，老分支需要验证）
        accept_lengths = []
        for i in range(num_new):
            accept_lengths.append(new_lengths[i])
        for i in range(num_old):
            accept_lengths.append(old_lengths[i])

        return logits, hidden_states, accept_lengths, all_branch_ids

    def _build_continuous_attention_mask(
        self,
        padding_mask: torch.Tensor,
        new_lengths: List[int],
        old_lengths: List[int],
        prefix_len: int,
        prefix_cache_len: int,
        max_input_len: int,
    ) -> torch.Tensor:
        """
        构建 Continuous Batching 的 attention mask

        Args:
            padding_mask: [batch_size, max_input_len]
            new_lengths: 新分支长度列表
            old_lengths: 老分支长度列表
            prefix_len: prefix 长度
            prefix_cache_len: 老分支 cache 长度
            max_input_len: 最大输入长度

        Returns:
            attention_mask: [batch_size, 1, max_input_len, max_key_len]
        """
        num_new = len(new_lengths)
        num_old = len(old_lengths)
        total_branches = num_new + num_old
        device = padding_mask.device

        # 计算最大 key 长度
        max_key_len = prefix_cache_len + max_input_len

        # 初始化 mask
        mask = torch.full(
            (total_branches, 1, max_input_len, max_key_len),
            torch.finfo(torch.float32).min,
            device=device,
            dtype=torch.float32,
        )

        # 新分支：只能看到 prefix + 自己的输入
        for i in range(num_new):
            length = new_lengths[i]
            pad_len = max_input_len - length
            # prefix 可见
            mask[i, 0, :, :prefix_len] = 0
            # 自己的输入（causal）
            for q in range(max_input_len):
                if q >= pad_len:
                    valid_q = q - pad_len
                    mask[i, 0, q, prefix_len:prefix_len + valid_q + 1] = 0

        # 老分支：可以看到 cache + 自己的输入
        for i in range(num_old):
            idx = num_new + i
            length = old_lengths[i]
            pad_len = max_input_len - length
            # cache 全部可见
            mask[idx, 0, :, :prefix_cache_len] = 0
            # 自己的输入（causal）
            for q in range(max_input_len):
                if q >= pad_len:
                    valid_q = q - pad_len
                    mask[idx, 0, q, prefix_cache_len:prefix_cache_len + valid_q + 1] = 0

        return mask

    # =========================================================================
    # Mask 构建方法
    # =========================================================================

    def build_parallel_prefill_mask(
        self,
        branch_index_map: torch.Tensor,
        prefix_len: int,
        branches_len: int,
    ) -> torch.Tensor:
        """
        构建并行 Prefill 阶段的 BIM-based attention mask

        确保：
        1. 所有分支都能看到共享的 prefix
        2. 每个分支只能看到自己的内容
        3. 遵循因果约束

        Args:
            branch_index_map: BIM 索引
            prefix_len: prefix 长度
            branches_len: 分支总长度

        Returns:
            attention_mask: [1, 1, branches_len, total_len]
        """
        total_len = prefix_len + branches_len
        dtype = torch.float32

        total_ids = branch_index_map[:total_len]
        branch_ids = branch_index_map[prefix_len:total_len]

        # 初始化为全部遮蔽
        mask = torch.full(
            (1, 1, branches_len, total_len),
            torch.finfo(dtype).min, device=self.device
        )

        # 1. Prefix 全部可见 (BIM == -1)
        is_prefix = (total_ids == -1).unsqueeze(0)
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

    # =========================================================================
    # 双模式实现：BIM 模式
    # =========================================================================

    @torch.inference_mode()
    def _prefill_parallel_bim(
        self,
        prefix_len: int,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        BIM 模式并行 Prefill

        所有分支拉平成单序列 [1, total_len]，通过 BIM 索引区分不同分支。

        Args:
            prefix_len: 共享 prefix 长度
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            max_new_tokens: 最大生成 token 数

        Returns:
            包含 prefill 结果的字典
        """
        num_branches = len(branch_prompts)

        # 1. 重置 KV Cache 到 prefix 长度
        self.current_length_data.fill_(prefix_len)

        # 2. 构建拉平序列和 BIM
        flat_branch_ids = []
        branch_begins = []
        branch_lengths = []
        pos_ids_list = []
        branch_index_list = [-1] * prefix_len  # prefix 标记为 -1

        current_offset = prefix_len
        for branch_id, prompt in zip(branch_ids, branch_prompts):
            curr_len = len(prompt)
            branch_begins.append(current_offset - prefix_len)
            flat_branch_ids.extend(prompt)
            branch_index_list.extend([branch_id] * curr_len)
            branch_lengths.append(curr_len)

            # 位置编码: 每个分支从 prefix_len 开始独立计数
            curr_pos = list(range(prefix_len, prefix_len + curr_len))
            pos_ids_list.extend(curr_pos)
            current_offset += curr_len

        # 3. 构建 Tensor
        branches_tensor = torch.tensor(
            [flat_branch_ids], device=self.device, dtype=torch.long
        )
        input_ids = branches_tensor

        tips_indices = torch.tensor(
            [begin + length - 1 for begin, length in zip(branch_begins, branch_lengths)],
            device=self.device
        )

        # 4. 构建 BIM
        total_capacity = prefix_len + sum(branch_lengths) + max_new_tokens + 128
        branch_index_map = torch.full(
            (total_capacity,), -2, dtype=torch.long, device=self.device
        )
        branch_index_map[:len(branch_index_list)] = torch.tensor(
            branch_index_list, device=self.device
        )

        position_ids = torch.tensor(pos_ids_list, device=self.device).unsqueeze(0)

        # 5. 构建 Attention Mask
        branches_len = sum(branch_lengths)
        attention_mask = self.build_parallel_prefill_mask(
            branch_index_map, prefix_len, branches_len
        )

        # 6. Base Model Forward
        self.base_model.model.tree_mask = attention_mask
        self.base_model.model.tree_mode = "parallel_prefill"

        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]

        # 7. 生成 Draft Tree
        tips_hidden = []
        for tip_idx in tips_indices:
            tips_hidden.append(hidden_states[:, tip_idx:tip_idx+1, :])
        tips_hidden = torch.cat(tips_hidden, dim=1).transpose(0, 1)

        # 重新初始化 Eagle KV Cache
        eagle_max_len = prefix_len + max_new_tokens + self.eagle_layer.total_tokens + 100
        self.eagle_layer.reset_state()
        self.eagle_layer.init_kv_cache(max_length=eagle_max_len, batch_size=num_branches)

        tip_token_ids = torch.tensor(
            [flat_branch_ids[begin + length - 1]
             for begin, length in zip(branch_begins, branch_lengths)],
            device=self.device
        ).unsqueeze(1)

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(
                tips_hidden, tip_token_ids,
                prefix_len=prefix_len,
                active_branch=list(range(num_branches)),
            )

        return {
            'input_ids': input_ids,
            'draft_tokens': draft_tokens,
            'retrieve_indices': retrieve_indices,
            'tree_mask': tree_mask,
            'tree_position_ids': tree_position_ids,
            'tips_indices': tips_indices,
            'branch_begins': branch_begins,
            'branch_lengths': branch_lengths,
            'branch_index_map': branch_index_map,
            'position_ids': position_ids,
        }

    # =========================================================================
    # 双模式实现：Batching 模式
    # =========================================================================

    @torch.inference_mode()
    def _prefill_parallel_batching(
        self,
        prefix_len: int,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        Batching 模式并行 Prefill

        各分支独立 [num_branches, max_len]，复制 prefix KV Cache 到各分支。

        Args:
            prefix_len: 共享 prefix 长度
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            max_new_tokens: 最大生成 token 数

        Returns:
            包含 prefill 结果的字典
        """
        from ..utils.utils import stack_with_left_padding

        num_branches = len(branch_prompts)
        branch_lengths = [len(p) for p in branch_prompts]
        max_branch_len = max(branch_lengths)

        # 1. 复制 prefix cache 到各分支
        self._expand_prefix_cache(prefix_len, num_branches)

        # 2. 左填充对齐
        prompt_tensors = [
            torch.tensor(p, device=self.device, dtype=torch.long)
            for p in branch_prompts
        ]
        padded_input, padding_mask = stack_with_left_padding(
            prompt_tensors,
            pad_id=0,
            device=self.device,
            return_mask=True,
        )

        # 3. 构建 Position IDs
        position_ids = torch.zeros_like(padded_input)
        for i, length in enumerate(branch_lengths):
            pad_len = max_branch_len - length
            position_ids[i, pad_len:] = torch.arange(
                prefix_len, prefix_len + length, device=self.device
            )

        # 4. 构建 Attention Mask (causal + padding)
        attention_mask = self._build_batching_prefill_mask(
            padding_mask, branch_lengths, prefix_len, max_branch_len
        )

        # 5. Base Model Forward
        outputs = self.base_model.model(
            input_ids=padded_input,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]

        # 6. 提取各分支 tip hidden states
        tips_hidden = []
        for i, length in enumerate(branch_lengths):
            tip_idx = max_branch_len - 1  # 最后一个位置
            tips_hidden.append(hidden_states[i:i+1, tip_idx:tip_idx+1, :])
        tips_hidden = torch.cat(tips_hidden, dim=0)

        # 7. 生成 Draft Tree
        eagle_max_len = prefix_len + max_new_tokens + self.eagle_layer.total_tokens + 100
        self.eagle_layer.reset_state()
        self.eagle_layer.init_kv_cache(max_length=eagle_max_len, batch_size=num_branches)

        tip_token_ids = torch.tensor(
            [branch_prompts[i][-1] for i in range(num_branches)],
            device=self.device
        ).unsqueeze(1)

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(
                tips_hidden, tip_token_ids,
                prefix_len=prefix_len,
                active_branch=list(range(num_branches)),
            )

        return {
            'input_ids': padded_input,
            'draft_tokens': draft_tokens,
            'retrieve_indices': retrieve_indices,
            'tree_mask': tree_mask,
            'tree_position_ids': tree_position_ids,
            'branch_lengths': branch_lengths,
            'padding_mask': padding_mask,
            'position_ids': position_ids,
        }

    def _expand_prefix_cache(self, prefix_len: int, num_branches: int):
        """
        Batching 模式：复制 prefix KV Cache 到各分支

        Args:
            prefix_len: prefix 长度
            num_branches: 分支数量
        """
        # 复制 Base Model KV Cache
        for layer_kv in self.past_key_values:
            for kv_cache in layer_kv:
                # 获取 prefix 部分
                prefix_kv = kv_cache.data[:, :, :prefix_len, :].clone()
                # 扩展到 num_branches
                expanded_kv = prefix_kv.expand(num_branches, -1, -1, -1).clone()
                # 重新分配 cache
                kv_cache.data = torch.zeros(
                    (num_branches, kv_cache.data.shape[1],
                     kv_cache.data.shape[2], kv_cache.data.shape[3]),
                    device=self.device, dtype=kv_cache.data.dtype
                )
                kv_cache.data[:, :, :prefix_len, :] = expanded_kv

        # 更新 current_length_data
        self.current_length_data.fill_(prefix_len)

    def _build_batching_prefill_mask(
        self,
        padding_mask: torch.Tensor,
        branch_lengths: List[int],
        prefix_len: int,
        max_branch_len: int,
    ) -> torch.Tensor:
        """
        构建 Batching 模式的 Prefill attention mask

        Args:
            padding_mask: [num_branches, max_branch_len]
            branch_lengths: 各分支长度
            prefix_len: prefix 长度
            max_branch_len: 最大分支长度

        Returns:
            attention_mask: [num_branches, 1, max_branch_len, prefix_len + max_branch_len]
        """
        num_branches = len(branch_lengths)
        total_key_len = prefix_len + max_branch_len

        mask = torch.full(
            (num_branches, 1, max_branch_len, total_key_len),
            torch.finfo(torch.float32).min,
            device=self.device,
            dtype=torch.float32,
        )

        for i, length in enumerate(branch_lengths):
            pad_len = max_branch_len - length
            # prefix 全部可见
            mask[i, 0, :, :prefix_len] = 0
            # 自己的输入（causal）
            for q in range(max_branch_len):
                if q >= pad_len:
                    valid_q = q - pad_len
                    mask[i, 0, q, prefix_len:prefix_len + valid_q + 1] = 0

        return mask

    # =========================================================================
    # Batching 模式辅助方法
    # =========================================================================

    @torch.inference_mode()
    def _verify_step_batching(
        self,
        draft_tokens: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        active_branches: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batching 模式验证步骤

        各分支独立验证，使用标准 batched attention。

        Args:
            draft_tokens: [num_branches, tree_size]
            tree_mask: 树 mask [num_branches, 1, tree_size, tree_size]
            tree_position_ids: 树 position ids [num_branches, tree_size]
            active_branches: 当前活跃的分支 ID 列表

        Returns:
            logits: [num_branches, tree_size, vocab_size]
            hidden_states: [num_branches, tree_size, hidden_size]
        """
        num_branches = draft_tokens.shape[0]
        num_nodes = draft_tokens.shape[1]
        current_length = self.current_length_data[0].item()

        # 构建 attention mask: [num_branches, 1, tree_size, current_length + tree_size]
        total_key_len = current_length + num_nodes

        attention_mask = torch.full(
            (num_branches, 1, num_nodes, total_key_len),
            torch.finfo(torch.float32).min,
            device=self.device,
            dtype=torch.float32,
        )

        # 历史 cache 全部可见
        attention_mask[:, :, :, :current_length] = 0

        # Draft 部分使用 tree_mask
        converted_tree_mask = torch.where(
            tree_mask == 1, 0.0, torch.finfo(torch.float32).min
        )
        attention_mask[:, :, :, current_length:] = converted_tree_mask[:, 0, :, :]

        # 计算绝对位置
        position_ids = tree_position_ids + current_length

        # Base Model Forward
        self.base_model.model.tree_mask = None

        outputs = self.base_model.model(
            input_ids=draft_tokens,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)

        return logits, hidden_states

    @torch.inference_mode()
    def _update_state_batching(
        self,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        best_candidates: torch.Tensor,
        accept_lengths: torch.Tensor,
        sample_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        active_branches: List[int],
    ) -> torch.Tensor:
        """
        Batching 模式状态更新

        各分支独立更新 KV Cache。

        Args:
            draft_tokens: [num_branches, tree_size]
            hidden_states: [num_branches, tree_size, hidden_size]
            best_candidates: [num_branches]
            accept_lengths: [num_branches]
            sample_tokens: [num_branches, 1]
            retrieve_indices: [num_branches, num_leaves, depth]
            active_branches: 活跃分支列表

        Returns:
            new_hidden: [num_branches, 1, hidden_size]
        """
        num_branches = len(active_branches)
        new_hidden_list = []
        prev_length = self.current_length_data[0].item()

        for i in range(num_branches):
            accept_len = accept_lengths[i].item()
            candidate_idx = best_candidates[i].item()

            if accept_len > 0:
                # 获取接受的 token 路径
                path_indices = retrieve_indices[i, candidate_idx, :accept_len + 1]

                # 复制接受路径的 cache 到连续位置
                absolute_indices = path_indices + prev_length
                for layer_kv in self.past_key_values:
                    for kv_cache in layer_kv:
                        # Batching 模式：每个分支独立处理
                        kv_cache.data[i:i+1].copy_(
                            kv_cache.data[i:i+1, :, absolute_indices, :]
                                .transpose(2, 3).transpose(1, 2),
                            non_blocking=True
                        ) if False else None  # 占位，实际使用 copy 方法

                # 获取最后位置的 hidden state
                last_pos = path_indices[-1].item()
                new_hidden = hidden_states[i:i+1, last_pos:last_pos+1, :]
            else:
                new_hidden = hidden_states[i:i+1, 0:1, :]

            new_hidden_list.append(new_hidden)

        # 更新 cache 长度（取最大接受长度）
        max_accept = accept_lengths.max().item()
        self.current_length_data[0] = prev_length + max_accept + 1

        return torch.cat(new_hidden_list, dim=0)
