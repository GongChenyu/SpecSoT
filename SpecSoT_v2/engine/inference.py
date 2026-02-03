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
from ..core import initialize_past_key_values, initialize_eagle_past_key_values
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
        use_eagle3: bool = True,
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
            use_eagle3: 是否使用 EAGLE3（需要拼接 3 层 hidden states）
        """
        self.base_model = base_model
        self.eagle_layer = eagle_layer
        self.drafter = drafter
        self.state_manager = state_manager
        self.device = device or next(base_model.parameters()).device
        self.use_bim_mode = use_bim_mode
        self.use_eagle3 = use_eagle3

        # KV Cache 引用
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

    # =========================================================================
    # KV Cache 初始化
    # =========================================================================

    def init_kv_caches(self, input_len: int, max_new_tokens: int):
        """
        统一初始化 Base Model 和 Eagle Layer 的 KV Cache

        Args:
            input_len: 输入序列长度
            max_new_tokens: 最大生成 token 数
        """
        # 1. 计算最大长度
        tree_buffer = self.eagle_layer.total_tokens if hasattr(self.eagle_layer, 'total_tokens') else 100
        max_kv_len = input_len + max_new_tokens + tree_buffer + 200

        # 2. 初始化 Base Model KV Cache
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)

        # 3. 初始化 Eagle Layer KV Cache（统一使用 initialize_eagle_past_key_values）
        self.draft_past_key_values, self.draft_past_key_values_data, self.draft_current_length_data = \
            initialize_eagle_past_key_values(self.eagle_layer, max_length=max_kv_len)
        
        # 4. 将 KV Cache 设置到 Eagle Layer 上（供 forward 使用）
        self.eagle_layer.draft_past_key_values = self.draft_past_key_values
        self.eagle_layer.draft_past_key_values_data = self.draft_past_key_values_data
        self.eagle_layer.draft_current_length_data = self.draft_current_length_data

    # =========================================================================
    # Hidden States 处理辅助方法
    # =========================================================================

    def _prepare_draft_input_hidden(
        self,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        从 Base Model 输出中提取 Draft Model (Eagle) 的输入 hidden states
        
        两种用途的 hidden states:
        1. last_hidden_state: 用于计算 logits (通过 lm_head)
        2. draft_input_hidden: 用于输入 Draft Model (Eagle)
           - EAGLE3: 拼接 3 层 hidden states，维度 hidden_size * 3
           - EAGLE2: 只需要最后一层，维度 hidden_size
        
        Args:
            outputs: Base Model 的输出字典，包含:
                - "last_hidden_state": [batch, seq_len, hidden_size]
                - "hidden_states": tuple of 3 hidden states (for EAGLE3)
                
        Returns:
            draft_input_hidden: Eagle 输入的 hidden states
                - EAGLE3: [batch, seq_len, hidden_size * 3]
                - EAGLE2: [batch, seq_len, hidden_size]
        """
        if self.use_eagle3:
            # EAGLE3: 拼接 3 层 hidden states
            ea_device = self.eagle_layer.lm_head.weight.device
            eagle_hidden_states = outputs["hidden_states"]  # tuple of 3 hidden states
            if eagle_hidden_states[0].device != ea_device:
                eagle_hidden_states = [x.to(ea_device) for x in eagle_hidden_states]
            draft_input_hidden = torch.cat(eagle_hidden_states, dim=-1)
        else:
            # EAGLE2: 只需要最后一层 hidden state
            draft_input_hidden = outputs["last_hidden_state"]
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if draft_input_hidden.device != ea_device:
                draft_input_hidden = draft_input_hidden.to(ea_device)
        
        return draft_input_hidden

    # =========================================================================
    # Prefill 操作
    # =========================================================================

    @torch.inference_mode()
    def prefill_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        logits_processor: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单序列 Prefill
        
        执行首次前向传播。如果提供 max_new_tokens，会自动初始化 KV Cache。
        
        Args:
            input_ids: [1, seq_len]
            attention_mask: 可选的 attention mask
            position_ids: 可选的 position ids
            logits_processor: logits 处理器
            max_new_tokens: 最大生成 token 数（用于初始化 KV Cache）
            
        Returns:
            draft_tokens: 候选 token 树
            retrieve_indices: 检索索引
            tree_mask: 树 attention mask
            tree_position_ids: 树 position ids
            hidden_states: 最后的隐藏状态
            logits: 输出 logits
            sample_token: 采样的 token
        """
        # 如果提供了 max_new_tokens，则初始化 KV Cache
        if max_new_tokens is not None:
            input_len = input_ids.shape[1]
            self.init_kv_caches(input_len, max_new_tokens)

        # Base Model Forward
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]  # 用于计算 logits
        logits = self.base_model.lm_head(last_hidden_state)
        
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

        # 提取 Draft Model 的输入 hidden states（完整序列）
        draft_input_hidden = self._prepare_draft_input_hidden(outputs)  # [batch, seq_len, hidden(*3)]

        # 生成 Draft Tree（传入完整的 hidden states 和 input_ids）
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(draft_input_hidden, input_ids)
        
        return (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)

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
        logits, all_hidden_states = self.verify_step_single(
            input_ids, draft_tokens, tree_mask, tree_position_ids, logits_processor
        )
        
        # Evaluate: 选择最佳路径
        best_candidate, accept_length, sample_token = evaluate_single(
            input_ids, logits, draft_tokens, retrieve_indices, logits_processor
        )
        print(f"Accept Length: {accept_length.item()}")
        
        # Update: 更新状态（传入 verify 前的长度）
        updated_input_ids, draft_input_hidden, draft_input_ids = self.update_state_single(
            input_ids, draft_tokens, all_hidden_states,
            best_candidate, accept_length, sample_token, retrieve_indices,
            prev_length=prev_length,  # 传入 verify 前的长度
        )
        
        # 生成新的 Draft Tree
        # accept_hidden: [batch, accept_len+1, hidden]
        # draft_input_ids: 用于 Eagle 的输入
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(draft_input_hidden, draft_input_ids)
        
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
            hidden_states: 用于 Eagle 的隐藏状态（EAGLE3: hidden*3, EAGLE2: hidden）
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
        last_hidden_state = outputs["last_hidden_state"]  # 用于计算 logits
        logits = self.base_model.lm_head(last_hidden_state)
        
        # 提取 Draft Model 的输入 hidden states
        draft_input_hidden = self._prepare_draft_input_hidden(outputs)
        
        return logits, draft_input_hidden
    
    @torch.inference_mode()
    def update_state_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        all_hidden_states: torch.Tensor,
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
        # 维度标准化：确保 retrieve_indices 是 [B, L, D]
        if retrieve_indices.ndim == 2:
            retrieve_indices = retrieve_indices.unsqueeze(0)

        # 安全获取标量值（兼容 [B] 和标量两种情况）
        accept_len = accept_length[0].item() if accept_length.ndim > 0 else accept_length.item()
        candidate_idx = best_candidate[0].item() if best_candidate.ndim > 0 else best_candidate.item()

        # 获取接受的 token 路径 - 始终使用 3D 索引
        batch_idx = 0  # 单序列模式
        path_indices = retrieve_indices[batch_idx, candidate_idx, :accept_len + 1]

        # 确保 path_indices 是 1D 向量且在正确设备上
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
        draft_input_hidden = all_hidden_states[:, full_path_indices, :]  # [batch, accept_len+2, hidden_dim]
        
        # 更新当前长度
        self.current_length_data[0] = prev_length + len(path_indices)
        
        return updated_input_ids, draft_input_hidden, draft_input_ids
    
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
        logits, draft_input_hidden = self.verify_step_parallel(
            draft_tokens, tree_mask, tree_position_ids, branch_index_map, active_branches
        )

        # Step 2: Evaluate
        best_candidates, accept_lengths, sample_tokens = evaluate_parallel(
            logits, draft_tokens, retrieve_indices, logits_processor
        )

        # Step 3: Update
        draft_input_hidden = self.update_state_parallel(
            draft_tokens, draft_input_hidden, best_candidates, accept_lengths,
            sample_tokens, retrieve_indices, branch_index_map, active_branches, prefix_len
        )

        # Step 4: Draft
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(
                draft_input_hidden, sample_tokens,
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
        draft_input_hidden = self._update_state_batching(
            draft_tokens, hidden_states, best_candidates, accept_lengths,
            sample_tokens, retrieve_indices, active_branches
        )

        # Step 4: Draft
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(
                draft_input_hidden, sample_tokens,
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
        last_hidden_state = outputs["last_hidden_state"]  # [1, packed_draft_len, hidden_size]
        logits = self.base_model.lm_head(last_hidden_state)
        
        # 提取 Draft Model 的输入 hidden states
        eagle_full_hidden = self._prepare_draft_input_hidden(outputs)
        
        # 重新组织为分支维度
        logits = logits.view(num_branches, num_nodes, -1)
        hidden_dim = eagle_full_hidden.shape[-1]
        draft_input_hidden = eagle_full_hidden.view(num_branches, num_nodes, hidden_dim)
        
        return logits, draft_input_hidden
    
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
        
        draft_input_hidden_list = []
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
                # last_pos = path_indices[-1].item() if accept_len > 0 else 0
                # draft_input_hidden = hidden_states[i:i+1, last_pos:last_pos + 1, :]
                # 获取所有接受token的 hidden states
                draft_input_hidden = hidden_states[i:i+1, path_indices, :]
            else:
                # 如果没有接受任何 token，使用 root 的 hidden state
                draft_input_hidden = hidden_states[i:i+1, 0:1, :]
            
            draft_input_hidden_list.append(draft_input_hidden)
        
        # 拼接所有分支的 hidden states
        draft_input_hidden = torch.cat(draft_input_hidden_list, dim=0)

        return draft_input_hidden

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

        # 2. 使用 state_manager 准备所有数据（BIM, position_ids, attention_mask 等）
        if self.state_manager is None:
            raise RuntimeError("state_manager 未初始化")
        
        prefill_data = self.state_manager.prepare_parallel_prefill_bim(
            branch_prompts=branch_prompts,
            branch_ids=branch_ids,
            max_new_tokens=max_new_tokens,
        )

        input_ids = prefill_data['input_ids']
        branch_index_map = prefill_data['branch_index_map']
        position_ids = prefill_data['position_ids']
        attention_mask = prefill_data['attention_mask']
        tips_indices = prefill_data['tips_indices']
        branch_begins = prefill_data['branch_begins']
        branch_lengths = prefill_data['branch_lengths']

        # 3. Base Model Forward
        self.base_model.model.tree_mask = attention_mask
        self.base_model.model.tree_mode = "parallel_prefill"

        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]  # [1, total_len, hidden_size]

        # 提取 Draft Model 的输入 hidden states
        draft_input_hidden = self._prepare_draft_input_hidden(outputs)  # [1, total_len, hidden(*3)]

        # 5. 提取各分支 Tip 的 Hidden States
        tips_hidden = []
        for tip_idx in tips_indices:
            tips_hidden.append(draft_input_hidden[:, tip_idx:tip_idx+1, :])
        tips_hidden = torch.cat(tips_hidden, dim=1).transpose(0, 1)  # [num_branches, 1, hidden(*3)]

        # 注意：Eagle KV Cache 已在 state_manager.init_parallel_state 中初始化并复用 prefix

        # 6. 提取各分支的 tip token
        flat_branch_ids = []
        for prompt in branch_prompts:
            flat_branch_ids.extend(prompt)
        tip_token_ids = torch.tensor(
            [flat_branch_ids[begin + length - 1]
             for begin, length in zip(branch_begins, branch_lengths)],
            device=self.device
        ).unsqueeze(1)

        # 8. 生成 Draft Tree
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

        各分支独立 [num_branches, max_len]。
        注意：KV Cache 的扩展已在 state_manager.init_parallel_state 中完成。

        Args:
            prefix_len: 共享 prefix 长度
            branch_prompts: 各分支的 prompt token 列表
            branch_ids: 分支 ID 列表
            max_new_tokens: 最大生成 token 数

        Returns:
            包含 prefill 结果的字典
        """
        num_branches = len(branch_prompts)

        # 1. 使用 state_manager 准备所有数据
        # 注意：KV Cache 已在 generator 的 _parallel_prefill 中通过 
        # state_manager.init_parallel_state 完成初始化和扩展
        if self.state_manager is None:
            raise RuntimeError("state_manager 未初始化")

        prefill_data = self.state_manager.prepare_parallel_prefill_batching(
            branch_prompts=branch_prompts,
            branch_ids=branch_ids,
            pad_token_id=0,
        )

        padded_input = prefill_data['input_ids']
        position_ids = prefill_data['position_ids']
        padding_mask = prefill_data['padding_mask']
        attention_mask = prefill_data['attention_mask']
        branch_lengths = prefill_data['branch_lengths']

        max_branch_len = padded_input.shape[1]

        # 3. Base Model Forward
        outputs = self.base_model.model(
            input_ids=padded_input,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        last_hidden_state = outputs["last_hidden_state"]  # [num_branches, max_branch_len, hidden_size]

        # 提取 Draft Model 的输入 hidden states
        draft_input_hidden = self._prepare_draft_input_hidden(outputs)  # [num_branches, max_branch_len, hidden(*3)]

        # 5. 提取各分支 tip hidden states
        tips_hidden = []
        for i, length in enumerate(branch_lengths):
            tip_idx = max_branch_len - 1  # 最后一个位置
            tips_hidden.append(draft_input_hidden[i:i+1, tip_idx:tip_idx+1, :])
        tips_hidden = torch.cat(tips_hidden, dim=0)  # [num_branches, 1, hidden(*3)]

        # 注意：Eagle KV Cache 已在 state_manager.init_parallel_state 中初始化并复用 prefix

        # 6. 生成 Draft Tree
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
            hidden_states: Eagle 输入格式的隐藏状态 [num_branches, tree_size, hidden(*3)]
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
        last_hidden_state = outputs["last_hidden_state"]  # [num_branches, tree_size, hidden_size]
        logits = self.base_model.lm_head(last_hidden_state)

        # 提取 Draft Model 的输入 hidden states
        draft_input_hidden = self._prepare_draft_input_hidden(outputs)  # [num_branches, tree_size, hidden(*3)]

        return logits, draft_input_hidden

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
        draft_input_hidden_list = []
        prev_length = self.current_length_data[0].item()

        for i in range(num_branches):
            accept_len = accept_lengths[i].item()
            candidate_idx = best_candidates[i].item()

            if accept_len > 0:
                # 获取接受的 token 路径
                path_indices = retrieve_indices[i, candidate_idx, :accept_len + 1]
                path_indices = path_indices.flatten()  # 确保是 1D

                # 复制接受路径的 cache 到连续位置
                # absolute_indices: verify 输入中的位置 + prev_length = cache 中的绝对位置
                absolute_indices = path_indices + prev_length

                for layer_kv in self.past_key_values:
                    for kv_cache in layer_kv:
                        # Batching 模式：每个分支独立处理
                        # 提取接受路径的 cache 数据
                        src = kv_cache.data[i:i+1].index_select(2, absolute_indices)
                        # 复制到连续位置 [prev_length, prev_length + accept_len + 1)
                        dst_start = prev_length
                        dst_end = prev_length + len(path_indices)
                        kv_cache.data[i, :, dst_start:dst_end, :] = src.squeeze(0)

                # 获取最后位置的 hidden state
                # last_pos = path_indices[-1].item()
                # new_hidden = hidden_states[i:i+1, last_pos:last_pos+1, :]
                # 所有被接受的token的hidden states
                draft_input_hidden = hidden_states[i:i+1, path_indices, :]
            else:
                draft_input_hidden = hidden_states[i:i+1, 0:1, :]

            draft_input_hidden_list.append(draft_input_hidden)

        # 更新 cache 长度（取最大接受长度）
        max_accept = accept_lengths.max().item()
        self.current_length_data[0] = prev_length + max_accept + 1

        return torch.cat(draft_input_hidden_list, dim=0)

    # =========================================================================
    # Continuous Batching (混合 Prefill + Decode)
    # =========================================================================

    @torch.inference_mode()
    def continuous_decode_step_parallel_bim(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        prefilling_branches: List[int],
        pending_prefill_prompts: Dict[int, List[int]],
        prefix_len: int,
        logits_processor: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, List[int], Dict[int, List[int]]]:
        """
        带新分支 Prefill 的并行解码步骤（Continuous Batching 核心）- BIM 模式
        
        该方法同时处理：
        1. 老分支的 draft 验证（speculative decoding）
        2. 新分支的 prompt prefill
        
        流程：
        1. 构建混合输入：老分支的 draft tokens + 新分支的 prompt tokens
        2. 构建混合 attention mask（使用 state_manager）
        3. Base Model Forward（同时完成验证和 prefill）
        4. 处理老分支的验证结果
        5. 为新分支采样 root token
        6. 更新状态并生成下一轮 draft
        
        Args:
            draft_tokens: 老分支的 draft tokens [num_old_branches, num_nodes]
            retrieve_indices: 检索索引
            tree_mask: tree attention mask
            tree_position_ids: tree 位置编码
            branch_index_map: BIM 索引
            active_branches: 老分支 ID 列表
            prefilling_branches: 新分支 ID 列表
            pending_prefill_prompts: 新分支的 prompt tokens {bid: [token_ids]}
            prefix_len: 共享前缀长度
            logits_processor: logits 处理器
            
        Returns:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, 
            accept_lengths, all_finished, 
            updated_active_branches, branch_outputs
        """
        device = self.device

        # 获取分支信息
        num_old = len(active_branches)
        num_new = len(prefilling_branches)
        num_nodes = draft_tokens.shape[1] if draft_tokens is not None and num_old > 0 else 0

        branch_outputs = {}  # 记录每个分支的输出

        # Step 1: 构建混合输入
        combined_input, combined_positions, new_prompt_lengths, new_branch_bim = \
            self._build_mixed_input_continuous(
                draft_tokens, tree_position_ids, prefilling_branches,
                pending_prefill_prompts, prefix_len, num_old, num_nodes
            )

        if combined_input is None:
            return None, None, None, None, None, True, [], {}

        # Step 2: 构建 Attention Mask
        current_length = self.current_length_data[0].item()
        combined_mask = self._build_combined_mask_continuous(
            branch_index_map, active_branches, new_branch_bim,
            current_length, num_old, num_nodes, tree_mask, new_prompt_lengths
        )

        # Step 3: Base Model Forward
        self.base_model.model.tree_mask = None
        outputs = self.base_model.model(
            input_ids=combined_input,
            attention_mask=combined_mask,
            past_key_values=self.past_key_values,
            position_ids=combined_positions,
        )
        last_hidden_state = outputs["last_hidden_state"]  # [1, combined_len, hidden_size]
        all_logits = self.base_model.lm_head(last_hidden_state)

        # 提取 Draft Model 的输入 hidden states
        draft_input_hidden_full = self._prepare_draft_input_hidden(outputs)  # [1, combined_len, hidden(*3)]

        # Step 4: 处理老分支
        accept_lengths, next_draft_input_hidden_old, next_draft_input_tokens_old = \
            self._process_old_branches_continuous(
                all_logits, draft_input_hidden_full, draft_tokens, retrieve_indices,
                branch_index_map, active_branches, prefix_len,
                num_old, num_nodes, logits_processor, branch_outputs
            )

        # Step 5: 处理新分支
        new_draft_input_hidden_list, new_draft_input_tokens_list, active_branches = \
            self._process_new_branches_continuous(
                all_logits, draft_input_hidden_full, prefilling_branches,
                new_prompt_lengths, branch_index_map, active_branches,
                prefix_len, num_old, num_nodes, current_length, logits_processor
            )

        # Step 6: 生成下一轮 Draft
        return self._generate_next_draft_continuous(
            next_draft_input_hidden_old, next_draft_input_tokens_old,
            new_draft_input_hidden_list, new_draft_input_tokens_list,
            active_branches, prefix_len, accept_lengths, branch_outputs
        )

    # =========================================================================
    # Continuous Batching 辅助方法
    # =========================================================================

    def _build_mixed_input_continuous(
        self,
        draft_tokens: Optional[torch.Tensor],
        tree_position_ids: Optional[torch.Tensor],
        prefilling_branches: List[int],
        pending_prefill_prompts: Dict[int, List[int]],
        prefix_len: int,
        num_old: int,
        num_nodes: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int], List[int]]:
        """
        构建混合输入（老分支 draft + 新分支 prompt）

        Args:
            draft_tokens: 老分支的 draft tokens
            tree_position_ids: tree 位置编码
            prefilling_branches: 新分支 ID 列表
            pending_prefill_prompts: 新分支的 prompt tokens
            prefix_len: 共享前缀长度
            num_old: 老分支数量
            num_nodes: 每个老分支的节点数

        Returns:
            combined_input: 混合输入 tensor
            combined_positions: 混合位置编码
            new_prompt_lengths: 新分支 prompt 长度列表
            new_branch_bim: 新分支的 BIM 列表
        """
        device = self.device

        # 老分支的 draft tokens
        if num_old > 0 and draft_tokens is not None:
            flat_draft_tokens = draft_tokens.reshape(1, -1)
            current_tip_pos = self.eagle_layer.full_position_ids[:, -1].unsqueeze(-1)
            abs_draft_pos = tree_position_ids + current_tip_pos + 1
            flat_draft_pos = abs_draft_pos.view(1, -1)
        else:
            flat_draft_tokens = None
            flat_draft_pos = None

        # 新分支的 prompt tokens
        new_prompt_tensors = []
        new_prompt_positions = []
        new_branch_bim = []
        new_prompt_lengths = []

        for bid in prefilling_branches:
            prompt = pending_prefill_prompts[bid]
            prompt_len = len(prompt)
            new_prompt_lengths.append(prompt_len)

            prompt_tensor = torch.tensor(prompt, device=device, dtype=torch.long)
            new_prompt_tensors.append(prompt_tensor)

            pos = torch.arange(prefix_len, prefix_len + prompt_len, device=device)
            new_prompt_positions.append(pos)

            new_branch_bim.extend([bid] * prompt_len)

        # 拼接所有新分支的 prompt
        if new_prompt_tensors:
            flat_new_prompts = torch.cat(new_prompt_tensors).unsqueeze(0)
            flat_new_positions = torch.cat(new_prompt_positions).unsqueeze(0)
        else:
            flat_new_prompts = None
            flat_new_positions = None

        # 合并输入
        if flat_draft_tokens is not None and flat_new_prompts is not None:
            combined_input = torch.cat([flat_draft_tokens, flat_new_prompts], dim=1)
            combined_positions = torch.cat([flat_draft_pos, flat_new_positions], dim=1)
        elif flat_draft_tokens is not None:
            combined_input = flat_draft_tokens
            combined_positions = flat_draft_pos
        elif flat_new_prompts is not None:
            combined_input = flat_new_prompts
            combined_positions = flat_new_positions
        else:
            return None, None, [], []

        return combined_input, combined_positions, new_prompt_lengths, new_branch_bim

    def _build_combined_mask_continuous(
        self,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        new_branch_bim: List[int],
        current_length: int,
        num_old: int,
        num_nodes: int,
        tree_mask: Optional[torch.Tensor],
        new_prompt_lengths: List[int],
    ) -> torch.Tensor:
        """
        构建 Continuous Batching 的混合 Attention Mask

        Args:
            branch_index_map: BIM 索引
            active_branches: 老分支 ID 列表
            new_branch_bim: 新分支的 BIM 列表
            current_length: 当前 KV Cache 长度
            num_old: 老分支数量
            num_nodes: 每个老分支的节点数
            tree_mask: tree attention mask
            new_prompt_lengths: 新分支 prompt 长度列表

        Returns:
            combined_mask: 混合 attention mask
        """
        device = self.device
        history_bim = branch_index_map[:current_length]

        # 计算每个输入 token 的分支归属
        combined_bim = []
        if num_old > 0:
            for bid in active_branches:
                combined_bim.extend([bid] * num_nodes)
        combined_bim.extend(new_branch_bim)
        combined_bim_tensor = torch.tensor(combined_bim, device=device)

        # 使用 state_manager 构建 mask
        combined_mask = self.state_manager.build_continuous_decode_mask(
            history_bim=history_bim,
            combined_bim_tensor=combined_bim_tensor,
            current_length=current_length,
            num_old_branches=num_old,
            num_nodes=num_nodes,
            tree_mask=tree_mask if num_old > 0 else None,
            new_prompt_lengths=new_prompt_lengths,
        )

        return combined_mask

    def _process_old_branches_continuous(
        self,
        all_logits: torch.Tensor,
        eagle_hidden: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        prefix_len: int,
        num_old: int,
        num_nodes: int,
        logits_processor: Optional[Any],
        branch_outputs: Dict[int, List[int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        处理老分支的验证结果

        Args:
            all_logits: 所有 logits [1, combined_len, vocab_size]
            eagle_hidden: Eagle 输入格式的隐藏状态 [1, combined_len, hidden(*3)]
                         （EAGLE3: hidden*3, EAGLE2: hidden）
            draft_tokens: draft tokens
            retrieve_indices: 检索索引
            branch_index_map: BIM 索引
            active_branches: 老分支 ID 列表
            prefix_len: 共享前缀长度
            num_old: 老分支数量
            num_nodes: 每个老分支的节点数
            logits_processor: logits 处理器
            branch_outputs: 分支输出字典（会被修改）

        Returns:
            accept_lengths: 接受长度
            next_tips_hidden_old: 老分支的 tips hidden（Eagle 输入格式）
            next_tips_tokens_old: 老分支的 tips tokens
        """
        if num_old == 0 or draft_tokens is None:
            return None, None, None

        from .evaluator import evaluate_parallel

        # 提取老分支的 logits 和 hidden states
        old_total_len = num_old * num_nodes
        old_logits = all_logits[:, :old_total_len, :].view(num_old, num_nodes, -1)
        hidden_dim = eagle_hidden.shape[-1]
        old_hidden = eagle_hidden[:, :old_total_len, :].view(num_old, num_nodes, hidden_dim)

        # 评估（验证 draft tokens）
        best_candidates, accept_lengths, sample_tokens = evaluate_parallel(
            old_logits, draft_tokens, retrieve_indices, logits_processor
        )

        # 更新状态
        next_draft_input_hidden_old = self.update_state_parallel(
            draft_tokens, old_hidden, best_candidates, accept_lengths,
            sample_tokens, retrieve_indices, branch_index_map, active_branches, prefix_len
        )
        next_tips_tokens_old = sample_tokens

        # 记录接受的 tokens 到 branch_outputs
        for i, bid in enumerate(active_branches):
            accept_len = accept_lengths[i].item()
            if accept_len > 0:
                path_indices = retrieve_indices[i, best_candidates[i].item(), :accept_len + 1]
                accepted = draft_tokens[i, path_indices].tolist()
                branch_outputs[bid] = accepted

        return accept_lengths, next_draft_input_hidden_old, next_tips_tokens_old

    def _process_new_branches_continuous(
        self,
        all_logits: torch.Tensor,
        eagle_hidden: torch.Tensor,
        prefilling_branches: List[int],
        new_prompt_lengths: List[int],
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        prefix_len: int,
        num_old: int,
        num_nodes: int,
        current_length: int,
        logits_processor: Optional[Any],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        处理新分支的 Prefill 结果

        Args:
            all_logits: 所有 logits [1, combined_len, vocab_size]
            eagle_hidden: Eagle 输入格式的隐藏状态 [1, combined_len, hidden(*3)]
                         （EAGLE3: hidden*3, EAGLE2: hidden）
            prefilling_branches: 新分支 ID 列表
            new_prompt_lengths: 新分支 prompt 长度列表
            branch_index_map: BIM 索引
            active_branches: 当前活跃分支列表
            prefix_len: 共享前缀长度
            num_old: 老分支数量
            num_nodes: 每个老分支的节点数
            current_length: 当前 KV Cache 长度
            logits_processor: logits 处理器

        Returns:
            new_tips_hidden_list: 新分支的 tips hidden 列表（Eagle 输入格式）
            new_tips_tokens_list: 新分支的 tips tokens 列表
            updated_active_branches: 更新后的活跃分支列表
        """
        device = self.device
        num_new = len(prefilling_branches)

        if num_new == 0:
            return [], [], list(active_branches)

        new_draft_input_hidden_list = []
        new_draft_input_tokens_list = []

        old_total_len = num_old * num_nodes if num_old > 0 else 0

        # 新分支 KV Cache 位置搬移
        current_len_after_old = self.current_length_data[0].item()
        new_branches_kv_src_start = current_length + old_total_len
        new_branches_kv_dst_start = current_len_after_old
        total_new_prompt_len = sum(new_prompt_lengths)

        # 搬移新分支的 KV Cache
        if new_branches_kv_src_start != new_branches_kv_dst_start and total_new_prompt_len > 0:
            for layer_kv in self.past_key_values:
                for kv_cache in layer_kv:
                    src_kv = kv_cache.data[..., new_branches_kv_src_start:new_branches_kv_src_start + total_new_prompt_len, :].clone()
                    kv_cache.data[..., new_branches_kv_dst_start:new_branches_kv_dst_start + total_new_prompt_len, :].copy_(src_kv)

        # 提取每个新分支的结果
        offset = old_total_len
        bim_ptr = current_len_after_old

        for i, (bid, prompt_len) in enumerate(zip(prefilling_branches, new_prompt_lengths)):
            # 获取该分支最后一个 token 的 logits
            tip_logits = all_logits[0, offset + prompt_len - 1, :]

            # 采样 root token
            if logits_processor is not None:
                tip_logits = logits_processor(None, tip_logits.unsqueeze(0)).squeeze(0)
                probs = torch.nn.functional.softmax(tip_logits, dim=-1)
                root_token = torch.multinomial(probs, num_samples=1).item()
            else:
                root_token = torch.argmax(tip_logits).item()

            # 获取该分支最后一个 token 的 hidden state（已经是 Eagle 输入格式）
            branch_last_hidden = eagle_hidden[0, offset + prompt_len - 1, :].unsqueeze(0)

            # 准备 draft 输入
            new_draft_input_tokens_list.append(
                torch.tensor([root_token], device=device, dtype=torch.long)
            )
            new_draft_input_hidden_list.append(branch_last_hidden)

            # 更新 BIM
            bim_start = bim_ptr
            bim_end = bim_start + prompt_len
            branch_index_map[bim_start:bim_end] = bid
            bim_ptr = bim_end

            offset += prompt_len

        # 更新 current_length
        final_kv_len = current_len_after_old + total_new_prompt_len
        self.current_length_data.fill_(final_kv_len)

        # 将新分支加入活跃列表
        updated_active_branches = list(active_branches) + list(prefilling_branches)

        # 更新 Eagle Layer 状态
        self._update_eagle_for_new_branches(
            prefilling_branches, new_prompt_lengths, prefix_len
        )

        return new_draft_input_hidden_list, new_draft_input_tokens_list, updated_active_branches

    def _generate_next_draft_continuous(
        self,
        next_draft_input_hidden_old: Optional[torch.Tensor],
        next_draft_input_tokens_old: Optional[torch.Tensor],
        new_draft_input_hidden_list: List[torch.Tensor],
        new_draft_input_tokens_list: List[torch.Tensor],
        active_branches: List[int],
        prefix_len: int,
        accept_lengths: Optional[torch.Tensor],
        branch_outputs: Dict[int, List[int]],
    ) -> Tuple:
        """
        生成下一轮 Draft

        Args:
            next_draft_input_hidden_old: 老分支的 draft input hidden
            next_draft_input_tokens_old: 老分支的 draft input tokens
            new_draft_input_hidden_list: 新分支的 draft input hidden 列表
            new_draft_input_tokens_list: 新分支的 draft input tokens 列表
            active_branches: 活跃分支列表
            prefix_len: 共享前缀长度
            accept_lengths: 接受长度
            branch_outputs: 分支输出字典

        Returns:
            完整的返回元组
        """
        from ..utils.utils import stack_with_left_padding
        device = self.device

        all_finished = len(active_branches) == 0

        if all_finished:
            return None, None, None, None, accept_lengths, True, [], branch_outputs

        # 合并老分支和新分支的 hidden states 和 tokens
        all_draft_input_hidden_list = []
        all_draft_input_tokens_list = []

        if next_draft_input_hidden_old is not None and next_draft_input_tokens_old is not None:
            for i in range(next_draft_input_hidden_old.shape[0]):
                all_draft_input_hidden_list.append(next_draft_input_hidden_old[i:i+1])
                all_draft_input_tokens_list.append(next_draft_input_tokens_old[i:i+1])

        if new_draft_input_hidden_list:
            for hidden, tokens in zip(new_draft_input_hidden_list, new_draft_input_tokens_list):
                all_draft_input_hidden_list.append(hidden.unsqueeze(0))
                all_draft_input_tokens_list.append(tokens.unsqueeze(0))

        if not all_draft_input_hidden_list:
            return None, None, None, None, accept_lengths, True, active_branches, branch_outputs

        # 对齐并堆叠
        batched_hidden = stack_with_left_padding(all_draft_input_hidden_list, pad_id=0, device=device)
        batched_tokens = stack_with_left_padding(all_draft_input_tokens_list, pad_id=0, device=device)

        # 生成下一轮 Draft
        new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids = \
            self.drafter.generate_draft_tree(
                batched_hidden, batched_tokens,
                prefix_len=prefix_len,
                active_branch=active_branches,
            )

        return (
            new_draft_tokens, new_retrieve_indices, new_tree_mask, new_tree_position_ids,
            accept_lengths, False, active_branches, branch_outputs
        )

    def _update_eagle_for_new_branches(
        self,
        new_branch_ids: List[int],
        new_prompt_lengths: List[int],
        prefix_len: int,
    ):
        """
        更新 Eagle Layer 状态以支持新分支
        
        Args:
            new_branch_ids: 新分支 ID 列表
            new_prompt_lengths: 新分支 prompt 长度列表
            prefix_len: 共享 prefix 长度
        """
        device = self.device
        num_new = len(new_branch_ids)
        
        if num_new == 0:
            return
        
        # 构建新分支的 cache_padding_mask 和 full_position_ids
        new_mask_list = []
        new_pos_list = []
        
        for prompt_len in new_prompt_lengths:
            total_len = prefix_len + prompt_len
            mask = torch.ones(total_len, device=device, dtype=torch.long)
            new_mask_list.append(mask)
            pos = torch.arange(total_len, device=device, dtype=torch.long)
            new_pos_list.append(pos)
        
        # 对齐到老分支的长度
        if self.eagle_layer.cache_padding_mask is not None:
            old_len = self.eagle_layer.cache_padding_mask.shape[1]
            
            padded_masks = []
            padded_positions = []
            for mask, pos in zip(new_mask_list, new_pos_list):
                cur_len = len(mask)
                if cur_len < old_len:
                    # 左填充到相同长度
                    pad_len = old_len - cur_len
                    mask = torch.cat([torch.zeros(pad_len, device=device, dtype=mask.dtype), mask])
                    pos = torch.cat([torch.zeros(pad_len, device=device, dtype=pos.dtype), pos])
                elif cur_len > old_len:
                    # 老分支需要右填充
                    pad_len = cur_len - old_len
                    old_mask_padded = torch.cat([
                        self.eagle_layer.cache_padding_mask,
                        torch.zeros(self.eagle_layer.cache_padding_mask.shape[0], pad_len, device=device, dtype=self.eagle_layer.cache_padding_mask.dtype)
                    ], dim=1)
                    old_pos_padded = torch.cat([
                        self.eagle_layer.full_position_ids,
                        torch.zeros(self.eagle_layer.full_position_ids.shape[0], pad_len, device=device, dtype=self.eagle_layer.full_position_ids.dtype)
                    ], dim=1)
                    self.eagle_layer.cache_padding_mask = old_mask_padded
                    self.eagle_layer.full_position_ids = old_pos_padded
                    old_len = cur_len
                padded_masks.append(mask)
                padded_positions.append(pos)
            
            new_cache_mask = torch.stack(padded_masks, dim=0)
            new_full_pos = torch.stack(padded_positions, dim=0)
            
            self.eagle_layer.cache_padding_mask = torch.cat([
                self.eagle_layer.cache_padding_mask, new_cache_mask
            ], dim=0)
            self.eagle_layer.full_position_ids = torch.cat([
                self.eagle_layer.full_position_ids, new_full_pos
            ], dim=0)
        
        # 扩展 Eagle Layer KV Cache
        if self.eagle_layer.draft_past_key_values is not None:
            key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
            
            if hasattr(key_cache, 'data'):
                k_draft = key_cache.data[:, :, :key_cache.shape[2], :]
                v_draft = value_cache.data[:, :, :value_cache.shape[2], :]
            else:
                k_draft = key_cache
                v_draft = value_cache
            
            old_kv_seq_len = k_draft.shape[2]
            
            # 新分支从 prefix 开始
            k_prefix = k_draft[:1, :, :prefix_len, :].clone()
            v_prefix = v_draft[:1, :, :prefix_len, :].clone()
            
            # 扩展到与老分支相同的序列长度
            if old_kv_seq_len > prefix_len:
                pad_len = old_kv_seq_len - prefix_len
                k_pad = torch.zeros(
                    1, k_prefix.shape[1], pad_len, k_prefix.shape[3],
                    device=device, dtype=k_prefix.dtype
                )
                v_pad = torch.zeros(
                    1, v_prefix.shape[1], pad_len, v_prefix.shape[3],
                    device=device, dtype=v_prefix.dtype
                )
                k_prefix = torch.cat([k_prefix, k_pad], dim=2)
                v_prefix = torch.cat([v_prefix, v_pad], dim=2)
            
            k_expanded = k_prefix.expand(num_new, -1, -1, -1).clone()
            v_expanded = v_prefix.expand(num_new, -1, -1, -1).clone()
            
            k_combined = torch.cat([k_draft, k_expanded], dim=0)
            v_combined = torch.cat([v_draft, v_expanded], dim=0)
            
            self.eagle_layer.draft_past_key_values = ((k_combined, v_combined),)
