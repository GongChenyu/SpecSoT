# coding=utf-8
"""
SpecSoT 工具函数模块 (Utils)

该模块提供推理过程中的各种工具函数，按功能分类组织：

1. Logits Processing (Logits 处理)
   - prepare_logits_processor: 准备温度、top-p、top-k 等 logits 处理器

2. Single Mode Helpers (单序列模式辅助函数)
   - prefill_single: 单序列 Prefill 阶段（处理输入 prompt，初始化 KV Cache 和首次 Draft）
   - verify_step_single: 单序列 Verify 阶段（Base Model 验证 Draft Tree）
   - update_inference_inputs: 单序列状态更新

3. Parallel Mode Helpers (并行模式辅助函数)
   - prefill_parallel: 并行 Prefill 阶段（处理各分支 prompt，初始化并行状态和首次 Draft）
   - verify_step_parallel: 并行 Verify 阶段（Base Model 验证多分支 Draft Tree）

4. Evaluation (评估)
   - evaluate_posterior: 评估候选序列，选择最佳路径
   - _evaluate_posterior_single: 单样本评估 (带 rejection sampling)

5. State Utilities (状态工具)
   - reset_tree_mode: 重置 tree mode
   - reset_past_key_values: 重置 KV Cache

6. Tensor Utilities (张量工具)
   - stack_with_left_padding: 左填充堆叠不等长序列
   - generate_candidates: 生成候选序列
   - pad_path: 路径填充

7. Skeleton Parsing (骨架解析)
   - parse_skeleton: 解析骨架，提取并行分支
"""

import copy
import random
from typing import List, Tuple, Optional, Dict

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .prompts import parallel_trigger_zh


# =============================================================================
# 1. Logits Processing (Logits 处理)
# =============================================================================

def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    """
    准备 logits 处理器列表
    
    Args:
        temperature: 采样温度 (0 表示 greedy)
        repetition_penalty: 重复惩罚系数
        top_p: nucleus sampling 阈值
        top_k: top-k sampling 数量
        
    Returns:
        LogitsProcessorList: 处理器列表
    """
    processor_list = LogitsProcessorList()
    
    if temperature > 1e-5:
        if temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
            
    return processor_list


# =============================================================================
# 2. Single Mode Helpers (单序列模式辅助函数)
# =============================================================================

def prefill_single(
    input_ids: torch.Tensor,
    model,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    单序列模式：Prefill 阶段（处理输入 prompt，初始化 KV Cache 和首次 Draft）
    
    流程：
    1. Base Model Prefill: 处理输入，获取最后一个 token 的 logits
    2. Sample Root Token: 采样第一个生成的 token
    3. Generate Draft Tree: 使用 Eagle Layer 生成候选树
    
    Args:
        input_ids: 输入 token IDs [1, seq_len]
        model: SpecSoT 模型
        past_key_values: KV Cache
        logits_processor: logits 处理器
        
    Returns:
        draft_tokens: 候选 tokens [1, tree_size]
        retrieve_indices: 检索索引 [num_leaves, depth]
        tree_mask: 树掩码 [1, 1, tree_size, tree_size]
        tree_position_ids: 位置编码 [1, tree_size]
        orig: 原始 logits
        hidden_states: 隐藏状态
        token: 采样的 root token
    """
    # Base Model Prefill
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=model.past_key_values, output_orig=True
    )

    # Sample Root Token
    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(input_ids, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Prepare Hidden States for Eagle Layer
    if model.use_eagle3:
        ea_device = model.eagle_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states = torch.cat(outputs["hidden_states"], dim=-1)

    # Generate Draft Tree
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
        model.eagle_layer.generate_draft_tree(hidden_states, input_ids)

    return (
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
        orig, hidden_states, token
    )


# =============================================================================
# 3. Parallel Mode Helpers (并行模式辅助函数)
# =============================================================================

def build_parallel_prefill_mask(
    branch_index_map: torch.Tensor,
    prefix_len: int,
    branch_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构建并行 Prefill 阶段的注意力掩码
    
    该掩码确保：
    1. 所有分支都能看到共享的 prefix
    2. 每个分支只能看到自己的内容
    3. 遵循因果约束（当前位置只能看到之前的位置）
    
    Args:
        branch_index_map: 分支索引映射 (BIM)
        prefix_len: 共享前缀长度
        branch_len: 分支区域长度
        device: 目标设备
        dtype: 数据类型
        
    Returns:
        attention_mask: [1, 1, branch_len, total_len]
    """
    total_len = prefix_len + branch_len

    total_ids = branch_index_map[:total_len]
    branch_ids = branch_index_map[prefix_len:total_len]

    # 初始化为全部遮蔽
    mask = torch.full(
        (1, 1, branch_len, total_len),
        torch.finfo(dtype).min, device=device
    )

    # 1. Prefix 全部可见 (BIM == -1)
    is_prefix = (total_ids == -1).unsqueeze(0)
    mask.masked_fill_(is_prefix, 0)

    # 2. 同分支可见 + 因果约束
    branch_ids_view = branch_ids.unsqueeze(1)  # [branch_len, 1]
    total_ids_view = total_ids.unsqueeze(0)     # [1, total_len]
    block_mask = (branch_ids_view == total_ids_view)

    branch_idx = torch.arange(prefix_len, total_len, device=device).unsqueeze(1)
    total_idx = torch.arange(total_len, device=device).unsqueeze(0)
    causal_mask = (total_idx <= branch_idx)

    valid_mask = block_mask & causal_mask
    mask.masked_fill_(valid_mask, 0)

    return mask


def prefill_parallel(
    prefix_len: int,
    input_ids: torch.Tensor,
    model,
    tips_indices: torch.Tensor,
    branch_begins: List[int],
    branch_lengths: List[int],
    draft_input_ids: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    并行模式：Prefill 阶段（处理各分支 prompt，初始化并行状态和首次 Draft）
    
    处理多个分支的并行 Prefill，共享 prefix 的 KV Cache。
    
    Args:
        prefix_len: 共享前缀长度
        input_ids: 打包后的输入 [1, total_len]
        model: SpecSoT 模型
        tips_indices: 各分支 tip 位置
        branch_begins: 各分支起始位置
        branch_lengths: 各分支长度
        draft_input_ids: Draft 模型输入 [num_para, max_len]
        logits_processor: logits 处理器
        
    Returns:
        input_ids: 更新后的输入
        draft_tokens: 候选 tokens
        retrieve_indices: 检索索引
        tree_mask: 树掩码
        tree_position_ids: 位置编码
        hidden_states: 隐藏状态
    """
    device = input_ids.device
    num_para = len(branch_lengths)

    # 构建并行 Prefill 的 Attention Mask
    attention_mask = build_parallel_prefill_mask(
        model.branch_index_map,
        prefix_len,
        branch_len=input_ids.shape[1] - prefix_len,
        device=device,
        dtype=torch.float32,
    )

    # 从 KV Cache 中获取已处理的长度
    kv_len = model.past_key_values[0][0].shape[2]
    position_ids = model.full_position_ids

    # Base Model Forward (Prefill)
    outputs, hidden_states = model(
        input_ids=input_ids[:, kv_len:],
        attention_mask=attention_mask,
        position_ids=position_ids[kv_len:],
        past_key_values=model.past_key_values,
        output_orig=False,
    )

    # 提取各分支 Tip 的 Logits
    tips_hidden = hidden_states[:, tips_indices - kv_len, :]
    tips_logits = model.base_model.lm_head(tips_hidden)
    current_logits = tips_logits.squeeze(0)  # [num_para, vocab]

    # Sample Root Tokens
    if logits_processor is not None:
        current_logits = logits_processor(None, current_logits)
        probs = torch.nn.functional.softmax(current_logits, dim=-1)
        root_tokens = torch.multinomial(probs, num_samples=1)
    else:
        root_tokens = torch.argmax(current_logits, dim=-1, keepdim=True)

    # 准备 Draft 模型输入
    draft_input_ids = torch.cat([draft_input_ids, root_tokens], dim=1)

    # 处理 Hidden States for Eagle Layer
    ea_device = model.eagle_layer.lm_head.weight.device
    if outputs["hidden_states"][0].device != ea_device:
        outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
    packed_hidden = torch.cat(outputs["hidden_states"], dim=-1)[0]

    # 提取各分支的 Hidden States
    branch_hidden_list = []
    for i in range(num_para):
        start = branch_begins[i]
        end = start + branch_lengths[i]
        branch_hidden_list.append(packed_hidden[start - prefix_len:end - prefix_len])
    
    batched_hidden = stack_with_left_padding(branch_hidden_list, pad_id=0, device=device)

    # Generate Draft Tree
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
        model.eagle_layer.generate_draft_tree(batched_hidden, draft_input_ids, prefix_len=prefix_len)

    return (
        input_ids, draft_tokens, retrieve_indices, tree_mask,
        tree_position_ids, hidden_states
    )


# =============================================================================
# 3. Verification (验证)
# =============================================================================

def verify_step_single(
    model,
    tree_candidates: torch.Tensor,
    past_key_values,
    tree_position_ids: torch.Tensor,
    input_ids: torch.Tensor,
    retrieve_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    单序列模式：Verify 步骤 (Base Model 验证)
    
    Args:
        model: SpecSoT 模型
        tree_candidates: 候选 tokens [1, tree_size]
        past_key_values: KV Cache
        tree_position_ids: 树位置编码 [1, tree_size]
        input_ids: 当前输入 [1, seq_len]
        retrieve_indices: 检索索引 [num_leaves, depth]
        
    Returns:
        logits: 验证后的 logits [num_leaves, depth, vocab]
        hidden_state: 隐藏状态
        outputs: 模型原始输出
    """
    # 计算绝对位置
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    # Base Model Forward
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # 处理 Hidden States for Eagle Layer (if Eagle3)
    if model.use_eagle3:
        ea_device = model.eagle_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    # 按检索索引重组 logits
    logits = tree_logits[0, retrieve_indices]
    
    return logits, hidden_state, outputs


def verify_step_parallel(
    model,
    draft_tokens: torch.Tensor,
    tree_position_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    num_nodes: int,
    branch_index_map: torch.Tensor,
    active_branches: List[int],
    eagle_full_position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    并行模式：Verify 步骤 (Parallel Base Model 验证)
    
    该函数包含并行验证掩码构建和 Base Model 前向传播两个步骤。
    
    Args:
        model: SpecSoT 模型
        draft_tokens: Draft tokens [num_para, num_nodes]
        tree_position_ids: Tree 位置编码 [num_para, num_nodes]
        tree_mask: Draft Tree 的掩码 [num_para, 1, num_nodes, num_nodes]
        num_nodes: 每个分支的 Draft 节点数
        branch_index_map: 分支索引映射
        active_branches: 当前活跃的分支列表
        eagle_full_position_ids: Eagle Layer 的完整位置编码
        
    Returns:
        logits: [num_para, num_nodes, vocab_size]
        hidden_states: 隐藏状态
    """
    device = draft_tokens.device
    num_para = draft_tokens.shape[0]
    current_length = model.current_length_data[0].item()
    
    # =========================================================================
    # Step 1: 构建并行验证的注意力掩码
    # =========================================================================
    history_bim = branch_index_map[:current_length]
    packed_draft_len = num_para * num_nodes
    
    # 初始化 Cross Mask (全部遮蔽)
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
    
    # 同分支可见
    draft_ids_view = draft_branch_ids.view(1, 1, -1, 1)
    hist_ids_view = history_bim.view(1, 1, 1, -1)
    is_same_branch = (draft_ids_view == hist_ids_view)
    cross_mask.masked_fill_(is_same_branch, 0)
    
    # 构建 Draft Block Mask (块对角)
    converted_tree_mask = torch.where(
        tree_mask == 1, 0.0, torch.finfo(torch.float32).min
    )
    draft_block_mask = torch.full(
        (packed_draft_len, packed_draft_len),
        torch.finfo(torch.float32).min, device=device
    )
    for i in range(num_para):
        st, ed = i * num_nodes, (i + 1) * num_nodes
        draft_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :]

    draft_block_mask = draft_block_mask.unsqueeze(0).unsqueeze(0)
    
    # 合并
    combined_mask = torch.cat([cross_mask, draft_block_mask], dim=-1)
    
    # =========================================================================
    # Step 2: Base Model Forward
    # =========================================================================
    flat_draft_tokens = draft_tokens.reshape(1, -1)
    
    # 计算绝对位置
    current_tip_pos = eagle_full_position_ids[:, -1].unsqueeze(-1)
    abs_draft_pos = tree_position_ids + current_tip_pos + 1
    flat_draft_pos = abs_draft_pos.view(1, -1)
    
    # Base Model Forward
    outputs, hidden_states = model(
        flat_draft_tokens,
        past_key_values=model.past_key_values,
        attention_mask=combined_mask,
        position_ids=flat_draft_pos,
        output_orig=False,
    )
    
    # 计算 Logits
    logits = model.base_model.lm_head(hidden_states)
    logits = logits.view(num_para, num_nodes, -1)
    
    # 处理 Hidden States for Eagle Layer
    ea_device = model.eagle_layer.lm_head.weight.device
    if outputs["hidden_states"][0].device != ea_device:
        outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
    hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
    
    return logits, hidden_states


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    para_token_ids: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    评估候选序列，选择最佳路径
    
    支持两种模式：
    1. Greedy (logits_processor=None): 直接比较 argmax
    2. Sampling (logits_processor!=None): 使用 rejection sampling
    
    Args:
        logits: 验证 logits [batch, num_paths, seq_len, vocab]
        candidates: 候选 tokens [batch, num_paths, seq_len]
        logits_processor: logits 处理器
        para_token_ids: 特殊 token IDs (用于骨架模式)
        
    Returns:
        best_candidate: 最佳候选索引 [batch]
        accept_length: 接受长度 [batch]
        sample_logits: 用于采样下一 token 的 logits [batch, vocab]
    """
    # 检测是否有语义约束处理器
    has_semantic_processor = False
    special_tokens = None
    
    if logits_processor is not None:
        for processor in logits_processor:
            if "SemanticLogitsProcessor" in processor.__class__.__name__:
                has_semantic_processor = True
                if para_token_ids:
                    special_tokens = [
                        para_token_ids['ellipsis_token_id'],
                        para_token_ids['line_break_token_id'],
                        para_token_ids['para_begin_token_id'],
                    ]
                break

    batch_size = logits.shape[0]

    # =========================================================================
    # Greedy Mode
    # =========================================================================
    if logits_processor is None:
        # candidates[:, :, 1:] 因为第一个是 root (已知正确)
        # logits[:, :, :-1] 预测的是下一个位置
        posterior_mask = (
            candidates[:, :, 1:].to(logits.device) == torch.argmax(logits[:, :, :-1, :], dim=-1)
        ).int()
        
        # 累积接受长度
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
        accept_length, best_candidate = candidates_accept_length.max(dim=-1)

        # 获取最佳路径的采样 logits
        batch_indices = torch.arange(batch_size, device=logits.device)
        best_path_logits = logits[batch_indices, best_candidate, :, :]
        
        seq_len = best_path_logits.shape[1]
        next_token_pos = accept_length.clamp(max=seq_len - 1)
        sample_logits = best_path_logits[batch_indices, next_token_pos]
        
        return best_candidate, accept_length, sample_logits

    # =========================================================================
    # Sampling Mode (with logits_processor)
    # =========================================================================
    best_candidates = []
    accept_lengths = []
    sample_logits_list = []

    for b in range(batch_size):
        bc, al, sl = _evaluate_posterior_single(
            logits[b], candidates[b], logits_processor,
            has_semantic_processor, special_tokens
        )
        best_candidates.append(bc)
        accept_lengths.append(al)
        sample_logits_list.append(sl)

    best_candidate = torch.stack(best_candidates)
    accept_length = torch.stack(accept_lengths)
    sample_logits = torch.stack(sample_logits_list)

    return best_candidate, accept_length, sample_logits


def _evaluate_posterior_single(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
    has_semantic_processor: bool,
    special_tokens: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    单样本评估（带 rejection sampling）
    
    逐 token 进行验证，使用 rejection sampling 决定是否接受。
    
    Args:
        logits: [num_paths, seq_len, vocab]
        candidates: [num_paths, seq_len]
        logits_processor: logits 处理器
        has_semantic_processor: 是否有语义处理器
        special_tokens: 特殊 token 列表
        
    Returns:
        best_candidate: 最佳候选索引
        accept_length: 接受长度 (不含 root)
        sample_logits: 采样 logits [vocab]
    """
    device = logits.device
    num_paths, seq_len = candidates.shape

    # 初始化：接受 root (位置 0)
    accept_cand = candidates[0, :1]  # [1]
    accept_length = 1
    best_candidate = 0
    active_mask = torch.ones(num_paths, dtype=torch.bool, device=device)

    for i in range(1, seq_len):
        # 找到与已接受前缀匹配的路径
        prefix_match = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        active_mask = active_mask & prefix_match

        if not active_mask.any():
            break

        # 获取第一个匹配路径来计算处理后的 logits
        fi = torch.nonzero(active_mask, as_tuple=True)[0][0].item()
        
        current_logits_input = logits[fi, i - 1].unsqueeze(0)
        current_context = candidates[fi, :i].unsqueeze(0)
        
        gt_logits = logits_processor(current_context, current_logits_input)[0]
        gtp = torch.softmax(gt_logits, dim=0)

        # 收集活跃路径在位置 i 的候选 tokens
        candidate_tokens = candidates[active_mask, i]
        unique_tokens = torch.unique(candidate_tokens)
        accepted = False

        for tok in unique_tokens:
            xi = tok.item()
            if xi == -1:
                continue

            r = random.random()
            px = gtp[xi].item()
            acp = min(1.0, px) if px > 0 else 0.0

            if r <= acp:
                # Accept
                accept_cand = torch.cat([accept_cand, tok.unsqueeze(0)], dim=0)
                accept_length += 1
                tok_mask = (candidates[:, i] == tok) & active_mask
                best_candidate = torch.nonzero(tok_mask, as_tuple=True)[0][0].item()
                accepted = True
                break
            else:
                # Reject: 对于语义处理器，只 mask 非特殊 token
                if has_semantic_processor and special_tokens:
                    if xi not in special_tokens:
                        gtp[xi] = 0.0
                else:
                    gtp[xi] = 0.0
                
                gtp_sum = gtp.sum()
                if gtp_sum > 0:
                    gtp = gtp / gtp_sum
                else:
                    break

        if not accepted:
            break

    # 计算最终采样 logits
    if accept_length < seq_len:
        final_logits = logits[best_candidate, accept_length - 1].unsqueeze(0)
        final_context = candidates[best_candidate, :accept_length].unsqueeze(0)
        processed = logits_processor(final_context, final_logits)[0]
    else:
        final_logits = logits[best_candidate, -1].unsqueeze(0)
        final_context = candidates[best_candidate, :].unsqueeze(0)
        processed = logits_processor(final_context, final_logits)[0]

    sample_logits = torch.softmax(processed, dim=0)

    return (
        torch.tensor(best_candidate, device=device),
        torch.tensor(accept_length - 1, device=device),  # 不含 root
        sample_logits,
    )


# =============================================================================
# 4. State Update (状态更新)
# =============================================================================

def update_inference_inputs(
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accept_length: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    model,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    更新推理状态（单序列模式）
    
    操作：
    1. 将接受的 tokens 添加到 input_ids
    2. 更新 KV Cache (搬运接受的 KV)
    3. 采样 Bonus Token
    4. 提取接受路径的 Hidden States (供上层生成 Draft Tree)
    
    Args:
        input_ids: 当前输入 [1, seq_len]
        candidates: 候选 tokens [num_leaves, depth]
        best_candidate: 最佳候选索引 [1]
        accept_length: 接受长度 [1]
        retrieve_indices: 检索索引 [num_leaves, depth]
        logits_processor: logits 处理器
        model: SpecSoT 模型
        hidden_state_new: 新的隐藏状态
        sample_p: 采样概率分布
        
    Returns:
        input_ids: 更新后的输入
        draft_input_ids: Draft 模型的输入 (含 Bonus Token)
        accept_hidden: 接受路径的隐藏状态 (供生成下一轮 Draft Tree)
    """
    prev_input_len = input_ids.shape[1]
    bc = best_candidate[0]
    al = accept_length[0]
    
    # 提取接受的 tokens
    new_tokens = candidates[0, bc, :al + 1].unsqueeze(0)
    select_indices = retrieve_indices[0, bc, :al + 1] + prev_input_len

    # 更新 input_ids
    input_ids = torch.cat([input_ids, new_tokens.to(input_ids.device)], dim=-1)

    # 更新 KV Cache (搬运接受的 KV 到正确位置)
    for past_kv_data in model.past_key_values_data:
        tgt = past_kv_data.index_select(dim=-2, index=select_indices.to(past_kv_data.device))
        dst = past_kv_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        model.current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # 采样 Bonus Token
    if logits_processor is not None:
        token = torch.multinomial(sample_p, 1)
        token = token[None] if token.ndim == 1 else token
    else:
        token = torch.argmax(sample_p, dim=-1, keepdim=True)
    
    draft_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # 提取接受路径的 Hidden States
    retrieve_hidden = hidden_state_new[:, retrieve_indices[0]]
    accept_hidden = retrieve_hidden[:, best_candidate[0], :accept_length[0] + 1]

    return input_ids, draft_input_ids, accept_hidden


def reset_tree_mode(model):
    """重置 Base Model 的 tree mode"""
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    重置 KV Cache 长度为零
    
    Args:
        passed_key_values: KV Cache 列表
        
    Returns:
        重置后的 KV Cache 列表
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


# =============================================================================
# 5. Utility Functions (工具函数)
# =============================================================================

def stack_with_left_padding(
    tensor_list: List[torch.Tensor],
    pad_id: int,
    device: torch.device,
    return_mask: bool = False,
) -> torch.Tensor:
    """
    将不等长的 Tensor 列表堆叠为 Batch，使用左填充
    
    支持 1D (tokens) 和 2D (hidden states) 输入。
    
    Args:
        tensor_list: Tensor 列表
        pad_id: 填充值
        device: 目标设备
        return_mask: 是否返回填充掩码
        
    Returns:
        padded_tensor: 填充后的 Tensor [batch, max_len, ...]
        padding_mask (可选): 填充掩码 [batch, max_len]
    """
    if not tensor_list:
        return None

    batch_size = len(tensor_list)
    max_len = max(t.size(0) for t in tensor_list)
    trailing_dims = list(tensor_list[0].shape[1:])
    target_shape = [batch_size, max_len] + trailing_dims

    padded_tensor = torch.full(
        target_shape, pad_id, dtype=tensor_list[0].dtype, device=device
    )

    if return_mask:
        padding_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, t in enumerate(tensor_list):
        length = t.size(0)
        start_idx = max_len - length
        padded_tensor[i, start_idx:] = t
        if return_mask:
            padding_mask[i, start_idx:] = 1

    if return_mask:
        return padded_tensor, padding_mask
    return padded_tensor


def generate_candidates(
    tree_logits: torch.Tensor,
    tree_indices: torch.Tensor,
    retrieve_indices: torch.Tensor,
    sample_token: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成候选序列
    
    Args:
        tree_logits: 树 logits
        tree_indices: 树索引
        retrieve_indices: 检索索引
        sample_token: 采样的 root token
        logits_processor: logits 处理器
        
    Returns:
        cart_candidates: 笛卡尔积候选
        tree_candidates: 树候选
    """
    sample_token = sample_token.to(tree_indices.device)
    candidates_logit = sample_token[0]
    candidates_tree_logits = tree_logits
    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]
    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1],
        dim=0,
    )

    cart_candidates = tree_candidates_ext[retrieve_indices]
    tree_candidates = tree_candidates.unsqueeze(0)
    
    return cart_candidates, tree_candidates


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    将路径列表填充到指定长度
    
    Args:
        path: 原始路径
        length: 目标长度
        pad_value: 填充值
        
    Returns:
        填充后的路径
    """
    return path + [pad_value] * (length - len(path))


def check_stop_conditions(
    input_ids: torch.Tensor,
    input_len: int,
    stop_token_id: Optional[int],
    eos_token_id: int,
    current_length: int,
    max_kv_len: int,
) -> bool:
    """
    检查停止条件
    
    Args:
        input_ids: 当前生成的 token IDs [1, total_len]
        input_len: 原始输入长度
        stop_token_id: 停止 token ID (可选)
        eos_token_id: EOS token ID
        current_length: 当前 KV Cache 长度
        max_kv_len: 最大 KV Cache 长度
        
    Returns:
        是否应该停止生成
    """
    generated_tokens = input_ids[0, input_len:].tolist()
    
    if stop_token_id and stop_token_id in generated_tokens:
        return True
    if eos_token_id in generated_tokens:
        return True
    if current_length >= max_kv_len - 200:
        return True
    
    return False


def parse_skeleton(
    tokenizer,
    skeleton_ids: torch.Tensor,
    para_token_ids: Dict[str, int],
) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    """
    解析骨架，提取并行分支
    
    骨架格式示例：
    ####标题1(100):...
    ####标题2(200):...
    ####%%%%
    
    Args:
        skeleton_ids: 骨架 token IDs
        para_token_ids: 特殊 token IDs 字典
        
    Returns:
        clean_branches: 清洗后的分支列表（含指令前缀）
        instruction_len: 每个分支的指令长度
    """
    seq_list = skeleton_ids[0].tolist()
    para_begin_id = para_token_ids['para_begin_token_id']
    para_end_id = para_token_ids['para_end_token_id']
    
    colon_ids = [
        para_token_ids['colon_token_id'],
        para_token_ids['cn_colon_token_id'],
        para_token_ids['colon_new_line_token_id'],
    ]

    # 查找骨架边界
    try:
        para_begin_idx = seq_list.index(para_begin_id)
        try:
            para_end_idx = seq_list.index(para_end_id, para_begin_idx)
        except ValueError:
            para_end_idx = len(seq_list)
    except ValueError:
        print("Warning: No '####' found in generated output.")
        return None, None

    # 提取并行片段
    para_segment = seq_list[para_begin_idx:para_end_idx - 1]

    # 分割分支
    raw_branches = []
    current_branch = []
    for token in para_segment:
        if token == para_begin_id:
            if current_branch:
                raw_branches.append(current_branch)
            current_branch = [token]
        else:
            current_branch.append(token)
    if current_branch:
        raw_branches.append(current_branch)

    # 清洗分支（截取到冒号）
    clean_branches = []
    for br in raw_branches:
        cut_idx = -1
        for i, token in enumerate(br):
            if token in colon_ids:
                cut_idx = i
                break
        
        if cut_idx != -1:
            clean_branches.append(br[:cut_idx + 1])
        else:
            clean_branches.append(br)

    # 构建骨架上下文
    result_skeleton = []
    for br in clean_branches:
        result_skeleton.extend(br)
    result_skeleton_str = tokenizer.decode(result_skeleton)

    # 为每个分支添加指令前缀
    instruction_len = []
    for i, br in enumerate(clean_branches):
        branch_str = tokenizer.decode(br)
        instruction = parallel_trigger_zh.format(
            skeleton_context=result_skeleton_str,
            current_point=branch_str
        )
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        clean_branches[i] = instruction_ids + br
        instruction_len.append(len(instruction_ids))

    return clean_branches, instruction_len



