# coding=utf-8
"""
SpecSoT 工具函数模块 (Utils)

该模块提供推理过程中的各种工具函数，按功能分类组织：

1. Logits Processing (Logits 处理)
   - prepare_logits_processor: 准备温度、top-p、top-k 等 logits 处理器
   - create_skeleton_logits_processor: 创建骨架生成的 logits 处理器

2. Mask Building (掩码构建)
   - build_parallel_prefill_mask: 构建并行 Prefill 阶段的注意力掩码

3. Evaluation (评估)
   - evaluate_single: 单序列评估 (Eagle 投机解码、SpecSoT Skeleton 阶段)
   - evaluate_parallel: 并行评估 (SpecSoT 并行分支解码阶段)
   - greedy_sampling: Greedy 采样算法
   - rejection_sampling: Rejection Sampling 采样算法
   - batched_sampling: 批次采样

4. Tensor Utilities (张量工具)
   - stack_with_left_padding: 左填充堆叠不等长序列
   - generate_candidates: 生成候选序列
   - pad_path: 路径填充

5. Stop Conditions (停止条件)
   - check_stop_conditions: 单序列停止条件检查
   - check_stop_conditions_parallel: 并行停止条件检查

6. Output Merge (输出合并)
   - merge_outputs: 合并骨架和并行分支的输出

7. Skeleton Parsing (骨架解析)
   - parse_skeleton: 解析骨架，提取并行分支
   - prepare_skeleton_input: 准备骨架输入
   - prepare_parallel_branches: 准备并行分支输入
"""

import copy
import random
import re
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
# Import prompt functions and constants from prompts module
from .prompts import (
    prepare_skeleton_input,
    prepare_parallel_branches,
    parse_skeleton_output,
)
from .logits_processor import SemanticLogitsProcessor


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
    准备采样 logits 处理器列表（温度、top-p、top-k 等）
    
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
# 2. Mask Building (掩码构建)
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


def build_continuous_decode_mask(
    history_bim: torch.Tensor,
    combined_bim_tensor: torch.Tensor,
    current_length: int,
    num_old_branches: int,
    num_nodes: int,
    tree_mask: Optional[torch.Tensor],
    new_prompt_lengths: List[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
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
        device: 目标设备
        dtype: 数据类型
        
    Returns:
        combined_mask: [1, 1, total_input_len, current_length + total_input_len]
    """
    total_input_len = combined_bim_tensor.shape[0]
    
    # =========================================================================
    # 1. 构建 Cross Mask (输入 -> 历史)
    # =========================================================================
    cross_mask = torch.full(
        (1, 1, total_input_len, current_length),
        torch.finfo(dtype).min, device=device
    )
    
    # Prefix 全部可见 (BIM == -1)
    is_prefix = (history_bim == -1).view(1, 1, 1, -1)
    cross_mask.masked_fill_(is_prefix, 0)
    
    # 同分支可见
    input_ids_view = combined_bim_tensor.view(1, 1, -1, 1)
    hist_ids_view = history_bim.view(1, 1, 1, -1)
    is_same_branch = (input_ids_view == hist_ids_view)
    cross_mask.masked_fill_(is_same_branch, 0)
    
    # =========================================================================
    # 2. 构建 Input Block Mask (输入 -> 输入)
    # =========================================================================
    input_block_mask = torch.full(
        (total_input_len, total_input_len),
        torch.finfo(dtype).min, device=device
    )
    
    # 老分支的 tree mask（块对角结构）
    if num_old_branches > 0 and tree_mask is not None:
        converted_tree_mask = torch.where(
            tree_mask == 1, 0.0, torch.finfo(dtype).min
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
    
    # =========================================================================
    # 3. 合并 Cross Mask 和 Input Block Mask
    # =========================================================================
    combined_mask = torch.cat([cross_mask, input_block_mask], dim=-1)
    
    return combined_mask


# =============================================================================
# 3. Verification Utilities
# =============================================================================

def _check_processors(logits_processor: Optional[LogitsProcessorList],) -> Tuple[bool, bool]:
    """
    检查 logits processor 的类型
    
    Args:
        logits_processor: logits 处理器列表
        
    Returns:
        use_sampling: 是否使用采样模式（有温度等 processor）
        has_semantic: 是否有语义约束 processor
    """
    if logits_processor is None or len(logits_processor) == 0:
        return False, False
    
    use_sampling = False
    has_semantic = False
    
    for processor in logits_processor:
        class_name = processor.__class__.__name__
        # 检测语义处理器
        if "SemanticLogitsProcessor" in class_name:
            has_semantic = True
        # 检测温度等采样处理器
        elif class_name in (
            "TemperatureLogitsWarper", 
            "TopKLogitsWarper", 
            "TopPLogitsWarper",
            "RepetitionPenaltyLogitsProcessor",
        ):
            use_sampling = True
    
    return use_sampling, has_semantic


def greedy_sampling(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [New] 逐 Token Greedy 评估（支持语义约束）
    
    适用于：无采样参数 但 有语义约束 (Semantic) 的情况。
    必须逐个 token 验证，因为 Semantic Processor 需要完整的上下文历史。
    
    逻辑：
    1. 拼接 context = prefix + accepted_draft
    2. 应用 processor
    3. Argmax 选择
    4. 对比 Draft，决定继续还是截断
    """
    device = logits.device
    num_paths, seq_len = candidates.shape
    
    # 1. 维度检查与 Context 准备
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0) # [1, prefix_len]
        
    # 初始化
    accept_cand = candidates[0, :1]  # [1] Root token
    accept_length = 1
    best_candidate = 0
    active_mask = torch.ones(num_paths, dtype=torch.bool, device=device)
    
    # [Modified] 使用独立的变量维护 context，不污染原始 input_ids
    # 初始 context 就是 input_ids
    # 注意：candidates[0,0] 通常是 input_ids 的最后一个 token，为了避免重复，
    # 我们通常只在后续拼接 candidates[..., 1:]
    current_full_context = input_ids.clone() 

    # 拼接root 
    # current_full_context = torch.cat([current_full_context, accept_cand.unsqueeze(0)], dim=1)

    # 循环验证 (从第 1 个生成的 token 开始，索引 0 是 root)
    for i in range(1, seq_len):
        # A. 筛选活跃路径
        prefix_match = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        active_mask = active_mask & prefix_match
        
        if not active_mask.any():
            break
            
        # 获取第一个活跃路径的索引
        fi = torch.nonzero(active_mask, as_tuple=True)[0][0].item()
        
        # B. 准备数据
        # Logits: 对应位置 i-1 的输出，用于预测位置 i
        current_logits_input = logits[fi, i - 1].unsqueeze(0) # [1, vocab]
        
        # Context: [Modified] 必须包含完整历史 (Prefix + Draft So Far)
        # candidates[fi, :i] 包括 root
        draft_context = candidates[fi, :i].unsqueeze(0)
        
        step_context = torch.cat([current_full_context, draft_context], dim=1)
            
        # C. 应用 Logits Processor (语义约束发生在这里)
        # [Critical] 传入完整 context
        # print(f"seleted token id before processor: {torch.argmax(current_logits_input).item()}")
        processed_logits = logits_processor(step_context, current_logits_input)[0] 
        
        # D. Greedy 选择 (Argmax)
        selected_token_id = torch.argmax(processed_logits).item()
        # print(f"seleted token id after processor: {selected_token_id}")
        
        # E. 验证 Draft
        # 检查是否有任何活跃路径预测正确
        # 注意：这里我们允许树中任何一条路径匹配（Speculative Decoding 逻辑）
        # 只要当前层级有任何一个 node 匹配 selected_token_id，我们就切换到那条路
        
        # 找出当前步骤预测为 selected_token_id 的活跃路径
        tok_mask = (candidates[:, i] == selected_token_id) & active_mask
        
        if tok_mask.any():
            # [Accept]
            # 更新最佳路径索引 (切换到匹配的那条路)
            best_candidate = torch.nonzero(tok_mask, as_tuple=True)[0][0].item()
            
            # 更新已接受序列
            new_tok = torch.tensor([selected_token_id], device=device)
            accept_cand = torch.cat([accept_cand, new_tok], dim=0)
            accept_length += 1
        else:
            # [Reject]
            # Draft 预测错了，或者被语义约束修改了
            # 停止接受，当前 selected_token_id 将作为修正后的 token 输出
            break

    # [Modified] 计算最终用于采样的 Logits (用于生成 bonus token 或返回)
    # 如果是非 EOS 结束，我们需要返回下一个 token 的概率分布
    
    # 准备最终的 Context
    if accept_length < seq_len:
        # 这是一个 Reject 的情况，或者长度耗尽
        # 我们使用 best_candidate 在 accept_length-1 位置的 logits
        final_logits_input = logits[best_candidate, accept_length - 1].unsqueeze(0)
        final_draft_part = candidates[best_candidate, :accept_length].unsqueeze(0)
    else:
        # 全部接受
        final_logits_input = logits[best_candidate, -1].unsqueeze(0)
        final_draft_part = candidates[best_candidate, :].unsqueeze(0)

    # 拼接完整 Context
    final_context = torch.cat([current_full_context, final_draft_part], dim=1)

    # 再次应用 Processor 得到最终分布
    processed_final = logits_processor(final_context, final_logits_input)[0]
    sample_p = torch.softmax(processed_final, dim=0)

    return (
        torch.tensor(best_candidate, device=device),
        torch.tensor(accept_length - 1, device=device), # 不含 root
        sample_p
    )


def rejection_sampling(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
    has_semantic_processor: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rejection Sampling 评估：逐 token 验证，使用 rejection sampling 决定是否接受
    
    Args:
        input_ids: [seq_len]
        logits: [num_paths, seq_len, vocab]
        candidates: [num_paths, seq_len]
        logits_processor: logits 处理器（用于温度、top-p 等）
        has_semantic_processor: 是否有语义处理器（影响 reject 时的 mask 策略）
        
    Returns:
        best_candidate: 最佳候选索引 (scalar tensor)
        accept_length: 接受长度，不含 root (scalar tensor)
        sample_p: 用于采样下一 token 的概率分布 [vocab]（softmax 后）
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
        
        current_logits_input = logits[fi, i - 1].unsqueeze(0)  # [1, vocab]
        current_context = candidates[fi, :i].unsqueeze(0)  # [1, i]
        
        gt_logits = logits_processor(current_context, current_logits_input)[0]  # [vocab]
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
                # Reject: 语义处理器时不 mask（保留语义约束的 token）
                # 非语义处理器时正常 mask
                if not has_semantic_processor:
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


def logits_sampling(
    input_ids: Optional[torch.Tensor],
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量评估候选序列（核心评估逻辑）
    
    根据 logits_processor 的组成自动选择采样策略：
    - 无 processor 或仅有 semantic processor -> Greedy
    - 有温度等 processor -> Rejection Sampling
    
    Args:
        input_ids: [batch, seq_len]
        logits: [batch, num_paths, seq_len, vocab]
        candidates: [batch, num_paths, seq_len]
        logits_processor: logits 处理器（可包含 semantic 和/或温度等 processor）
        
    Returns:
        best_candidate: 最佳候选索引 [batch]
        accept_length: 接受长度 [batch]
        sample_p: 采样概率分布 [batch, vocab]（softmax 后）
    """
    batch_size, num_paths, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # 检查 processor 类型
    use_sampling, has_semantic = _check_processors(logits_processor)

    best_candidates = []
    accept_lengths = []
    sample_logits_list = []
    
    # Greedy Mode: 无温度等 processor，或仅有 semantic processor
    if not use_sampling:   
        if not has_semantic:  # 没有任何约束       
            target_ids = torch.argmax(logits[:, :, :-1, :], dim=-1)
            draft_ids = candidates[:, :, 1:].to(device)
            posterior_mask = (draft_ids == target_ids).int()

            path_accept_lens = torch.cumprod(posterior_mask, dim=-1).sum(dim=-1)
            accept_length, best_candidate = path_accept_lens.max(dim=1)

            batch_idx = torch.arange(batch_size, device=device)
            next_pos = accept_length.clamp(max=seq_len - 1)
            sample_logits = logits[batch_idx, best_candidate, next_pos, :]
            sample_p = torch.softmax(sample_logits, dim=-1)

            return best_candidate, accept_length, sample_p
        
        else:   # 只有语意约束
            # 首先判断，进入该分支必定为skeleton阶段
            if has_semantic and batch_size > 1:
                raise RuntimeError(
                f"SemanticLogitsProcessor only supports skeleton phase and single sample inference, "
                f"but got batch size {batch_size}. Please set batch_size=1 when using SemanticLogitsProcessor.")
            
            for b in range(batch_size):
                bc, al, sl = greedy_sampling(input_ids[b], logits[b], candidates[b], logits_processor)
                # bc, al, sl = greedy_sampling(input_ids[b], logits[b], candidates[b], logits_processor)
                best_candidates.append(bc)
                accept_lengths.append(al)
                sample_logits_list.append(sl)
            
            return (
                torch.stack(best_candidates),
                torch.stack(accept_lengths),
                torch.stack(sample_logits_list),
            )
    
    # Rejection Sampling Mode: 有温度等 processor
    else: # use_sampling == True
        for b in range(batch_size):
            bc, al, sl = rejection_sampling(
                input_ids[b], logits[b], candidates[b], logits_processor, has_semantic
            )
            best_candidates.append(bc)
            accept_lengths.append(al)
            sample_logits_list.append(sl)
    
        return (
            torch.stack(best_candidates),
            torch.stack(accept_lengths),
            torch.stack(sample_logits_list),
        )


# =============================================================================
# 上层 API: evaluate_single 和 evaluate_parallel
# =============================================================================


def evaluate_single(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    单序列评估：用于 Eagle 投机解码和 SpecSoT Skeleton 阶段
    
    支持两种输入格式：
    - 带 batch 维度: logits [batch, num_paths, seq_len, vocab], candidates [batch, num_paths, seq_len]
    - 不带 batch 维度: logits [num_paths, seq_len, vocab], candidates [num_paths, seq_len]
    
    Args:
        input_ids: 前置的 token IDs, 辅助评估
        logits: 验证后的 logits
        candidates: 候选 tokens
        logits_processor: logits 处理器
            - None: Greedy 采样
            - 仅 SemanticLogitsProcessor: Greedy 采样 + 语义约束
            - 含温度等 processor: Rejection Sampling
        
    Returns:
        best_candidate: 最佳候选索引 [batch] 或 scalar
        accept_length: 接受长度 [batch] 或 scalar
        sample_token: 采样的 Bonus Token [batch, 1] 或 [1]
    """
    # 处理维度：如果没有 batch 维度，添加一个
    squeeze_output = False
    if logits.ndim == 3:
        # [num_paths, seq_len, vocab] -> [1, num_paths, seq_len, vocab]
        logits = logits.unsqueeze(0)
        candidates = candidates.unsqueeze(0)
        squeeze_output = True
    
    best_candidate, accept_length, sample_logits = logits_sampling(
        input_ids, logits, candidates, logits_processor
    )
    
    # 采样 Bonus Token
    if logits_processor is not None:
        sample_token = torch.multinomial(sample_logits, 1)
    else:
        sample_token = torch.argmax(sample_logits, dim=-1, keepdim=True)
    
    # 如果输入没有 batch 维度，输出也去掉 batch 维度
    if squeeze_output:
        best_candidate = best_candidate.squeeze(0)
        accept_length = accept_length.squeeze(0)
        sample_token = sample_token.squeeze(0)
    
    return best_candidate, accept_length, sample_token


def evaluate_parallel(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    并行评估：用于 SpecSoT 并行分支解码阶段
    
    注意：并行阶段通常无语义约束，但可能有温度等 processor
    
    Args:
        logits: 验证后的 logits [num_para, seq_len, vocab]
        draft_tokens: 候选 tokens [num_para, tree_size]
        retrieve_indices: 检索索引 [num_para, num_leaves, depth]
        logits_processor: logits 处理器（通常无语义约束）
        
    Returns:
        best_candidate: 最佳候选索引 [num_para]
        accept_length: 接受长度 [num_para]
        sample_token: 采样的 Bonus Token [num_para, 1]
    """
    num_para = logits.shape[0]
    device = logits.device
    
    retrieve_indices = retrieve_indices.to(device)
    draft_tokens = draft_tokens.to(device)
    
    # 处理无效索引
    padding_mask = (retrieve_indices == -1)
    safe_indices = retrieve_indices.clone()
    safe_indices[padding_mask] = 0
    
    # 提取候选 tokens: [num_para, num_leaves, depth]
    candidates = torch.gather(
        draft_tokens.unsqueeze(1).expand(-1, retrieve_indices.size(1), -1),
        2, safe_indices
    )
    candidates.masked_fill_(padding_mask, 0)
    
    # 提取候选 logits: [num_para, num_leaves, depth, vocab]
    vocab_size = logits.size(-1)
    flat_indices = safe_indices.view(num_para, -1).unsqueeze(-1).expand(-1, -1, vocab_size)
    candidate_logits = torch.gather(logits, 1, flat_indices)
    candidate_logits = candidate_logits.view(
        num_para, retrieve_indices.size(1), retrieve_indices.size(2), -1
    )
    
    # 调用核心评估逻辑
    # candidate_logits: [num_para, num_leaves, depth, vocab]
    # candidates: [num_para, num_leaves, depth]
    best_candidate, accept_length, sample_logits = logits_sampling(None, candidate_logits, candidates, logits_processor)
    
    # 采样 Bonus Token
    if logits_processor is not None:
        sample_token = torch.multinomial(sample_logits, 1)
    else:
        sample_token = torch.argmax(sample_logits, dim=-1, keepdim=True)
    
    return best_candidate, accept_length, sample_token


# =============================================================================
# 6. Utility Functions (工具函数)
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
    tokens_per_step: int = 60,
) -> bool:
    """
    检查停止条件（单序列解码）
    
    改进：使用峰值 KV cache 长度计算来提前停止，避免溢出。
    峰值长度 = current_length + tokens_per_step
    
    Args:
        input_ids: 当前生成的 token IDs [1, total_len]
        input_len: 原始输入长度
        stop_token_id: 停止 token ID (可选)
        eos_token_id: EOS token ID
        current_length: 当前 KV Cache 长度
        max_kv_len: 最大 KV Cache 长度
        tokens_per_step: 每步可能增加的最大 token 数 (default: 60 for Eagle)
        
    Returns:
        是否应该停止生成
    """
    generated_tokens = input_ids[0, input_len:].tolist()
    
    if stop_token_id and stop_token_id in generated_tokens:
        return True
    if eos_token_id in generated_tokens:
        return True
    
    # 峰值长度检查：确保下一步不会溢出
    peak_length = current_length + tokens_per_step
    if peak_length >= max_kv_len:
        print(f"Stopping due to KV cache limit: peak {peak_length} >= max {max_kv_len}")
        return True
    
    return False


def check_skeleton_stop(
    generated_text: str,
    eos_token_id: int,
    input_ids: torch.Tensor,
    input_len: int,
    current_length: int,
    max_kv_len: int,
    tokens_per_step: int = 60,
) -> bool:
    """
    检查骨架生成的停止条件
    
    统一检测 [DIRECT]...[END] 和 [PLAN]...[END] 两种格式的结束标记
    
    Args:
        generated_text: 已生成的文本
        eos_token_id: EOS token ID
        input_ids: 当前 input_ids [1, total_len]
        input_len: 原始输入长度
        
    Returns:
        should_stop: 是否应该停止生成
    """
    # 检查 EOS
    if eos_token_id in input_ids[0, input_len:].tolist():
        return True
    
    # 检查 [END] 标记（适用于 DIRECT 和 PLAN 两种模式）
    if "[END]" in generated_text:
        return True
    
    # KV Cache 溢出检查
    peak_length = current_length + tokens_per_step
    if peak_length >= max_kv_len:
        print(f"Stopping due to KV cache limit: peak {peak_length} >= max {max_kv_len}")
        return True
    
    return False


def check_stop_conditions_parallel(
    current_length: int,
    max_kv_len: int,
    num_active_branches: int,
    tokens_per_branch: int = 60,
) -> bool:
    """
    检查停止条件（并行解码）
    
    根据峰值 KV cache 长度计算，确定是否需要停止。
    峰值长度 = current_length + num_active_branches × tokens_per_branch
    
    Args:
        current_length: 当前 KV Cache 长度
        max_kv_len: 最大 KV Cache 长度
        num_active_branches: 当前活跃分支数量
        tokens_per_branch: 每个分支每步可能增加的最大 token 数
        
    Returns:
        should_stop: 是否应该停止
    """
    if num_active_branches == 0:
        return True
    
    # 计算峰值长度
    peak_length = current_length + num_active_branches * tokens_per_branch
    
    if peak_length >= max_kv_len:
        print(f"Stopping due to KV cache limit: peak {peak_length} >= max {max_kv_len}")
        return True
    
    return False


def merge_outputs(
    skeleton_output: torch.Tensor,
    parallel_branches_output: List[List[int]],
    instruction_len: List[int],
    device: torch.device,
    tasks: Optional[List[Dict]] = None,
    tokenizer = None,
    para_token_ids: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    合并骨架和并行分支的输出
    
    支持两种模式：
    1. 带任务标题模式 (tasks + tokenizer): 添加分支标题标记
    2. 简单合并模式 (para_token_ids): 使用特殊 token 分隔
    
    Args:
        skeleton_output: 骨架输出 tensor [1, seq_len]
        parallel_branches_output: 各分支的输出列表
        instruction_len: 各分支的指令长度
        device: 目标设备
        tasks: 任务列表（可选，用于添加标题）
        tokenizer: 分词器（可选，用于编码标题）
        para_token_ids: 特殊 token IDs（可选，用于简单合并模式）
        
    Returns:
        合并后的 token IDs tensor [1, merged_len]
    """
    num_para = len(parallel_branches_output)
    skeleton_part = skeleton_output[0].tolist()
    
    # 模式 1: 带任务标题模式
    if tasks is not None and tokenizer is not None:
        merged = skeleton_part.copy()
        
        for i, (branch_tokens, instr_len) in enumerate(zip(parallel_branches_output, instruction_len)):
            # 跳过指令前缀，只取生成的内容
            branch_content = branch_tokens[instr_len:]
            
            # 添加分支标题和内容
            if i < len(tasks):
                task = tasks[i]
                title_text = f"\n\n## {task['id']}. {task['title']}\n"
                title_tokens = tokenizer.encode(title_text, add_special_tokens=False)
                merged.extend(title_tokens)
            
            merged.extend(branch_content)
        
        return torch.tensor([merged], dtype=torch.long, device=device)
    
    # 模式 2: 简单合并模式（使用特殊 token）
    elif para_token_ids is not None:
        skeleton_part.append(para_token_ids['line_break_token_id'])

        parallel_part = []
        for i in range(num_para):
            branch_output = parallel_branches_output[i][instruction_len[i]:]
            parallel_part.extend(branch_output)
            parallel_part.append(para_token_ids['line_break_token_id'])
            print(f"Branch {i} Length: {len(branch_output)}")

        merged_ids = skeleton_part + parallel_part
        merged_ids.append(para_token_ids['para_end_token_id'])

        return torch.tensor([merged_ids], device=device)
    
    # 默认模式: 直接拼接
    else:
        merged = skeleton_part.copy()
        for i, (branch_tokens, instr_len) in enumerate(zip(parallel_branches_output, instruction_len)):
            branch_content = branch_tokens[instr_len:]
            merged.extend(branch_content)
        
        return torch.tensor([merged], dtype=torch.long, device=device)


def set_random_seed(seed: int):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# 7. Input Preparation & Skeleton Parser (Moved to prompts.py)
# =============================================================================
# NOTE: prepare_skeleton_input, parse_skeleton_output, prepare_parallel_branches
# have been moved to prompts.py and are re-exported from this module for 
# backward compatibility. Import them from prompts.py directly for new code.




