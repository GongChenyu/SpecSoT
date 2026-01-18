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
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .prompts import (
    base_prompt, skeleton_trigger_zh, parallel_trigger_zh,
    base_prompt_en, skeleton_trigger_en, parallel_trigger_en,
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


def create_skeleton_logits_processor(
    para_token_ids: Dict[str, int],
    prefix_len: int,
    sampling_processor: Optional[LogitsProcessorList] = None,
) -> LogitsProcessorList:
    """
    创建 skeleton 生成阶段使用的 LogitsProcessor
    
    包含语义约束 processor 和可选的采样 processor
    
    Args:
        para_token_ids: 特殊 token IDs
        prefix_len: 输入前缀长度
        sampling_processor: 采样相关的 processor (温度、top_p、top_k)
        
    Returns:
        组合后的 LogitsProcessorList
    """
    # 构造语义约束 Logits Processor
    sp_processor = SemanticLogitsProcessor(
        para_end_token_id=para_token_ids['para_end_token_id'],
        ellipsis_token_id=para_token_ids['ellipsis_token_id'],
        line_break_token_id=para_token_ids['line_break_token_id'],
        para_begin_token_id=para_token_ids['para_begin_token_id'],
        colon_token_id=para_token_ids['colon_token_id'],
        cn_colon_token_id=para_token_ids['cn_colon_token_id'],
        colon_new_line_token_id=para_token_ids['colon_new_line_token_id'],
        prefix_len=prefix_len
    )
    
    logits_processor = LogitsProcessorList([sp_processor])
    
    # 添加采样 processor (如果有)
    if sampling_processor is not None:
        logits_processor.extend(sampling_processor)
    
    return logits_processor


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


# =============================================================================
# 3. Verification Utilities
# =============================================================================

# =============================================================================
# Evaluation 模块重构
# 
# 采样逻辑说明：
# 1. Semantic Logits Processors 与温度等 Logits Processors 是正交关系
# 2. 通过有没有温度等 logits processors 判断是否使用 greedy 采样
# 3. 通过有没有 semantic logits processors 判断采样前的 logits 是否需要特殊语义处理
#
# 调用场景：
# - Single (单序列): Eagle 投机解码、SpecSoT Skeleton 阶段
# - Parallel (并行): SpecSoT 并行分支解码阶段（无语义约束）
#
# 架构：
# ┌─────────────────────┐     ┌─────────────────────┐
# │  evaluate_single    │     │  evaluate_parallel  │
# │  [batch, paths,     │     │  [num_para, paths,  │
# │   seq, vocab]       │     │   seq, vocab]       │
# └─────────┬───────────┘     └─────────┬───────────┘
#           │                           │
#           └───────────┬───────────────┘
#                       ▼
#           ┌───────────────────────┐
#           │  _evaluate_batch      │
#           │  (核心评估逻辑)        │
#           └───────────┬───────────┘
#                       │
#         ┌─────────────┴─────────────┐
#         ▼                           ▼
# ┌───────────────────┐     ┌───────────────────────┐
# │  _greedy_evaluate │     │ _rejection_sampling   │
# │                   │     │ _evaluate             │
# └───────────────────┘     └───────────────────────┘
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
    logits: torch.Tensor,
    candidates: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedy 评估：直接比较 argmax 选择最佳路径
    
    Args:
        logits: [num_paths, seq_len, vocab]
        candidates: [num_paths, seq_len]
        
    Returns:
        best_candidate: 最佳候选索引 (scalar tensor)
        accept_length: 接受长度，不含 root (scalar tensor)
        sample_p: 用于采样下一 token 的概率分布 [vocab]（softmax 后）
    """
    device = logits.device
    num_paths, seq_len, vocab_size = logits.shape
    
    # candidates[:, 1:] 因为第一个是 root (已知正确)
    # logits[:, :-1] 预测的是下一个位置
    posterior_mask = (
        candidates[:, 1:].to(device) == torch.argmax(logits[:, :-1, :], dim=-1)
    ).int()
    
    # 累积接受长度
    candidates_accept_length = torch.cumprod(posterior_mask, dim=-1).sum(dim=-1)
    accept_length, best_candidate = candidates_accept_length.max(dim=0)
    
    # 获取最佳路径的采样 logits
    best_path_logits = logits[best_candidate, :, :]  # [seq_len, vocab]
    next_token_pos = accept_length.clamp(max=seq_len - 1)
    sample_logits = best_path_logits[next_token_pos]  # [vocab]
    
    # 返回 softmax 后的概率分布，保持与 rejection sampling 输出一致
    sample_p = torch.softmax(sample_logits, dim=-1)
    
    return best_candidate, accept_length, sample_p


def rejection_sampling(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
    has_semantic_processor: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rejection Sampling 评估：逐 token 验证，使用 rejection sampling 决定是否接受
    
    Args:
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


def batched_sampling(
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
        logits: [batch, num_paths, seq_len, vocab]
        candidates: [batch, num_paths, seq_len]
        logits_processor: logits 处理器（可包含 semantic 和/或温度等 processor）
        
    Returns:
        best_candidate: 最佳候选索引 [batch]
        accept_length: 接受长度 [batch]
        sample_p: 采样概率分布 [batch, vocab]（softmax 后）
    """
    batch_size = logits.shape[0]
    device = logits.device
    
    # 检查 processor 类型
    use_sampling, has_semantic = _check_processors(logits_processor)
    
    # =========================================================================
    # Greedy Mode: 无温度等 processor，或仅有 semantic processor
    # =========================================================================
    if not use_sampling:
        best_candidates = []
        accept_lengths = []
        sample_logits_list = []
        
        for b in range(batch_size):
            bc, al, sl = greedy_sampling(logits[b], candidates[b])
            best_candidates.append(bc)
            accept_lengths.append(al)
            sample_logits_list.append(sl)
        
        return (
            torch.stack(best_candidates),
            torch.stack(accept_lengths),
            torch.stack(sample_logits_list),
        )
    
    # =========================================================================
    # Sampling Mode: 有温度等 processor
    # =========================================================================
    best_candidates = []
    accept_lengths = []
    sample_logits_list = []
    
    for b in range(batch_size):
        bc, al, sl = rejection_sampling(
            logits[b], candidates[b], logits_processor, has_semantic
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
        logits: 验证 logits
        candidates: 候选 tokens
        logits_processor: logits 处理器
            - None: Greedy 采样
            - 仅 SemanticLogitsProcessor: Greedy 采样 + 语义约束
            - 含温度等 processor: Rejection Sampling
        
    Returns:
        best_candidate: 最佳候选索引 [batch] 或 scalar
        accept_length: 接受长度 [batch] 或 scalar
        sample_p: 采样概率分布 [batch, vocab] 或 [vocab]（softmax 后）
    """
    # 处理维度：如果没有 batch 维度，添加一个
    squeeze_output = False
    if logits.ndim == 3:
        # [num_paths, seq_len, vocab] -> [1, num_paths, seq_len, vocab]
        logits = logits.unsqueeze(0)
        candidates = candidates.unsqueeze(0)
        squeeze_output = True
    
    best_candidate, accept_length, sample_logits = batched_sampling(
        logits, candidates, logits_processor
    )
    
    # 如果输入没有 batch 维度，输出也去掉 batch 维度
    if squeeze_output:
        best_candidate = best_candidate.squeeze(0)
        accept_length = accept_length.squeeze(0)
        sample_logits = sample_logits.squeeze(0)
    
    return best_candidate, accept_length, sample_logits


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
        sample_p: 采样概率分布 [num_para, vocab]（softmax 后）
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
    return batched_sampling(candidate_logits, candidates, logits_processor)


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
    para_token_ids: Dict[str, int],
    num_para: int,
    device: torch.device,
) -> torch.Tensor:
    """
    合并骨架和并行分支的输出
    
    Args:
        skeleton_output: 骨架输出 tensor
        parallel_branches_output: 各分支的输出列表
        instruction_len: 各分支的指令长度
        para_token_ids: 特殊 token IDs
        num_para: 分支数量
        device: 目标设备
        
    Returns:
        合并后的 token IDs tensor
    """
    skeleton_part = skeleton_output[0].tolist()
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


# =============================================================================
# 8. Skeleton & Parallel Input Preparation (骨架和并行输入准备)
# =============================================================================

def prepare_skeleton_input(
    tokenizer,
    task_prompt: str,
    model_type: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备 skeleton 生成阶段的输入
    
    根据模型类型选择对应的 prompt 模板，构建完整的输入序列
    
    Args:
        tokenizer: 分词器
        task_prompt: 用户输入的任务描述
        model_type: 模型类型 ('qwen', 'llama', 'other')
        device: 目标设备
        
    Returns:
        input_ids: 完整输入序列 [1, seq_len]
        task_input_ids: 任务输入部分 [1, task_len] (用于后续并行阶段的前缀复用)
    """
    if model_type == 'qwen':
        # Qwen 模型使用中文 prompt
        base_prompt_template = base_prompt
        skeleton_trigger = skeleton_trigger_zh
        print(f"Using Chinese prompts for {model_type} model")
    else:
        # Llama 和其他模型使用英文 prompt
        base_prompt_template = base_prompt_en
        skeleton_trigger = skeleton_trigger_en
        print(f"Using English prompts for {model_type} model")
    
    task_input = base_prompt_template.format(user_question=task_prompt)
    task_input_ids = tokenizer([task_input], return_tensors="pt").input_ids.to(device)
    skeleton_input_ids = tokenizer([skeleton_trigger], return_tensors="pt").input_ids.to(device)
    input_ids = torch.cat([task_input_ids, skeleton_input_ids], dim=-1)
    
    return input_ids, task_input_ids


def parse_skeleton(
    tokenizer,
    skeleton_ids: torch.Tensor,
    para_token_ids: Dict[str, int],
) -> Tuple[Optional[List[List[int]]], Optional[List[int]]]:
    """
    解析骨架，提取分支标题和预测长度
    
    骨架格式示例：
    ####标题1(100):...
    ####标题2(200):...
    ####%%%%
    
    注意：此函数只解析骨架，不添加指令前缀。
    要准备并行输入，请使用 prepare_parallel_branches。
    
    Args:
        tokenizer: 分词器
        skeleton_ids: 骨架 token IDs
        para_token_ids: 特殊 token IDs 字典
        
    Returns:
        branch_headers: 分支标题列表（不含指令前缀）
        predicted_lengths: 每个分支的预测长度（从括号中解析）
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

    # 清洗分支（截取到冒号）并解析长度
    branch_headers = []
    predicted_lengths = []
    
    for br in raw_branches:
        cut_idx = -1
        for i, token in enumerate(br):
            if token in colon_ids:
                cut_idx = i
                break
        
        if cut_idx != -1:
            branch_tokens = br[:cut_idx + 1]
            branch_headers.append(branch_tokens)
            
            # 解析长度：从 token 中提取括号内的数字
            branch_text = tokenizer.decode(branch_tokens)
            # 匹配格式：####标题(123):
            length_match = re.search(r'\((\d+)\)', branch_text)
            if length_match:
                predicted_length = int(length_match.group(1))
                predicted_lengths.append(predicted_length)
            else:
                # 如果没有找到长度，使用默认值
                predicted_lengths.append(200)  # 默认 200 tokens
        else:
            branch_headers.append(br)
            predicted_lengths.append(200)

    return branch_headers, predicted_lengths


def prepare_parallel_branches(
    tokenizer,
    branch_headers: List[List[int]],
    model_type: str = 'qwen',
) -> Tuple[List[List[int]], List[int]]:
    """
    准备并行分支的输入（添加指令前缀）
    
    为每个分支添加上下文指令，构建完整的并行分支输入
    
    Args:
        tokenizer: 分词器
        branch_headers: 分支标题列表（来自 parse_skeleton）
        model_type: 模型类型 ('qwen', 'llama', 'other')
        
    Returns:
        clean_branches: 完整的分支列表（含指令前缀）
        instruction_lengths: 每个分支的指令前缀长度
    """
    # 选择对应的 parallel trigger
    if model_type == 'qwen':
        parallel_trigger = parallel_trigger_zh
    else:
        parallel_trigger = parallel_trigger_en
    
    # 构建骨架上下文
    result_skeleton = []
    for br in branch_headers:
        result_skeleton.extend(br)
    result_skeleton_str = tokenizer.decode(result_skeleton)

    # 为每个分支添加指令前缀
    clean_branches = []
    instruction_lengths = []
    
    for br in branch_headers:
        branch_str = tokenizer.decode(br)
        instruction = parallel_trigger.format(
            skeleton_context=result_skeleton_str,
            current_point=branch_str
        )
        instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
        full_branch = instruction_ids + br
        clean_branches.append(full_branch)
        instruction_lengths.append(len(instruction_ids))

    return clean_branches, instruction_lengths



