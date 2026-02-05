# coding=utf-8
"""
评估工具模块

提供投机解码的评估和采样函数：
- evaluate_single(): 单序列评估
- evaluate_parallel(): 并行评估
- greedy_sampling(): Greedy 采样
- rejection_sampling(): Rejection Sampling
- logits_sampling(): 核心采样逻辑

设计原则：
1. 支持多种采样策略（Greedy, Rejection Sampling）
2. 支持语义约束（SemanticLogitsProcessor）
3. 批量处理和单序列处理统一接口
"""

import random
from typing import List, Tuple, Optional, Any

import torch
from transformers.generation.logits_process import LogitsProcessorList


# =============================================================================
# 辅助函数
# =============================================================================

def _check_processors(
    logits_processor: Optional[LogitsProcessorList],
) -> Tuple[bool, bool]:
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


# =============================================================================
# Greedy Sampling
# =============================================================================

def greedy_sampling(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    逐 Token Greedy 评估（支持语义约束）
    
    适用于：无采样参数 但 有语义约束 (Semantic) 的情况。
    必须逐个 token 验证，因为 Semantic Processor 需要完整的上下文历史。
    
    Args:
        input_ids: 前置 token IDs [batch_size=1, prefix_len] 
        logits: [batch_size=1, num_paths, seq_len, vocab] 
        candidates: [batch_size=1, num_paths, seq_len]
        logits_processor: logits 处理器
        
    Returns:
        best_candidate: 最佳候选索引 (scalar tensor)
        accept_length: 接受长度，不含 root (scalar tensor)
        sample_p: 采样概率分布 [vocab]
    """
    device = logits.device
    batch_size, num_paths, seq_len, vocab_size = logits.shape
    
    # 维度处理
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, prefix_len]
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)        # [1, num_paths, seq_len, vocab]
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0) # [1, num_paths, seq_len]
    
    # 初始化
    accept_cand = candidates[:, 0, :1]  # 取 Batch 0, Path 0, Root Token
    accept_length = 1
    best_candidate_idx = 0  # 暂时用 scalar 记录 index，最后转 tensor
    # active_mask: [1, num_paths] - 标记哪些路径还是活跃的
    active_mask = torch.ones((batch_size, num_paths), dtype=torch.bool, device=device)
    # full_context_prefix: [1, prefix_len]
    full_context_prefix = input_ids.clone()

    # 循环验证
    for i in range(1, seq_len):
        # 3.1 筛选活跃路径
        # candidates: [1, N, L] -> slice -> [1, N, i]
        # accept_cand: [1, i] -> unsqueeze -> [1, 1, i] 以便广播比较
        current_draft_slice = candidates[:, :, :accept_length]
        current_accept_seq = accept_cand.unsqueeze(1)
        
        # 检查前缀匹配: [1, N]
        prefix_match = (current_draft_slice == current_accept_seq).all(dim=2)
        active_mask = active_mask & prefix_match
        
        # 如果没有活跃路径，提前退出
        if not active_mask.any():
            break
        
        # 3.2 获取第一个活跃路径的索引 (Grid Search 逻辑通常只需验证第一个合法的即可)
        # torch.nonzero 返回 [k, 2] (batch_idx, path_idx)，我们需要 path_idx
        # 因为 batch=1, batch_idx 恒为 0
        active_indices = torch.nonzero(active_mask, as_tuple=False)
        fi = active_indices[0, 1].item() # first active path index
        
        # 3.3 准备 LogitsProcessor 的输入
        # 这里的 logits 需要是 [1, vocab]，对应当前步 (i-1)
        current_logits_input = logits[:, fi, i - 1, :].unsqueeze(0) # [1, 1, vocab]
        
        # 构造当前完整的 context: prefix + draft_so_far
        # draft_tokens: [1, i] (从 candidates 中取前 i 个 token)
        draft_context = candidates[:, fi, :i]
        step_context = torch.cat([full_context_prefix, draft_context], dim=1) # [1, prefix + i]
        
        # 3.4 应用 Logits Processor (核心：语义约束在这里生效)
        # 输入: input_ids=[1, seq], scores=[1, vocab]
        # 输出: scores=[1, vocab]
        processed_logits = logits_processor(step_context, current_logits_input)
        
        # 3.5 Greedy 选择
        selected_token_id = torch.argmax(processed_logits, dim=-1).item()
        
        # 3.6 验证 Draft 是否匹配 Greedy 结果
        # 检查当前步 (i) 的 token 是否等于 selected_token_id
        # candidates[:, :, i]: [1, N]
        target_token_mask = (candidates[:, :, i] == selected_token_id)
        
        # 只有既是 active 又是 target token 的路径才能继续
        valid_next_mask = target_token_mask & active_mask
        
        if valid_next_mask.any():
            # Accept: 更新最佳路径索引
            # 找到第一个符合条件的路径
            valid_indices = torch.nonzero(valid_next_mask, as_tuple=False)
            best_candidate_idx = valid_indices[0, 1].item()
            
            # 更新已接受序列
            new_tok = torch.tensor([[selected_token_id]], device=device) # [1, 1]
            accept_cand = torch.cat([accept_cand, new_tok], dim=1) # [1, len+1]
            accept_length += 1
        else:
            # Reject: 贪婪解码结果与 Draft 不符，且无备选路径
            break

    # ==========================================
    # 4. 计算最终用于采样的 Logits (Next Token Prediction)
    # ==========================================
    # 如果接受长度小于 seq_len，我们需要基于最后一个接受的 token 预测下一个
    # 如果接受长度等于 seq_len，我们基于最后一个 token 预测（即 drafting 结束后的下一步）
    
    # 确定最终使用的 logits 输入位置
    # 注意：logits 的 seq_len 维度对应的是 input 的位置
    # 如果 accept_length 是 3 (root, t1, t2)，我们要预测 t3
    # 对应的 logits 索引应该是 accept_length - 1 (因为 logits 是从第0个输入开始产生的输出)
    
    if accept_length < seq_len:
        logit_idx = accept_length - 1
        final_draft_part = candidates[:, best_candidate_idx, :accept_length] # [1, len]
    else:
        logit_idx = -1 # 最后一个
        final_draft_part = candidates[:, best_candidate_idx, :] # [1, len]

    final_logits_input = logits[:, best_candidate_idx, logit_idx, :].unsqueeze(0) # [1, 1, vocab]
    
    # 最终 Context
    final_context = torch.cat([full_context_prefix, final_draft_part], dim=1)
    
    # 再次经过 Processor 处理（确保最终输出也符合语义约束）
    processed_final = logits_processor(final_context, final_logits_input)
    
    # 计算概率
    sample_p = torch.softmax(processed_final, dim=-1) # [1, vocab]

    # ==========================================
    # 5. 返回结果 (保持 Tensor 格式)
    # ==========================================
    return (
        torch.tensor([best_candidate_idx], device=device), # [1]
        torch.tensor([accept_length - 1], device=device),  # [1], 减去 root
        sample_p # [1, vocab]
    )


# =============================================================================
# Rejection Sampling
# =============================================================================

def rejection_sampling(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: LogitsProcessorList,
    has_semantic_processor: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rejection Sampling 评估
    
    Args:
        input_ids: [seq_len]
        logits: [num_paths, seq_len, vocab]
        candidates: [num_paths, seq_len]
        logits_processor: logits 处理器
        has_semantic_processor: 是否有语义处理器
        
    Returns:
        best_candidate: 最佳候选索引
        accept_length: 接受长度，不含 root
        sample_p: 采样概率分布
    """
    device = logits.device
    num_paths, seq_len = candidates.shape
    
    # 初始化
    accept_cand = candidates[0, :1]
    accept_length = 1
    best_candidate = 0
    active_mask = torch.ones(num_paths, dtype=torch.bool, device=device)
    
    for i in range(1, seq_len):
        # 匹配前缀
        prefix_match = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        active_mask = active_mask & prefix_match
        
        if not active_mask.any():
            break
        
        fi = torch.nonzero(active_mask, as_tuple=True)[0][0].item()
        
        current_logits_input = logits[fi, i - 1].unsqueeze(0)
        current_context = candidates[fi, :i].unsqueeze(0)
        
        gt_logits = logits_processor(current_context, current_logits_input)[0]
        gtp = torch.softmax(gt_logits, dim=0)
        
        # 收集候选 tokens
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
                # Reject
                if not has_semantic_processor:
                    gtp[xi] = 0.0
                    gtp_sum = gtp.sum()
                    if gtp_sum > 0:
                        gtp = gtp / gtp_sum
                    else:
                        break
        
        if not accepted:
            break
    
    # 计算最终 logits
    if accept_length < seq_len:
        final_logits = logits[best_candidate, accept_length - 1].unsqueeze(0)
        final_context = candidates[best_candidate, :accept_length].unsqueeze(0)
    else:
        final_logits = logits[best_candidate, -1].unsqueeze(0)
        final_context = candidates[best_candidate, :].unsqueeze(0)
    
    processed = logits_processor(final_context, final_logits)[0]
    sample_logits = torch.softmax(processed, dim=0)
    
    return (
        torch.tensor([best_candidate], device=device),  # [1]
        torch.tensor([accept_length - 1], device=device),  # [1]
        sample_logits.unsqueeze(0) if sample_logits.ndim == 1 else sample_logits,  # [1, V]
    )


# =============================================================================
# Core Sampling Logic
# =============================================================================

def logits_sampling(
    input_ids: Optional[torch.Tensor],
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量评估候选序列（核心评估逻辑）
    
    根据 logits_processor 的组成自动选择采样策略。
    
    Args:
        input_ids: [batch, seq_len]
        logits: [batch, num_paths, seq_len, vocab]
        candidates: [batch, num_paths, seq_len]
        logits_processor: logits 处理器
        
    Returns:
        best_candidate: [batch]
        accept_length: [batch]
        sample_p: [batch, vocab]
    """
    batch_size, num_paths, seq_len, vocab_size = logits.shape
    device = logits.device
    
    # 检查 processor 类型
    use_sampling, has_semantic = _check_processors(logits_processor)

    best_candidates = []
    accept_lengths = []
    sample_logits_list = []
    
    # Greedy Mode
    if not use_sampling:
        if not has_semantic:
            # 纯 Greedy（无约束）
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
        
        else:
            # Greedy + 语义约束
            if batch_size > 1:
                raise RuntimeError(
                    f"SemanticLogitsProcessor only supports batch_size=1, got {batch_size}"
                )

            best_candidate, accept_length, sample_p = greedy_sampling(
                input_ids, logits, candidates, logits_processor
            )

            return best_candidate, accept_length, sample_p
    
    # Rejection Sampling Mode
    else:
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
# 上层 API
# =============================================================================

def evaluate_single(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    单序列评估
    
    用于 Eagle 投机解码和 SpecSoT Skeleton 阶段。
    
    Args:
        input_ids: 前置 token IDs
        logits: 验证后的 logits
        draft_tokens: 候选 tokens
        retrieve_indices: 检索索引
        logits_processor: logits 处理器
        
    Returns:
        best_candidate: 最佳候选索引
        accept_length: 接受长度
        sample_token: 采样的 Bonus Token
    """
    device = logits.device

    # 确保所有 tensor 在同一设备上
    if retrieve_indices.device != device:
        retrieve_indices = retrieve_indices.to(device)
    if draft_tokens.device != device:
        draft_tokens = draft_tokens.to(device)
    if input_ids.device != device:
        input_ids = input_ids.to(device)

    # 处理维度: logits 应该是 [batch, 1, seq, vocab]
    if logits.ndim == 3:
        # [batch, seq, vocab] -> [batch, 1, seq, vocab]
        logits = logits.unsqueeze(1)
    
    # 从 retrieve_indices 提取候选
    if retrieve_indices.ndim == 2:
        retrieve_indices = retrieve_indices.unsqueeze(0)
    
    # 处理无效索引
    padding_mask = (retrieve_indices == -1)
    safe_indices = retrieve_indices.clone()
    safe_indices[padding_mask] = 0
    
    # 提取候选 tokens
    if draft_tokens.ndim == 1:
        draft_tokens = draft_tokens.unsqueeze(0)
    
    batch_size = logits.shape[0]
    num_leaves = retrieve_indices.shape[1]
    
    candidates = torch.gather(
        draft_tokens.unsqueeze(1).expand(-1, num_leaves, -1),
        2, safe_indices
    )
    candidates[padding_mask] = -1
    
    # 提取候选对应的 logits
    # logits 现在是 [batch, 1, seq, vocab] (4D)
    # 需要 expand 到 [batch, num_leaves, seq, vocab]
    expanded_logits = logits.expand(-1, num_leaves, -1, -1)
    candidate_logits = torch.gather(
        expanded_logits,
        2, safe_indices.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])
    )
    
    # 调用核心采样逻辑
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    
    best_candidate, accept_length, sample_logits = logits_sampling(
        input_ids, candidate_logits, candidates, logits_processor
    )
    
    # 采样 Bonus Token
    # 确保 sample_logits 是 2D [batch, vocab]（torch.multinomial 只接受 1D 或 2D）
    if sample_logits.ndim == 3:
        sample_logits = sample_logits.squeeze(1)  # [batch, 1, vocab] -> [batch, vocab]
    
    if logits_processor is not None:
        sample_token = torch.multinomial(sample_logits, 1)
    else:
        sample_token = torch.argmax(sample_logits, dim=-1, keepdim=True)
    
    return best_candidate, accept_length, sample_token


def evaluate_parallel(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    并行评估
    
    用于 SpecSoT 并行分支解码阶段。
    
    Args:
        logits: [num_para, seq_len, vocab]
        draft_tokens: [num_para, tree_size]
        retrieve_indices: [num_para, num_leaves, depth]
        logits_processor: logits 处理器
        
    Returns:
        best_candidate: [num_para]
        accept_length: [num_para]
        sample_token: [num_para, 1]
    """
    num_para = logits.shape[0]
    device = logits.device
    
    retrieve_indices = retrieve_indices.to(device)
    draft_tokens = draft_tokens.to(device)
    
    # 处理无效索引
    padding_mask = (retrieve_indices == -1)
    safe_indices = retrieve_indices.clone()
    safe_indices[padding_mask] = 0
    
    # 提取候选 tokens
    candidates = torch.gather(
        draft_tokens.unsqueeze(1).expand(-1, retrieve_indices.size(1), -1),
        2, safe_indices
    )
    candidates[padding_mask] = -1
    
    # 提取候选对应的 logits
    candidate_logits = torch.gather(
        logits.unsqueeze(1).expand(-1, retrieve_indices.size(1), -1, -1),
        2, safe_indices.unsqueeze(-1).expand(-1, -1, -1, logits.shape[-1])
    )
    
    # 批量评估
    best_candidates = []
    accept_lengths = []
    sample_tokens = []
    
    for i in range(num_para):
        # 纯 Greedy（并行阶段通常无语义约束）
        target_ids = torch.argmax(candidate_logits[i, :, :-1, :], dim=-1)
        draft_ids = candidates[i, :, 1:]
        posterior_mask = (draft_ids == target_ids).int()

        path_accept_lens = torch.cumprod(posterior_mask, dim=-1).sum(dim=-1)
        accept_length, best_candidate = path_accept_lens.max(dim=0)

        # 维度规范：确保返回 [1] 维度的 tensor
        if accept_length.ndim == 0:
            accept_length = accept_length.unsqueeze(0)
        if best_candidate.ndim == 0:
            best_candidate = best_candidate.unsqueeze(0)

        # 采样
        next_pos = accept_length[0].clamp(max=candidate_logits.shape[2] - 1)
        sample_logits = candidate_logits[i, best_candidate[0], next_pos, :]
        probabilities = torch.nn.functional.softmax(sample_logits, dim=-1)
        if logits_processor is not None:
            sample_token = torch.multinomial(probabilities.unsqueeze(0), 1)  # [1, 1]
        else:
            sample_token = torch.argmax(sample_logits, dim=-1, keepdim=True)

        best_candidates.append(best_candidate)
        accept_lengths.append(accept_length)
        sample_tokens.append(sample_token)

    # 使用 cat 拼接，因为每个元素已经是 [1] 维度
    return (
        torch.cat(best_candidates, dim=0),  # [num_para]
        torch.cat(accept_lengths, dim=0),   # [num_para]
        torch.cat(sample_tokens, dim=0),    # [num_para, 1]
    )
