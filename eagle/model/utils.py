import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token


def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1] 
        logits = logits_processor(input_ids, logits)   # 修改：semantic processor需要输入前序token
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_generate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


# def evaluate_posterior(
#         logits: torch.Tensor,
#         candidates: torch.Tensor,
#         logits_processor,
#         input_ids=None,
# ):
#     """
#     Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

#     Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
#     probabilities to select the best candidate.

#     Args:
#     - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
#     - candidates (torch.Tensor): Candidate token sequences.
#     - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
#     - posterior_threshold (float): Threshold for posterior probability.
#     - posterior_alpha (float): Scaling factor for the threshold.

#     Returns:
#     - best_candidate (torch.Tensor): Index of the chosen best candidate.
#     - accept_length (int): Length of the accepted candidate sequence.
#     """
#     # Greedy decoding based on temperature value
#     if logits_processor is None:   # 缺乏semantic的验证
#         # Find the tokens that match the maximum logits for each position in the sequence
#         posterior_mask = (
#                 candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
#         ).int()
#         candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
#         accept_length = candidates_accept_length.max()
#         # Choose the best candidate
#         if accept_length == 0:
#             # Default to the first candidate if none are accepted
#             best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
#         else:
#             best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
#         return best_candidate, accept_length, logits[best_candidate, accept_length]

#     else:
#         accept_length = 1
#         accept_cand = candidates[0][:1]
#         best_candidate = 0
#         for i in range(1, candidates.shape[1]):
#             if i != accept_length:
#                 break
#             adjustflag = False
#             is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
#             fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
#             gt_logits = logits[fi, i - 1][None]
#             gt_logits = logits_processor(input_ids, gt_logits)[0]  # 修改：semantic processor需要输入前序token
#             gtp = torch.softmax(gt_logits, dim=0)
#             candidates_set = []
#             for j in range(candidates.shape[0]):
#                 if is_eq[j]:
#                     x = candidates[j, i]
#                     xi = x.item()
#                     if xi in candidates_set or xi == -1:
#                         continue
#                     candidates_set.append(xi)
#                     r = random.random()
#                     px = gtp[xi]
#                     qx = 1.0
#                     acp = px / qx
#                     if r <= acp:
#                         accept_cand = torch.cat((accept_cand, x[None]), dim=0)
#                         accept_length += 1
#                         best_candidate = j
#                         break
#                     else:
#                         gtp[xi] = 0
#                         gtp = gtp / gtp.sum()
#                         adjustflag = True
#         if adjustflag and accept_length != candidates.shape[1]:
#             sample_p = gtp
#         else:
#             gt_logits = logits[best_candidate, accept_length - 1][None]
#             gt_logits = logits_processor(input_ids, gt_logits)[0]  # 修改：semantic processor需要输入前序token
#             sample_p = torch.softmax(gt_logits, dim=0)
#         return torch.tensor(best_candidate), accept_length - 1, sample_p


# @torch.no_grad()
# def update_inference_inputs(
#         input_ids,
#         candidates,
#         best_candidate,
#         accept_length,
#         retrieve_indices,
#         logits_processor,
#         new_token,
#         past_key_values_data_list,
#         current_length_data,
#         model,
#         hidden_state_new,
#         sample_p
# ):
#     prev_input_len = input_ids.shape[1]
#     # Map the best candidate indices to the original indices in the sequence
#     select_indices = (
#             retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
#     )
#     # Append the tokens from the best candidate to the input sequence
#     input_ids = torch.cat(
#         [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
#     )
#     # Update the past key values based on the selected tokens
#     # Source tensor that contains relevant past information based on the selected candidate
#     for past_key_values_data in past_key_values_data_list:
#         tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
#         # Destination tensor where the relevant past information will be stored
#         dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
#         # Copy relevant past information from the source to the destination
#         dst.copy_(tgt, non_blocking=True)

#     # Update the current length tensor (currently only support batch size is 1)
#     current_length_data.fill_(prev_input_len + tgt.shape[-2])

#     retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
#     accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
#     # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
#     # token=token[None,None]
#     prob = sample_p
#     if logits_processor is not None:
#         token = torch.multinomial(prob, 1)
#         token = token[None]
#     else:
#         token = torch.argmax(prob)
#         token = token[None, None]
#     # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
#     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
#                                               input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
#                                               head=model.base_model.lm_head,logits_processor=logits_processor)


#     new_token += accept_length + 1

#     return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


# =========================== 新生成 ==============================



def initialize_tree_packed(prefix_len, input_ids, model, tips_indices, branch_begins, branch_lengths, draft_input_ids, logits_processor=None):
    """
    Args:
        input_ids: [1, Total_Len] (Prefix + Branches)
        branch_offsets: List[int], 每个 Branch 在 input_ids 中的起始索引
        branch_lengths: List[int], 每个 Branch 的长度
    """
    device = input_ids.device
    num_para = len(branch_lengths)

    # 2. Parallel Prefill (Initial Tree)
    # 构建初始 Mask (Prefix + Prompts)   这个函数可以放到initialize_tree_packed这里边去
    attention_mask = model._construct_parallel_mask(prefix_len, branch_len=input_ids.shape[1]-prefix_len, dtype=torch.float32)

    #  考虑到易读性，我们输入全量的input，因此这里需要切片处理
    kv_len = model.past_key_values[0][0].shape[2]
    position_ids = model.full_position_ids

    # Base Model Forward (Prefill)
    # outputs: odict_keys(['last_hidden_state', 'past_key_values', 'hidden_states'])
    outputs, hidden_states = model(
        input_ids=input_ids[:, kv_len:], 
        attention_mask=attention_mask, 
        position_ids=position_ids[kv_len:], 
        past_key_values=model.past_key_values, 
        output_orig=False)

    # 因为这里不会使用语义logits约束，因此只输入logits即可
    tips_hidden = hidden_states[:, tips_indices - kv_len, :]  # 需要通过kv len矫正index，因为这个变量是按照全量输入计算的
    tips_logits = model.base_model.lm_head(tips_hidden)   
    current_logits = tips_logits.squeeze(0)  # 调整维度以方便处理: [Num_Para, Vocab]

    if logits_processor is not None:
        # 集中logits processor都支持向量化操作
        current_logits = logits_processor(None, current_logits)
        probs = torch.nn.functional.softmax(current_logits, dim=-1)
        root_tokens = torch.multinomial(probs, num_samples=1)
    else:
        root_tokens = torch.argmax(current_logits, dim=-1, keepdim=True)

    # ================= 4. Prepare Draft Inputs (Batched Update) =================
    batched_input = draft_input_ids
    draft_input_ids = torch.cat([batched_input, root_tokens], dim=1) # [B, L+1]
    
    # Clone the output hidden states
    ea_device = model.ea_layer.lm_head.weight.device
    if outputs["hidden_states"][0].device != ea_device:
        outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
    packed_hidden=torch.cat(outputs["hidden_states"],dim=-1)[0]

    # packed_hidden = hidden_states[0]  # 提取 Hidden States (Base Model Output)    
    branch_hidden_list = []
    for i in range(num_para):
        start = branch_begins[i]
        end = start + branch_lengths[i]
        branch_hidden_list.append(packed_hidden[start-prefix_len:end-prefix_len])
    batched_hidden = stack_with_left_padding(branch_hidden_list, pad_id=0, device=device) # Hidden States 用 0 填充对齐

    # Draft Model Generation
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_generate(
        batched_hidden, draft_input_ids, model.base_model.lm_head, logits_processor=None, prefix_len=prefix_len)  # draft 不用 logits_processor
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states  # 返回的 hidden_states 后续可能修改


def evaluate_posterior(logits: torch.Tensor, candidates: torch.Tensor, logits_processor, para_token_ids=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    has_semantic_processor = False   # 判断有无 SemanticLogitsProcessor
    special_token = None
    if logits_processor is not None:
        # 遍历 processor 列表检查是否有自定义的 SemanticLogitsProcessor
        for processor in logits_processor:
            # 这里通过类名判断，确保你的类名字符串包含 'Semantic' 或者是你指定的类
            if "SemanticLogitsProcessor" in processor.__class__.__name__:
                has_semantic_processor = True
                special_token = [para_token_ids['ellipsis_token_id'], para_token_ids['line_break_token_id'], para_token_ids['para_begin_token_id']]
                break
    
    batch_size, num_paths, seq_len, vocab_size = logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3]

    # Greedy decoding based on temperature value
    if logits_processor is None:
        # candidates包括root，但是logits是从root后的第一个子节点开始
        posterior_mask = (candidates[:, :, 1:].to(logits.device) == torch.argmax(logits[:, :, :-1, :], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=-1)
        accept_length, best_candidate = candidates_accept_length.max(dim=-1)

        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        best_path_logits = logits[batch_indices, best_candidate, :, :]

        seq_len = best_path_logits.shape[1]    # 最优路径上的token
        next_token_pos = accept_length.clamp(max=seq_len - 1)   # 不能超出最优路径的最大长度获取index，但是这里应该考虑全部接收的情况，就需要从logits里边取值了。
        sample_logits = best_path_logits[batch_indices, next_token_pos]  # sample的token就是接收token的下一个，也就是index=accept_length的那个
        
    else:
        # Process each batch item separately
        best_candidates = []
        accept_lengths = []
        sample_logits_list = []

        for b in range(batch_size):
            bc, al, sl = _evaluate_posterior_single(logits[b], candidates[b], logits_processor, has_semantic_processor, special_token)
            best_candidates.append(bc)
            accept_lengths.append(al)
            sample_logits_list.append(sl)

        best_candidate = torch.stack(best_candidates)      # [B]
        accept_length = torch.stack(accept_lengths)        # [B]
        sample_logits = torch.stack(sample_logits_list)    # [B, V]
    
    return best_candidate, accept_length, sample_logits


def _evaluate_posterior_single(
    logits: torch.Tensor,           # [num_paths, seq_len, vocab_size]
    candidates: torch.Tensor,       # [num_paths, seq_len]
    logits_processor: List,
    has_semantic_processor: bool,
    special_token: List[int]
):
    """
    Evaluate posterior for a single batch item (no batch dimension).
    Implements rejection sampling token-by-token.
    """
    device = logits.device
    num_paths, seq_len = candidates.shape

    # Initial: accept the root (position 0 is given)
    accept_cand = candidates[0, :1]  # [1]
    accept_length = 1
    best_candidate = 0

    # Track which paths are still viable (initially all)
    active_mask = torch.ones(num_paths, dtype=torch.bool, device=device)

    for i in range(1, seq_len):  # i is the candidate position to validate (1 to L-1)
        # Find paths that match the currently accepted prefix
        prefix_match = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        active_mask = active_mask & prefix_match

        if not active_mask.any():
            break  # No path matches the accepted prefix

        # Get the first matching path to compute processed logits
        fi = torch.nonzero(active_mask, as_tuple=True)[0][0].item()

        # Get base logits from path `fi` at position i-1 (predicts token at i)
        current_logits_input = logits[fi, i - 1].unsqueeze(0)  # [1, V]
        current_context_ids = candidates[fi, :i].unsqueeze(0)   # [1, i]

        # Apply logits processor
        gt_logits = logits_processor(current_context_ids, current_logits_input)[0]  # [V]
        gtp = torch.softmax(gt_logits, dim=0)  # [V]

        # Gather unique candidate tokens at position i from active paths
        candidate_tokens = candidates[active_mask, i]  # [M]
        unique_tokens = torch.unique(candidate_tokens)
        accepted = False

        for tok in unique_tokens:
            xi = tok.item()
            if xi == -1:
                continue

            r = random.random()
            px = gtp[xi].item()
            qx = 1.0  # proposal distribution is uniform over draft tokens? or deterministic?
            acp = min(1.0, px / qx) if qx > 0 else 0.0

            if r <= acp:
                # Accept this token
                accept_cand = torch.cat([accept_cand, tok.unsqueeze(0)], dim=0)
                accept_length += 1
                # Update best_candidate to any path that has this token
                tok_mask = (candidates[:, i] == tok) & active_mask
                best_candidate = torch.nonzero(tok_mask, as_tuple=True)[0][0].item()
                accepted = True
                break
            else:
                # Reject: zero out this token in distribution (for potential retry in same step? 
                # But typically we break on first accept or move to next step)
                if has_semantic_processor:
                    if xi not in special_token:
                        gtp[xi] = 0.0
                else:
                    gtp[xi] = 0.0
                # Renormalize
                gtp_sum = gtp.sum()
                if gtp_sum > 0:
                    gtp = gtp / gtp_sum
                else:
                    break  # No valid tokens left

        if not accepted:
            break  # No token accepted at this step

    # Final logits for sampling next token
    # Use the best path and position = accept_length (predicts next token)
    if accept_length < seq_len:
        final_logits_input = logits[best_candidate, accept_length - 1].unsqueeze(0)  # predicts token at `accept_length`
        final_context_ids = candidates[best_candidate, :accept_length].unsqueeze(0)
        processed_logits = logits_processor(final_context_ids, final_logits_input)[0]
    else:
        # All tokens accepted; use last logits
        final_logits_input = logits[best_candidate, -1].unsqueeze(0)
        final_context_ids = candidates[best_candidate, :].unsqueeze(0)
        processed_logits = logits_processor(final_context_ids, final_logits_input)[0]

    sample_logits = torch.softmax(processed_logits, dim=0)  # [V]

    return torch.tensor(best_candidate, device=device), torch.tensor(accept_length - 1, device=device), sample_logits


@torch.no_grad()
def update_inference_inputs(input_ids, candidates, best_candidate_index, accept_length, retrieve_indices, logits_processor,
        model, hidden_state_new, sample_p):
    
    new_tokens = None
    select_indices = None
    prev_input_len = input_ids.shape[1]
    
    bc = best_candidate_index[0]
    al = accept_length[0]
    new_tokens = candidates[0, bc, : al + 1].unsqueeze(0)   # 提取 tokens [1, Len]
    select_indices = retrieve_indices[0, bc, : al + 1] + prev_input_len
    # 更新验证通过token
    input_ids = torch.cat([input_ids, new_tokens.to(input_ids.device)], dim=-1)   # 更新 Input IDs
    
    # 更新 KV Cache
    for past_key_values_data in model.past_key_values_data:
        tgt = past_key_values_data.index_select(dim=-2, index=select_indices.to(past_key_values_data.device))
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        model.current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # 构建下一轮draft输入token，采样 Bonus Token 
    prob = sample_p 
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None] if token.ndim == 1 else token # Ensure [B, 1]
    else:
        token = torch.argmax(prob, dim=-1, keepdim=True) # [B, 1]
    draft_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # 提取 Hidden States    
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices[0]] # [1, Leaf, Depth, H]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate_index[0], : accept_length[0] + 1]

    # 下一轮draft
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_generate(
        accept_hidden_state_new,
        input_ids=draft_input_ids,
        head=model.base_model.lm_head,
        logits_processor=logits_processor
    )

    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def stack_with_left_padding(tensor_list, pad_id, device, return_mask=False): 
    """
    将不等长的 Tensor 列表堆叠为 Batch，使用 Left Padding。
    支持 1D (Tokens) 和 2D (Hidden States) 输入。
    """
    if not tensor_list:
        return None
    
    batch_size = len(tensor_list)
    max_len = max(t.size(0) for t in tensor_list)
    trailing_dims = list(tensor_list[0].shape[1:])   # 获取除长度外的其他维度形状
    target_shape = [batch_size, max_len] + trailing_dims   # 构建目标形状
    
    padded_tensor = torch.full(target_shape, pad_id, dtype=tensor_list[0].dtype, device=device)
    
    if return_mask:
        padding_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, t in enumerate(tensor_list):
        length = t.size(0)
        # Left Padding
        start_idx = max_len - length
        padded_tensor[i, start_idx:] = t
        if return_mask:
            padding_mask[i, start_idx:] = 1

    if return_mask:
        return padded_tensor, padding_mask
    return padded_tensor





if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
