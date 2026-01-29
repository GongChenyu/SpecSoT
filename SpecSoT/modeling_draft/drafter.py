# coding=utf-8
"""
Drafter - Draft Tree 生成器

独立于 Eagle Layer 的 Draft Tree 生成逻辑。
使用 Eagle3 的实现作为基准，支持词表映射。

主要方法：
- generate_draft_tree(): 主入口
- generate_draft_tree_dist_prefill(): 分布式版本
- _expand_root(): 根节点扩展
- _expand_root_dist_prefill(): 分布式根节点扩展
- _grow_tree(): 树生长
- _post_process_tree(): 后处理
- _apply_vocab_mapping(): 词表映射
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Drafter:
    """
    Draft Tree 生成器

    与 Eagle Layer 解耦，负责所有 Draft Tree 生成逻辑。
    使用 Eagle3 的实现作为基准，通过 _apply_vocab_mapping 兼容 Eagle2。

    Attributes:
        eagle_layer: Eagle Layer 实例 (EagleLayer2 或 EagleLayer3)
        config: Eagle 配置
        top_k: 每层保留的候选数
        total_tokens: 最大生成 token 数
        depth: 树的最大深度
    """

    def __init__(self, eagle_layer: nn.Module):
        """
        初始化 Drafter

        Args:
            eagle_layer: Eagle Layer 实例
        """
        self.eagle_layer = eagle_layer
        self.config = eagle_layer.config
        self.top_k = eagle_layer.top_k
        self.total_tokens = eagle_layer.total_tokens
        self.depth = eagle_layer.depth
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    # =========================================================================
    # 词表映射（关键：兼容 Eagle2/3）
    # =========================================================================

    def _apply_vocab_mapping(self, topk_index: torch.Tensor) -> torch.Tensor:
        """
        应用词表映射

        Eagle3: 如果 vocab_size != draft_vocab_size，需要映射
        Eagle2: vocab_size == draft_vocab_size，直接返回
                Eagle2 的 config 可能没有 draft_vocab_size 属性

        Args:
            topk_index: top-k 索引 [batch, k] 或 [batch, k, k]

        Returns:
            映射后的 token 索引
        """
        # 获取 draft_vocab_size，如果不存在则默认为 vocab_size（Eagle2 兼容）
        draft_vocab_size = getattr(self.config, 'draft_vocab_size', self.config.vocab_size)
        
        if self.config.vocab_size == draft_vocab_size:
            return topk_index
        else:
            return topk_index + self.eagle_layer.d2t[topk_index]

    # =========================================================================
    # Draft Tree 生成：主入口
    # =========================================================================

    @torch.no_grad()
    def generate_draft_tree(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        prefix_len: int = -1,
        active_branch: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成 Draft Tree

        三阶段流程：
        1. Root Expansion: 生成 top-k 个根候选（cache 需要保留）
        2. Tree Growth: 递归扩展树（cache 是临时的）
        3. Post Process: 构建最终树结构
        4. Cache Cleanup: 重置 cache 到 expand_root 后的长度

        Args:
            hidden_states: 基础模型的 hidden states
            input_ids: 当前输入 token IDs
            prefix_len: 前缀长度 (用于并行模式)
            active_branch: 活跃分支列表 (用于并行模式)

        Returns:
            draft_tokens: 候选 token 树 [batch, total_nodes]
            retrieve_indices: 叶节点到根的路径 [batch, num_leaves, depth]
            tree_mask: 树的注意力掩码 [batch, 1, nodes, nodes]
            tree_position_ids: 节点位置编码 [batch, nodes]
        """
        bsz = input_ids.shape[0]
        input_ids = input_ids.to(hidden_states.device)
        sample_token = input_ids[:, -1]
        len_posi = input_ids.shape[1]
        self.eagle_layer.tree_mask = None

        scores_list = []
        parents_list = []
        tokens_list = []

        # Phase 1: Root Expansion
        scores, parents, next_token, next_input_ids, last_hidden = self._expand_root(
            hidden_states, input_ids, prefix_len=prefix_len, active_branch=active_branch
        )
        scores_list.append(scores)
        parents_list.append(parents)
        tokens_list.append(next_token)
        
        # 记录 expand_root 后的 cache 长度（用于之后重置）
        expand_root_cache_len = self.eagle_layer.get_kv_cache_length()

        # Phase 2: Tree Growth
        loop_scores, loop_parents, loop_tokens, _ = self._grow_tree(
            last_hidden, next_input_ids, scores, bsz, self.top_k, self.depth, len_posi
        )
        scores_list.extend(loop_scores)
        parents_list.extend(loop_parents)
        tokens_list.extend(loop_tokens)
        
        # Phase 4: Cache Cleanup - 重置 cache 到 expand_root 后的长度
        # tree_grow 阶段产生的 cache 是临时的，需要丢弃
        if self.eagle_layer.kv_cache_initialized:
            self.eagle_layer.set_kv_cache_length(expand_root_cache_len)

        # Phase 3: Post Process
        return self._post_process_tree(
            bsz, scores_list, tokens_list, parents_list,
            sample_token, self.total_tokens, self.top_k
        )

    # =========================================================================
    # Phase 1: Root Expansion (根节点扩展)
    # =========================================================================

    def _expand_root(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        prefix_len: int = -1,
        active_branch: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根节点扩展：从单个 hidden state 生成 top-k 个候选

        处理三种模式：
        1. Skeleton Prefill: 首次运行，无 KV Cache
        2. Skeleton Decoding: 单序列解码
        3. Parallel Decoding: 多分支并行解码
        """
        eagle = self.eagle_layer
        actual_hidden = hidden_states

        # 获取 KV cache 长度（兼容 KVCache 类和 tuple 格式）
        kv_len = eagle.get_kv_cache_length() if eagle.kv_cache_initialized else 0
        if not eagle.kv_cache_initialized and eagle.draft_past_key_values is not None:
            # 回退到旧的 tuple 格式
            kv_len = eagle.draft_past_key_values[0][0].shape[2] if hasattr(eagle.draft_past_key_values[0][0], 'shape') else eagle.draft_past_key_values[0][0].shape[-2]

        # 根据当前状态确定输入
        if kv_len > 0:
            if input_ids.shape[0] == 1:
                if active_branch is not None:
                    actual_input = input_ids
                else:
                    actual_input = input_ids[:, 1:]
                    actual_input = actual_input[:, kv_len:]
            elif hidden_states.shape[1] != input_ids.shape[1]:
                actual_input = input_ids[:, 1:]
            else:
                actual_input = input_ids
        else:
            actual_input = input_ids[:, 1:]

        # 处理位置编码
        if eagle.full_position_ids is not None and kv_len > 0:
            position_start = kv_len
            step = actual_input.shape[1]
            position_ids = eagle.full_position_ids[:, position_start:position_start + step]

        # Eagle Layer Forward
        out_hidden, past_key_values = eagle(
            actual_hidden,
            input_ids=actual_input,
            position_ids=position_ids,
            past_key_values=eagle.draft_past_key_values,
            use_cache=True,
        )

        # 如果使用 KVCache 类，draft_past_key_values 已经原地更新，不需要重新赋值
        # 但为了兼容旧的 tuple 格式，仍然保留赋值
        if not eagle.kv_cache_initialized:
            eagle.draft_past_key_values = past_key_values

        # 生成 top-k 候选
        last_hidden = out_hidden[:, -1]
        last_headout = eagle.get_head_output(last_hidden)
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # 应用词表映射
        mapped_tokens = self._apply_vocab_mapping(topk_index)
        next_token = mapped_tokens
        next_input_ids = mapped_tokens

        return scores, parents, next_token, next_input_ids, last_hidden

    # =========================================================================
    # Phase 2: Tree Growth (树生长)
    # =========================================================================

    def _grow_tree(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        bsz: int,
        top_k: int,
        depth: int,
        len_posi: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        树生长：递归扩展候选树

        每层：
        1. 对当前 top-k 节点生成 top-k 个子节点
        2. 从 k*k 个候选中选择全局 top-k
        3. 更新 tree mask
        """
        eagle = self.eagle_layer
        hidden_size = eagle.hidden_size

        input_hidden = last_hidden[:, None, :].repeat(1, top_k, 1)
        tree_mask = eagle.tree_mask_init.repeat(bsz, 1, 1, 1)
        local_range = torch.arange(top_k, device=eagle.embed_tokens.weight.device)
        past_key_values = eagle.draft_past_key_values

        loop_scores = []
        loop_parents = []
        loop_tokens = []

        for i in range(depth):
            eagle.tree_mask = tree_mask

            # 计算位置编码
            if eagle.full_position_ids is not None:
                root_pos = eagle.full_position_ids[:, -1]
                current_pos = root_pos + i + 1
                position_ids = current_pos.unsqueeze(1).expand(-1, top_k)
            else:
                position_ids = len_posi - 1 + eagle.position_ids
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)

            # Eagle Layer Forward
            out_hidden, past_key_values = eagle(
                input_hidden, input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids, use_cache=True
            )

            # 计算父节点索引
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (local_range + bias)
            loop_parents.append(parents.unsqueeze(0).repeat(bsz, 1))

            # 预测下一层
            last_headout = eagle.get_head_output(out_hidden)
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            local_topk_index, topk_p = top.indices, top.values

            # 累积分数
            cu_scores = topk_p + scores[:, :, None]

            # 全局 Top-K 选择
            topk_cs = torch.topk(cu_scores.view(bsz, -1), top_k, dim=-1)
            selected_indices, selected_scores = topk_cs.indices, topk_cs.values
            scores = selected_scores

            # 回溯父节点
            out_ids = selected_indices // top_k
            out_ids_expanded = out_ids.unsqueeze(-1).expand(-1, -1, hidden_size)
            input_hidden = torch.gather(out_hidden, 1, out_ids_expanded)

            # 应用词表映射并保存 tokens
            mapped_tokens = self._apply_vocab_mapping(local_topk_index)
            loop_tokens.append(mapped_tokens)
            flat_source = mapped_tokens.view(bsz, -1)
            input_ids = torch.gather(flat_source, 1, selected_indices)
            loop_scores.append(cu_scores)

            # 更新 Tree Mask
            new_masks = []
            for b in range(bsz):
                current_mask = tree_mask[b:b+1]
                selected_cols = current_mask[:, :, :, out_ids[b]]
                init_mask = eagle.tree_mask_init
                new_mask = torch.cat((selected_cols, init_mask), dim=3)
                new_masks.append(new_mask)
            tree_mask = torch.cat(new_masks, dim=0)

        return loop_scores, loop_parents, loop_tokens, tree_mask

    # =========================================================================
    # Phase 3: Post Process (后处理)
    # =========================================================================

    def _post_process_tree(
        self,
        bsz: int,
        scores_list: List[torch.Tensor],
        tokens_list: List[torch.Tensor],
        parents_list: List[torch.Tensor],
        sample_token: torch.Tensor,
        total_tokens: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """后处理：构建最终的 draft tree 结构"""
        draft_tokens_list = []
        retrieve_indices_list = []
        tree_mask_list = []
        tree_position_ids_list = []

        for b in range(bsz):
            # 展平所有层的数据
            b_scores = [scores_list[0][b].view(-1)]
            b_scores += [s[b].view(-1) for s in scores_list[1:]]
            flat_scores = torch.cat(b_scores, dim=0)

            b_tokens = [tokens_list[0][b].view(-1)]
            b_tokens += [t[b].view(-1) for t in tokens_list[1:]]
            flat_tokens = torch.cat(b_tokens, dim=0)

            b_parents = [parents_list[0][b].view(-1)]
            b_parents += [p[b].view(-1) for p in parents_list[1:]]
            flat_parents = torch.cat(b_parents, dim=0)

            # 选择 top-total_tokens 个节点
            top_scores = torch.topk(flat_scores, total_tokens, dim=-1)
            top_indices = torch.sort(top_scores.indices).values

            draft_tokens_b = flat_tokens[top_indices]
            draft_tokens_b = torch.cat(
                (sample_token[b].unsqueeze(0), draft_tokens_b), dim=0
            )

            # 构建 mask 索引
            raw_parents = flat_parents[top_indices // top_k].long()
            mask_index = torch.searchsorted(top_indices, raw_parents - 1, right=False)
            mask_index[raw_parents == 0] = -1
            mask_index = mask_index + 1
            mask_index_list = mask_index.tolist()

            # 构建 tree mask
            tree_mask_b = torch.eye(total_tokens + 1).bool()
            tree_mask_b[:, 0] = True
            for i in range(total_tokens):
                tree_mask_b[i + 1].add_(tree_mask_b[mask_index_list[i]])

            tree_position_ids_b = torch.sum(tree_mask_b, dim=1) - 1

            # 构建 retrieve indices
            max_depth = torch.max(tree_position_ids_b) + 1
            noleaf_index = torch.unique(mask_index).tolist()
            leaf_num = total_tokens - (len(noleaf_index) - 1)
            retrieve_indices_b = [[-1] * max_depth.item() for _ in range(leaf_num)]

            rid = 0
            position_list = tree_position_ids_b.tolist()
            for i in range(total_tokens + 1):
                if i not in noleaf_index:
                    cid = i
                    depth_i = position_list[i]
                    for j in reversed(range(depth_i + 1)):
                        retrieve_indices_b[rid][j] = cid
                        cid = mask_index_list[cid - 1]
                    rid += 1

            # 排序
            def custom_sort(lst):
                return [lst[i] if lst[i] >= 0 else total_tokens + 5 for i in range(len(lst))]
            retrieve_indices_b = sorted(retrieve_indices_b, key=custom_sort)

            draft_tokens_list.append(draft_tokens_b)
            retrieve_indices_list.append(torch.tensor(retrieve_indices_b, dtype=torch.long))
            tree_mask_list.append(tree_mask_b.float())
            tree_position_ids_list.append(tree_position_ids_b)

        # 批次对齐和堆叠
        draft_tokens = torch.stack(draft_tokens_list, dim=0)
        tree_mask = torch.stack(tree_mask_list, dim=0)[:, None, :, :]
        tree_position_ids = torch.stack(tree_position_ids_list, dim=0).to(sample_token.device)

        # 对齐 retrieve indices
        max_depth = max([ri.shape[1] for ri in retrieve_indices_list])
        max_leaves = max([ri.shape[0] for ri in retrieve_indices_list])

        padded_retrieve = []
        for ri in retrieve_indices_list:
            curr_l, curr_d = ri.shape
            pad_d = max_depth - curr_d
            if pad_d > 0:
                ri = torch.cat((ri, torch.full((curr_l, pad_d), -1, dtype=torch.long)), dim=1)
            pad_l = max_leaves - curr_l
            if pad_l > 0:
                ri = torch.cat((ri, torch.full((pad_l, max_depth), -1, dtype=torch.long)), dim=0)
            padded_retrieve.append(ri)

        retrieve_indices = torch.stack(padded_retrieve, dim=0)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    # =========================================================================
    # 分布式 Prefill 支持
    # =========================================================================

    @torch.no_grad()
    def generate_draft_tree_dist_prefill(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        is_last_chunk: bool = False,
        chunk_idx: int = 0,
        original_input_len: int = -1,
    ) -> Tuple[
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        分布式 Prefill 专用的 Draft Tree 生成

        Args:
            hidden_states: 基础模型的 hidden states
            input_ids: 当前 chunk 的 token IDs
            is_last_chunk: 是否是最后一个 chunk
            chunk_idx: chunk 索引
            original_input_len: 原始完整 input_ids 的长度

        Returns:
            (tree_result, incremental_kv)
        """
        eagle = self.eagle_layer
        bsz = input_ids.shape[0]
        input_ids = input_ids.to(hidden_states.device)

        # 记录之前的 KV 长度（兼容 KVCache 类和 tuple 格式）
        prev_kv_len = eagle.get_kv_cache_length() if eagle.kv_cache_initialized else 0
        if not eagle.kv_cache_initialized and eagle.draft_past_key_values is not None:
            prev_kv_len = eagle.draft_past_key_values[0][0].shape[2]

        # Phase 1: Expand Root
        scores, parents, next_token, next_input_ids, last_hidden = \
            self._expand_root_dist_prefill(hidden_states, input_ids, chunk_idx)

        # 记录 expand_root 后的 cache 长度（用于之后重置）
        expand_root_cache_len = eagle.get_kv_cache_length() if eagle.kv_cache_initialized else 0

        # 计算增量 KV cache
        incremental_kv = None
        if eagle.kv_cache_initialized and eagle.draft_past_key_values is not None:
            # 使用 KVCache 类
            key_cache, value_cache = eagle.draft_past_key_values[0]
            current_len = key_cache.shape[2]  # KVCache.shape[2] 是 current_length
            if current_len > prev_kv_len:
                # 从 KVCache 的底层数据中提取增量部分
                new_key = key_cache.data[:, :, prev_kv_len:current_len, :].clone()
                new_value = value_cache.data[:, :, prev_kv_len:current_len, :].clone()
                incremental_kv = (new_key, new_value)
        elif eagle.draft_past_key_values is not None:
            # 旧的 tuple 格式
            key, value = eagle.draft_past_key_values[0]
            current_len = key.shape[2]
            if current_len > prev_kv_len:
                new_key = key[:, :, prev_kv_len:current_len, :].clone()
                new_value = value[:, :, prev_kv_len:current_len, :].clone()
                incremental_kv = (new_key, new_value)

        # 非最后 chunk
        if not is_last_chunk:
            return None, incremental_kv

        # 最后 chunk：执行完整的 tree growth 和 post process
        sample_token = input_ids[:, -1]
        len_posi = original_input_len if original_input_len > 0 else input_ids.shape[1] + 1
        eagle.tree_mask = None

        scores_list = [scores]
        parents_list = [parents]
        tokens_list = [next_token]

        # Phase 2: Tree Growth
        loop_scores, loop_parents, loop_tokens, _ = self._grow_tree(
            last_hidden, next_input_ids, scores, bsz, self.top_k, self.depth, len_posi
        )
        scores_list.extend(loop_scores)
        parents_list.extend(loop_parents)
        tokens_list.extend(loop_tokens)

        # Phase 4: Cache Cleanup - 重置 cache 到 expand_root 后的长度
        if eagle.kv_cache_initialized:
            eagle.set_kv_cache_length(expand_root_cache_len)

        # Phase 3: Post Process
        tree_result = self._post_process_tree(
            bsz, scores_list, tokens_list, parents_list,
            sample_token, self.total_tokens, self.top_k
        )
        return tree_result, incremental_kv

    def _expand_root_dist_prefill(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        chunk_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分布式 Prefill 专用的根节点扩展"""
        eagle = self.eagle_layer
        actual_input = input_ids
        actual_hidden = hidden_states

        # 获取 KV cache 长度（兼容 KVCache 类和 tuple 格式）
        kv_len = eagle.get_kv_cache_length() if eagle.kv_cache_initialized else 0
        if not eagle.kv_cache_initialized and eagle.draft_past_key_values is not None:
            kv_len = eagle.draft_past_key_values[0][0].shape[2]

        position_ids = None
        if eagle.full_position_ids is not None and kv_len > 0:
            position_start = kv_len
            step = actual_input.shape[1]
            position_ids = eagle.full_position_ids[:, position_start:position_start + step]

        out_hidden, past_key_values = eagle(
            actual_hidden,
            input_ids=actual_input,
            position_ids=position_ids,
            past_key_values=eagle.draft_past_key_values,
            use_cache=True,
        )

        # 如果使用 KVCache 类，draft_past_key_values 已经原地更新
        if not eagle.kv_cache_initialized:
            eagle.draft_past_key_values = past_key_values

        last_hidden = out_hidden[:, -1]
        last_headout = eagle.get_head_output(last_hidden)
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # 应用词表映射
        mapped_tokens = self._apply_vocab_mapping(topk_index)
        next_token = mapped_tokens
        next_input_ids = mapped_tokens

        return scores, parents, next_token, next_input_ids, last_hidden
