# coding=utf-8
"""
Eagle Layer2: EAGLE2 适配版本

本文件适配 EAGLE2 架构 (对应 cnets1.py)，用于支持 Vicuna 等使用 EAGLE2 的模型。

EAGLE2 与 EAGLE3 的主要区别：
1. Decoder 层数：EAGLE2 使用多层 Decoder Stack，EAGLE3 使用单层
2. 注意力输入维度：EAGLE2 使用 hidden_size，EAGLE3 使用 hidden_size * 2
3. fc 层输入维度：EAGLE2 使用 hidden_size * 2，EAGLE3 使用 hidden_size * 3
4. lm_head：EAGLE2 使用外部传入的 head，EAGLE3 使用内置的 lm_head
5. DecoderLayer：EAGLE2 使用标准 LlamaDecoderLayer，EAGLE3 使用特殊的 LlamaDecoderLayeremb

继承关系：
- 本模块尽可能复用 eagle_layer.py (EAGLE3) 中的公共组件
- 仅重写必要的组件：EagleAttention2, EagleDecoderLayer2, EagleLayer2
"""

import math
import os
import json
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers.activations import ACT2FN

from .configs import EConfig
from .utils_c import *

# 复用 eagle_layer3.py 中的公共组件
from .eagle_layer3 import (
    _make_causal_mask,
    _expand_mask,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    EagleMLP,
)


# =============================================================================
# EAGLE2 Attention Layer
# =============================================================================

class EagleAttention2(nn.Module):
    """
    EAGLE2 的注意力模块
    
    与 EAGLE3 的区别：
    - 输入维度是 hidden_size (不是 hidden_size * 2)
    - 用于标准的单输入 Decoder Layer
    """

    def __init__(self, config: EConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 检查是否有 qkv_bias 配置
        qkv_bias = getattr(config, 'qkv_bias', False)
        
        # 标准投影层（输入维度是 hidden_size）
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self._init_rope(config)

    def _init_rope(self, config):
        """初始化旋转位置编码"""
        if config.rope_scaling is None:
            base = getattr(config, 'rope_theta', 10000)
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
            )
        else:
            # 可扩展其他 RoPE 变体
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # GQA: 重复 KV heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# =============================================================================
# EAGLE2 Decoder Layer
# =============================================================================

class EagleDecoderLayer2(nn.Module):
    """
    EAGLE2 的 Decoder 层
    
    与 EAGLE3 的区别：
    - 标准单输入接口（不需要额外的 input_emb 参数）
    - 第一层没有 input_layernorm
    - 用于多层堆叠
    """

    def __init__(self, config: EConfig, layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = EagleAttention2(config)
        self.mlp = EagleMLP(config)
        self.layer_index = layer_index
        
        # 第一层没有 input_layernorm
        if layer_index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        标准 Decoder Layer 前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_value: KV Cache
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用缓存
            
        Returns:
            outputs: (hidden_states, [attn_weights], [present_kv])
        """
        residual = hidden_states

        # 第一层跳过 input_layernorm
        if self.layer_index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# =============================================================================
# EAGLE2 Layer 主类
# =============================================================================

class EagleLayer2(nn.Module):
    """
    EAGLE2 Layer: 多层 Decoder Stack 版本
    
    与 EAGLE3 的区别：
    1. 多层结构：使用 ModuleList 存储多个 Decoder Layer
    2. fc 层：输入是 hidden_size * 2 (embedding + hidden)
    3. 无内置 lm_head：使用外部传入的 head（基础模型的 lm_head）
    
    Attributes:
        config: 模型配置
        embed_tokens: Token 嵌入层
        layers: Decoder Layer 堆叠
        fc: 特征融合层 (embedding + hidden -> hidden)
        
        # Tree 生成参数
        total_tokens: 最大生成 token 数
        depth: 树的最大深度
        top_k: 每层保留的候选数
        threshold: 接受阈值
        
        # KV Cache
        stable_kv: 稳定的 KV Cache (已接受的 tokens)
    """

    def __init__(
        self,
        config: EConfig,
        load_emb: bool = False,
        path: str = None,
        bias: bool = True,
        total_tokens: int = 63,
        depth: int = 5,
        top_k: int = 8,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token 嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 加载预训练嵌入 (可选)
        if load_emb:
            self._load_embeddings(path)

        # Tree 生成参数
        self.top_k = top_k
        self.total_tokens = total_tokens - 1  # 减去 root
        self.depth = depth
        self.threshold = math.log(threshold)

        # 网络结构：多层 Decoder
        self.layers = nn.ModuleList([
            EagleDecoderLayer2(config, index) 
            for index in range(config.num_hidden_layers)
        ])
        
        # 特征融合层 (embedding + hidden -> hidden)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # 冻结嵌入层
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        # 外部 lm_head（由 from_pretrained 设置）
        self.head = None
        self.diff_device = False
        self.headweight = None
        
        self.reset_state()

    @classmethod
    def from_pretrained(
        cls,
        ea_model_path: str,
        base_model: nn.Module,
        base_model_name_or_path: str,
        use_eagle3: bool = False,  # EAGLE2 固定为 False
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
    ):
        """
        从预训练模型加载 Eagle Layer2
        
        Args:
            ea_model_path: Eagle 模型目录路径
            base_model: 基础模型实例
            base_model_name_or_path: 基础模型路径
            use_eagle3: 是否使用 Eagle3（EAGLE2 固定为 False）
            total_token: 每次 draft 生成的总 token 数
            depth: draft 树的深度
            top_k: 每层选择的 top-k 数量
            threshold: 接受阈值
            
        Returns:
            初始化好的 EagleLayer2 实例
        """
        # =====================================================================
        # 1. 加载配置
        # =====================================================================
        config_path = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(ea_model_path, "config.json")
        
        config = EConfig.from_pretrained(config_path)
        with open(config_path, "r") as f:
            con = json.loads(f.read())
        bias = con.get("bias", True)

        # =====================================================================
        # 2. 加载权重
        # =====================================================================
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        
        try:
            load_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_path):
                load_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_state_dict = torch.load(load_path, map_location=device)
        except:
            from safetensors.torch import load_file
            load_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_path):
                load_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_state_dict = load_file(load_path)

        # =====================================================================
        # 3. 创建 Eagle Layer2 实例
        # =====================================================================
        eagle_layer = cls(
            config=config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=base_model_name_or_path,
            load_emb=True,
        )

        # =====================================================================
        # 4. 设置外部 lm_head
        # =====================================================================
        eagle_layer.head = base_model.lm_head
        
        # 处理跨设备权重
        if device != base_model.lm_head.weight.device:
            eagle_layer.diff_device = True
            eagle_layer.headweight = base_model.lm_head.weight.clone().to(device)
        else:
            eagle_layer.diff_device = False

        # =====================================================================
        # 5. 加载权重并移动到正确设备
        # =====================================================================
        eagle_layer.load_state_dict(ea_state_dict, strict=False)
        eagle_layer.to(base_model.dtype).to(device)
        
        eagle_layer.reset_state()
        
        return eagle_layer

    def _load_embeddings(self, path: str):
        """从基础模型加载预训练嵌入"""
        from safetensors import safe_open
        try:
            index_path = os.path.join(path, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                index_path = hf_hub_download(path, "model.safetensors.index.json")
            with open(index_path, "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            local_path = os.path.join(path, emb_path)
            if not os.path.exists(local_path):
                local_path = hf_hub_download(path, emb_path)
            with safe_open(local_path, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except:
            index_path = os.path.join(path, "pytorch_model.bin.index.json")
            if not os.path.exists(index_path):
                index_path = hf_hub_download(path, "pytorch_model.bin.index.json")
            with open(index_path, "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            local_path = os.path.join(path, emb_path)
            if not os.path.exists(local_path):
                local_path = hf_hub_download(path, emb_path)
            weights = torch.load(local_path)
            tensor = weights["model.embed_tokens.weight"].float()
        
        self.embed_tokens.weight.data = tensor

    # =========================================================================
    # 初始化和状态管理
    # =========================================================================

    def reset_state(self):
        """重置状态"""
        self.stable_kv = None
        self.cache_padding_mask = None
        self.full_position_ids = None
        self.tree_mask = None
        
        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)
        
        # 分布式同步回调
        self._stable_kv_sync_callback = None
    
    def set_stable_kv_sync_callback(self, callback):
        """设置 stable_kv 同步回调函数"""
        self._stable_kv_sync_callback = callback

    # =========================================================================
    # 注意力掩码构建
    # =========================================================================

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """构建解码器注意力掩码"""
        combined_attention_mask = None
        
        # 因果掩码
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # 扩展 attention mask
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # 应用 Tree Mask
        if self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        # 应用 Cache Padding Mask (并行解码用)
        if self.cache_padding_mask is not None:
            padding_bool = (self.cache_padding_mask == 0)
            current_src_len = combined_attention_mask.shape[-1]
            mask_len = padding_bool.shape[1]
            bsz = padding_bool.shape[0]
            
            if current_src_len > mask_len:
                diff = current_src_len - mask_len
                new_valid = torch.zeros((bsz, diff), dtype=torch.bool, device=padding_bool.device)
                padding_bool = torch.cat([padding_bool, new_valid], dim=1)
            
            padding_bool = padding_bool[:, None, None, :]
            combined_attention_mask = combined_attention_mask.masked_fill(
                padding_bool, torch.finfo(torch.float32).min
            )

        return combined_attention_mask

    # =========================================================================
    # 前向传播
    # =========================================================================

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        EAGLE2 前向传播
        
        与 EAGLE3 的区别：
        - 使用 fc 融合 embedding 和 hidden states: fc(cat(emb, hidden))
        - 多层 Decoder 堆叠
        
        Args:
            hidden_states: 基础模型的 hidden states [batch, seq, hidden]
            input_ids: 输入 token IDs [batch, seq]
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_values: KV Cache (list of tuples for each layer)
            use_cache: 是否使用 cache
            
        Returns:
            hidden_states: 输出 hidden states
            next_decoder_cache: 更新后的 KV Cache
        """
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # 获取 token embeddings
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        # 处理 KV Cache
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length + past_key_values_length

        # 处理 position_ids
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # 构建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # EAGLE2: fc(cat(embedding, hidden)) -> hidden
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        # 多层 Decoder
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    # =========================================================================
    # 获取 lm_head 输出
    # =========================================================================

    def _get_head_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取 lm_head 输出
        
        EAGLE2 使用外部传入的 head（基础模型的 lm_head）
        """
        if self.diff_device:
            # 使用复制的权重
            return nn.functional.linear(hidden_states, self.headweight)
        else:
            return self.head(hidden_states)

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
        self.tree_mask = None

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

        # Phase 2: Tree Growth
        loop_scores, loop_parents, loop_tokens, _ = self._grow_tree(
            last_hidden, next_input_ids, scores, bsz, self.top_k, self.depth, len_posi
        )
        scores_list.extend(loop_scores)
        parents_list.extend(loop_parents)
        tokens_list.extend(loop_tokens)

        # Phase 3: Post Process
        return self._post_process_tree(
            bsz, scores_list, tokens_list, parents_list,
            sample_token, self.total_tokens, self.top_k
        )

    # =========================================================================
    # Phase 1: Root Expansion
    # =========================================================================

    def _expand_root(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        prefix_len: int = -1,
        active_branch: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根节点扩展"""
        actual_hidden = hidden_states

        # 根据当前状态确定输入
        if self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            
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
        if self.full_position_ids is not None:
            position_start = self.stable_kv[0][0].shape[2]
            step = actual_input.shape[1]
            position_ids = self.full_position_ids[:, position_start:position_start + step]

        # EAGLE2 Forward
        out_hidden, past_key_values = self(
            actual_hidden,
            input_ids=actual_input,
            position_ids=position_ids,
            past_key_values=self.stable_kv,
            use_cache=True,
        )
        
        self.stable_kv = past_key_values
        
        # 生成 top-k 候选
        last_hidden = out_hidden[:, -1]
        last_headout = self._get_head_output(last_hidden)
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # EAGLE2 没有词表映射
        next_token = topk_index
        next_input_ids = topk_index

        return scores, parents, next_token, next_input_ids, last_hidden

    # =========================================================================
    # Phase 2: Tree Growth
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
        """树生长"""
        input_hidden = last_hidden[:, None, :].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init.repeat(bsz, 1, 1, 1)
        local_range = torch.arange(top_k, device=self.embed_tokens.weight.device)
        past_key_values = self.stable_kv
        
        loop_scores = []
        loop_parents = []
        loop_tokens = []

        for i in range(depth):
            self.tree_mask = tree_mask
            
            # 计算位置编码
            if self.full_position_ids is not None:
                root_pos = self.full_position_ids[:, -1]
                current_pos = root_pos + i + 1
                position_ids = current_pos.unsqueeze(1).expand(-1, top_k)
            else:
                position_ids = len_posi - 1 + self.position_ids
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)

            # EAGLE2 Forward
            out_hidden, past_key_values = self(
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
            last_headout = self._get_head_output(out_hidden)
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
            out_ids_expanded = out_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            input_hidden = torch.gather(out_hidden, 1, out_ids_expanded)

            # EAGLE2 没有词表映射
            loop_tokens.append(local_topk_index)
            flat_source = local_topk_index.view(bsz, -1)
            input_ids = torch.gather(flat_source, 1, selected_indices)
            loop_scores.append(cu_scores)

            # 更新 Tree Mask
            new_masks = []
            for b in range(bsz):
                current_mask = tree_mask[b:b+1]
                selected_cols = current_mask[:, :, :, out_ids[b]]
                init_mask = self.tree_mask_init
                new_mask = torch.cat((selected_cols, init_mask), dim=3)
                new_masks.append(new_mask)
            tree_mask = torch.cat(new_masks, dim=0)

        return loop_scores, loop_parents, loop_tokens, tree_mask

    # =========================================================================
    # Phase 3: Post Process
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
            b_scores = [scores_list[0][b].view(-1)] + [s[b].view(-1) for s in scores_list[1:]]
            flat_scores = torch.cat(b_scores, dim=0)
            
            b_tokens = [tokens_list[0][b].view(-1)] + [t[b].view(-1) for t in tokens_list[1:]]
            flat_tokens = torch.cat(b_tokens, dim=0)
            
            b_parents = [parents_list[0][b].view(-1)] + [p[b].view(-1) for p in parents_list[1:]]
            flat_parents = torch.cat(b_parents, dim=0)

            # 选择 top-total_tokens 个节点
            top_scores = torch.topk(flat_scores, total_tokens, dim=-1)
            top_indices = torch.sort(top_scores.indices).values

            draft_tokens_b = flat_tokens[top_indices]
            draft_tokens_b = torch.cat((sample_token[b].unsqueeze(0), draft_tokens_b), dim=0)

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
    # 分布式 Prefill 支持（可选）
    # =========================================================================

    @torch.no_grad()
    def generate_draft_tree_dist_prefill(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        is_last_chunk: bool = False,
        chunk_idx: int = 0,
        original_input_len: int = -1,
    ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """分布式 Prefill 专用的 Draft Tree 生成"""
        bsz = input_ids.shape[0]
        input_ids = input_ids.to(hidden_states.device)
        
        # 记录之前的 KV 长度
        prev_kv_len = 0
        if self.stable_kv is not None:
            prev_kv_len = self.stable_kv[0][0].shape[2]
        
        # Phase 1: Expand Root
        scores, parents, next_token, next_input_ids, last_hidden = self._expand_root_dist_prefill(
            hidden_states, input_ids, chunk_idx
        )
        
        # 计算增量 KV cache
        incremental_kv = None
        if self.stable_kv is not None:
            # 对于多层，返回所有层的增量
            incremental_kvs = []
            for layer_kv in self.stable_kv:
                key, value = layer_kv
                current_len = key.shape[2]
                if current_len > prev_kv_len:
                    new_key = key[:, :, prev_kv_len:current_len, :].clone()
                    new_value = value[:, :, prev_kv_len:current_len, :].clone()
                    incremental_kvs.append((new_key, new_value))
            if incremental_kvs:
                incremental_kv = tuple(incremental_kvs)
        
        # 非最后 chunk
        if not is_last_chunk:
            return None, incremental_kv
        
        # 最后 chunk：执行完整的 tree growth 和 post process
        sample_token = input_ids[:, -1]
        len_posi = original_input_len if original_input_len > 0 else input_ids.shape[1] + 1
        self.tree_mask = None
        
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
        actual_input = input_ids
        actual_hidden = hidden_states
        
        position_ids = None
        if self.full_position_ids is not None and self.stable_kv is not None:
            position_start = self.stable_kv[0][0].shape[2]
            step = actual_input.shape[1]
            position_ids = self.full_position_ids[:, position_start:position_start + step]
        
        out_hidden, past_key_values = self(
            actual_hidden,
            input_ids=actual_input,
            position_ids=position_ids,
            past_key_values=self.stable_kv,
            use_cache=True,
        )
        
        self.stable_kv = past_key_values
        
        last_hidden = out_hidden[:, -1]
        last_headout = self._get_head_output(last_hidden)
        last_p = self.logsoftmax(last_headout)
        
        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        
        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        
        # EAGLE2 没有词表映射
        next_token = topk_index
        next_input_ids = topk_index
        
        return scores, parents, next_token, next_input_ids, last_hidden
