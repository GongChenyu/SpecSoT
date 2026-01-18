# coding=utf-8
"""
Eagle Layer: 轻量级草稿模型

Eagle Layer 是投机推理中的核心组件，负责快速生成候选 token 树。
它使用基础模型的 hidden states 和 embeddings 作为输入，预测下一批可能的 tokens。

Draft Tree 生成流程：
1. Root Expansion (根节点扩展): 从单个 hidden state 生成 top-k 个候选 token
2. Tree Growth (树生长): 递归扩展，每层保留 top-k 个最优路径
3. Post Process (后处理): 构建最终的 draft tree 结构

关键数据结构：
- draft_tokens: 候选 token 树 [batch, total_nodes]
- retrieve_indices: 叶节点到根的路径索引 [batch, num_leaves, depth]
- tree_mask: 树结构的注意力掩码 [batch, 1, nodes, nodes]
- tree_position_ids: 树节点的位置编码 [batch, nodes]

分布式适配说明：
- generate_draft_tree: 主入口，可独立部署
- _expand_root: 根节点扩展，依赖基础模型 hidden states
- _grow_tree: 树生长循环，纯 Eagle Layer 计算
- _post_process: 后处理，CPU/GPU 均可执行
"""

import math
import os
import json
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .configs import EConfig
from .utils_c import *


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# =============================================================================
# Rotary Embedding (旋转位置编码)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转张量的后半部分"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 KV heads 以匹配 Query heads 数量"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LlamaRotaryEmbedding(nn.Module):
    """Llama 风格的旋转位置编码"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# =============================================================================
# RMSNorm
# =============================================================================

class LlamaRMSNorm(nn.Module):
    """Llama 风格的 RMS 归一化"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# =============================================================================
# Attention Layer
# =============================================================================

class EagleAttention(nn.Module):
    """Eagle Layer 的注意力模块"""

    def __init__(self, config: EConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # 处理 head_dim
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # 投影层 (输入维度是 hidden_size * 2，因为拼接了 embedding 和 hidden state)
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self._init_rope(config)

    def _init_rope(self, config):
        """初始化旋转位置编码"""
        if config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
        else:
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
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
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# =============================================================================
# MLP Layer
# =============================================================================

class EagleMLP(nn.Module):
    """Eagle Layer 的 MLP 模块"""

    def __init__(self, config: EConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # 激活函数
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Decoder Layer
# =============================================================================

class EagleDecoderLayer(nn.Module):
    """Eagle Layer 的 Decoder 层"""

    def __init__(self, config: EConfig, last: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = EagleAttention(config)
        self.mlp = EagleMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.last = last
        if last:
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # 拼接 embedding 和 hidden state
        combined = torch.cat([input_emb, hidden_states], dim=-1)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = torch.cat([input_emb, hidden_states], dim=-1)

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

        # MLP (仅在最后一层)
        if self.last:
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
# Eagle Layer 主类
# =============================================================================

class EagleLayer(nn.Module):
    """
    Eagle Layer: 轻量级草稿模型
    
    负责快速生成候选 token 树，供基础模型验证。
    
    Attributes:
        config: 模型配置
        embed_tokens: Token 嵌入层
        lm_head: 语言模型头
        midlayer: Decoder 层
        fc: 特征融合层 (用于 Eagle3)
        norm: 输出归一化
        
        # Tree 生成参数
        total_tokens: 最大生成 token 数
        depth: 树的最大深度
        top_k: 每层保留的候选数
        threshold: 接受阈值
        
        # KV Cache
        stable_kv: 稳定的 KV Cache (已接受的 tokens)
        
        # 并行状态
        cache_padding_mask: 批次填充掩码
        full_position_ids: 完整位置编码
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
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        
        # 加载预训练嵌入 (可选)
        if load_emb and not hasattr(config, "target_hidden_size"):
            self._load_embeddings(path)

        # Tree 生成参数
        self.top_k = top_k
        self.total_tokens = total_tokens - 1  # 减去 root
        self.depth = depth
        self.threshold = math.log(threshold)

        # 网络结构
        self.midlayer = EagleDecoderLayer(config)
        
        # 特征融合层 (用于处理基础模型的 hidden states)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # 词表映射 (用于处理不同大小的草稿词表)
        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        # 冻结嵌入层
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        self.reset_state()

    @classmethod
    def from_pretrained(
        cls,
        ea_model_path: str,
        base_model,
        base_model_name_or_path: str,
        use_eagle3: bool,
        total_token: int,
        depth: int,
        top_k: int,
        threshold: float,
        ea_layer_state_dict: dict,
    ):
        """
        从预训练模型加载 Eagle Layer
        
        步骤：
        1. 加载配置: 从 config.json 加载 Eagle 配置
        2. 创建 Eagle Layer: 初始化草稿模型
        3. 设备管理: 处理跨设备权重
        4. 词表映射: 配置 draft vocab 到 target vocab 的映射 (Eagle3)
        5. 加载权重: 加载预训练的 Eagle 权重
        6. Tree 初始化: 初始化 draft tree 相关 buffer
        
        Args:
            ea_model_path: Eagle 模型配置路径
            base_model: 基础模型实例
            base_model_name_or_path: 基础模型路径
            use_eagle3: 是否使用 Eagle3 架构
            total_token: 每次 draft 生成的总 token 数
            depth: draft 树的深度
            top_k: 每层选择的 top-k 数量
            threshold: 接受阈值
            ea_layer_state_dict: Eagle 层的预训练权重
            
        Returns:
            初始化好的 EagleLayer 实例
        """
        # 1. 加载配置
        from .configs import EConfig
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        bias = con.get("bias", True)

        # 2. 创建 Eagle Layer
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

        # 3. 加载权重
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        eagle_layer.load_state_dict(ea_layer_state_dict, strict=False)
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
    # 初始化和状态管理 (Initialization & State Management)
    # =========================================================================

    def reset_state(self):
        # 状态变量
        self.stable_kv = None
        self.cache_padding_mask = None
        self.full_position_ids = None
        self.tree_mask = None
        
        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)

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
        """
        构建解码器注意力掩码
        
        处理三种情况：
        1. 基本因果掩码
        2. Tree mask (用于 draft tree 内部)
        3. Cache padding mask (用于并行解码的批次对齐)
        """
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
            
            # 扩展 mask 以适应新生成的 tokens
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
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Eagle Layer 前向传播
        
        Args:
            hidden_states: 基础模型的 hidden states [batch, seq, hidden*3] 或 [batch, seq, hidden]
            input_ids: 输入 token IDs [batch, seq]
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_values: KV Cache
            use_cache: 是否使用 cache
            
        Returns:
            hidden_states: 输出 hidden states
            next_decoder_cache: 更新后的 KV Cache (如果 use_cache=True)
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

        # 对齐 hidden_states 和 embeddings 维度
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # Decoder Layer
        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
        )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = (layer_outputs[1],)
            return hidden_states, next_decoder_cache

        return hidden_states

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
        1. Root Expansion: 生成 top-k 个根候选
        2. Tree Growth: 递归扩展树
        3. Post Process: 构建最终树结构
        
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
        self.tree_mask = None  # 重置 tree mask

        scores_list = []
        parents_list = []
        tokens_list = []

        # -----------------------------------------------------------------
        # Phase 1: Root Expansion (根节点扩展)
        # -----------------------------------------------------------------
        scores, parents, next_token, next_input_ids, last_hidden = self._expand_root(
            hidden_states, input_ids, prefix_len=prefix_len, active_branch=active_branch
        )
        scores_list.append(scores)
        parents_list.append(parents)
        tokens_list.append(next_token)

        # -----------------------------------------------------------------
        # Phase 2: Tree Growth (树生长)
        # -----------------------------------------------------------------
        loop_scores, loop_parents, loop_tokens, _ = self._grow_tree(
            last_hidden, next_input_ids, scores, bsz, self.top_k, self.depth, len_posi
        )
        scores_list.extend(loop_scores)
        parents_list.extend(loop_parents)
        tokens_list.extend(loop_tokens)

        # -----------------------------------------------------------------
        # Phase 3: Post Process (后处理)
        # -----------------------------------------------------------------
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
        
        Args:
            hidden_states: 输入 hidden states
            input_ids: 输入 token IDs
            position_ids: 位置编码 (可选)
            prefix_len: 前缀长度
            active_branch: 活跃分支列表
            
        Returns:
            scores: 候选分数 [batch, top_k]
            parents: 父节点索引 [batch]
            next_token: 下一个 token [batch, top_k]
            next_input_ids: 下一个输入 IDs [batch, top_k]
            last_hidden: 最后一个 hidden state [batch, hidden]
        """
        actual_hidden = hidden_states

        # 根据当前状态确定输入
        if self.stable_kv is not None:
            # 非 Prefill 阶段
            kv_len = self.stable_kv[0][0].shape[2]
            
            if input_ids.shape[0] == 1:
                if active_branch is not None:
                    # 并行阶段只剩一个 branch
                    actual_input = input_ids
                else:
                    # Skeleton 解码
                    actual_input = input_ids[:, 1:]
                    actual_input = actual_input[:, kv_len:]
            elif hidden_states.shape[1] != input_ids.shape[1]:
                # 并行初始化
                actual_input = input_ids[:, 1:]
            else:
                # 并行解码
                actual_input = input_ids
        else:
            # Skeleton Prefill
            actual_input = input_ids[:, 1:]

        # 处理位置编码
        if self.full_position_ids is not None:
            position_start = self.stable_kv[0][0].shape[2]
            step = actual_input.shape[1]
            position_ids = self.full_position_ids[:, position_start:position_start + step]

        # Eagle Layer Forward
        out_hidden, past_key_values = self(
            actual_hidden,
            input_ids=actual_input,
            position_ids=position_ids,
            past_key_values=self.stable_kv,
            use_cache=True,
        )
        
        # 更新 stable KV (只保存已接受的 tokens)
        self.stable_kv = past_key_values
        
        # 生成 top-k 候选
        last_hidden = out_hidden[:, -1]
        last_headout = self.lm_head(self.norm(last_hidden))
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # 处理词表映射
        if self.config.vocab_size == self.config.draft_vocab_size:
            next_token = topk_index
            next_input_ids = topk_index
        else:
            mapped_tokens = topk_index + self.d2t[topk_index]
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
        1. 对当前 top-k 节点生成 top-k 个子节点 (共 k*k 个)
        2. 从 k*k 个候选中选择全局 top-k
        3. 更新 tree mask
        
        Args:
            last_hidden: 根节点的 hidden state
            input_ids: 当前输入 token IDs
            scores: 当前分数
            bsz: batch size
            top_k: 每层保留数量
            depth: 树深度
            len_posi: 位置起点
            
        Returns:
            loop_scores: 各层分数列表
            loop_parents: 各层父节点列表
            loop_tokens: 各层 token 列表
            tree_mask: 最终 tree mask
        """
        # 初始化
        input_hidden = last_hidden[:, None, :].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init.repeat(bsz, 1, 1, 1)
        local_range = torch.arange(top_k, device=self.embed_tokens.weight.device)
        past_key_values = self.stable_kv
        
        loop_scores = []
        loop_parents = []
        loop_tokens = []
        len_posi -= 1  # 对齐调整

        for i in range(depth):
            self.tree_mask = tree_mask
            
            # 计算位置编码
            if self.full_position_ids is not None:
                root_pos = self.full_position_ids[:, -1]
                current_pos = root_pos + i + 1
                position_ids = current_pos.unsqueeze(1).expand(-1, top_k)
            else:
                position_ids = len_posi + self.position_ids
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
                len_posi += 1

            # Eagle Layer Forward
            out_hidden, past_key_values = self(
                input_hidden, input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids, use_cache=True
            )

            # 计算父节点索引
            # bias 用于计算节点在整棵树中的全局索引
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (local_range + bias)
            loop_parents.append(parents.unsqueeze(0).repeat(bsz, 1))

            # 预测下一层
            last_headout = self.lm_head(self.norm(out_hidden))
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

            # 保存 tokens
            if self.config.vocab_size == self.config.draft_vocab_size:
                loop_tokens.append(local_topk_index)
                flat_source = local_topk_index.view(bsz, -1)
            else:
                mapped = local_topk_index + self.d2t[local_topk_index]
                loop_tokens.append(mapped)
                flat_source = mapped.view(bsz, -1)

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
        """
        后处理：构建最终的 draft tree 结构
        
        Args:
            bsz: batch size
            scores_list: 各层分数
            tokens_list: 各层 tokens
            parents_list: 各层父节点
            sample_token: 采样的 root token
            total_tokens: 总节点数
            top_k: top-k 值
            
        Returns:
            draft_tokens: 候选 tokens [batch, total_nodes]
            retrieve_indices: 路径索引 [batch, num_leaves, depth]
            tree_mask: 注意力掩码 [batch, 1, nodes, nodes]
            tree_position_ids: 位置编码 [batch, nodes]
        """
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

            # 构建 retrieve indices (叶节点到根的路径)
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

            # 排序 retrieve indices
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

        # 对齐 retrieve indices (不同批次可能有不同的叶节点数和深度)
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
