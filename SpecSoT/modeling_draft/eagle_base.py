# coding=utf-8
"""
Eagle Base - Eagle Layer 基类

定义 Eagle Layer 的公共接口和共享实现：
- forward(): 前向传播（抽象方法）
- get_head_output(): 获取 lm_head 输出（抽象方法）
- reset_state(): 重置状态
- init_kv_cache(): 初始化 KV Cache
- set_draft_past_key_values_sync_callback(): 设置同步回调
- _prepare_decoder_attention_mask(): 构建注意力掩码

Eagle2 与 Eagle3 的主要区别：
1. Decoder 层数：Eagle2 多层，Eagle3 单层
2. Attention 输入维度：Eagle2 用 hidden_size，Eagle3 用 hidden_size * 2
3. fc 层输入维度：Eagle2 用 hidden_size * 2，Eagle3 用 hidden_size * 3
4. lm_head：Eagle2 用外部传入，Eagle3 用内置

KV Cache 管理：
- 使用预分配内存的 KVCache 类管理（与 Base Model 一致）
- draft_past_key_values: List[List[KVCache]] 结构
- draft_past_key_values_data: 底层 tensor 数据
- draft_current_length_data: 当前长度追踪
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import os
import json
import math
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .configs import EConfig


# =============================================================================
# Attention Mask 工具函数
# =============================================================================

def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """构建因果掩码（用于自回归注意力）"""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([
            torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
            mask
        ], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None
) -> torch.Tensor:
    """扩展 attention_mask 从 [bsz, seq_len] 到 [bsz, 1, tgt_seq_len, src_seq_len]"""
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
    """重复 KV heads 以匹配 Query heads 数量（用于 GQA）"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LlamaRotaryEmbedding(nn.Module):
    """Llama 风格的旋转位置编码"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device=None
    ):
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
# MLP Layer
# =============================================================================

class LlamaMLP(nn.Module):
    """Eagle Layer 的 MLP 模块"""

    def __init__(self, config: EConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# EagleBase 基类
# =============================================================================

class EagleBase(nn.Module, ABC):
    """
    Eagle Layer 基类

    定义 Eagle2 和 Eagle3 的公共接口和共享实现。
    子类需要实现 forward() 和 get_head_output() 方法。
    
    KV Cache 管理：
    - 使用 KVCache 类管理（与 Base Model 一致）
    - draft_past_key_values: List[List[KVCache]] - 每层的 [key_cache, value_cache]
    - draft_past_key_values_data: torch.Tensor - 底层数据存储
    - draft_current_length_data: torch.Tensor - 当前长度追踪
    - kv_cache_initialized: bool - 是否已初始化
    """

    def __init__(self):
        super().__init__()
        # 状态变量（子类初始化具体值）
        self.draft_past_key_values = None
        self.draft_past_key_values_data = None
        self.draft_current_length_data = None
        self.kv_cache_initialized = False
        
        self.cache_padding_mask = None
        self.full_position_ids = None
        self.tree_mask = None
        self.tree_mask_init = None
        self.position_ids = None
        self._draft_kv_sync_callback = None

        # 子类需要设置的属性
        self.config = None
        self.top_k = None
        self.total_tokens = None
        self.depth = None
        self.hidden_size = None
        self.embed_tokens = None
        
        # KV Cache 配置（子类设置）
        self._num_layers = 1  # 默认单层，Eagle2 会覆盖
        self._num_key_value_heads = None
        self._head_dim = None

    
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
    # 抽象方法（子类必须实现）
    # =========================================================================

    @abstractmethod
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
            hidden_states: 基础模型的 hidden states
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            position_ids: 位置编码
            past_key_values: KV Cache
            use_cache: 是否使用 cache

        Returns:
            hidden_states: 输出 hidden states
            next_decoder_cache: 更新后的 KV Cache
        """
        pass

    @abstractmethod
    def get_head_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取 lm_head 输出

        Args:
            hidden_states: 输入 hidden states

        Returns:
            logits: lm_head 输出
        """
        pass

    # =========================================================================
    # 公共方法
    # =========================================================================

    def reset_state(self):
        """重置状态（不释放 KV Cache 内存，只重置长度）"""
        # 重置 KV Cache 长度为 0（保留预分配的内存）
        if self.kv_cache_initialized and self.draft_past_key_values is not None:
            for layer_kv in self.draft_past_key_values:
                for kv_cache in layer_kv:
                    kv_cache.current_length.fill_(0)
        else:
            self.draft_past_key_values = None
            
        self.cache_padding_mask = None
        self.full_position_ids = None
        self.tree_mask = None

        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)
        self._draft_kv_sync_callback = None

    def init_kv_cache(self, max_length: int, batch_size: int = 1):
        """
        初始化 KV Cache（预分配内存）
        
        Args:
            max_length: 最大序列长度
                       注意：应该是 prefill_len + max_draft_steps * (top_k * depth)
                       但实际上只需要保留 expand_root 的 cache，所以可以更小
            batch_size: 批次大小
        """
        # 延迟导入避免循环依赖
        from ..kv_cache import KVCache
        
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        
        # 获取 KV cache 配置
        num_layers = self._num_layers
        num_key_value_heads = self._num_key_value_heads or self.config.num_key_value_heads
        head_dim = self._head_dim or getattr(
            self.config, 'head_dim', 
            self.config.hidden_size // self.config.num_attention_heads
        )
        
        # 分配底层数据存储
        # shape: [num_layers * 2, batch_size, num_kv_heads, max_length, head_dim]
        self.draft_past_key_values_data = torch.zeros(
            num_layers * 2,
            batch_size,
            num_key_value_heads,
            max_length,
            head_dim,
            device=device,
            dtype=dtype,
        )
        
        # 长度追踪 tensor
        self.draft_current_length_data = torch.zeros(
            num_layers * 2, dtype=torch.long, device="cpu"
        )
        
        # 为每层创建 KVCache 对象
        self.draft_past_key_values = []
        for i in range(num_layers):
            self.draft_past_key_values.append([
                KVCache(self.draft_past_key_values_data[2 * i], self.draft_current_length_data[2 * i]),
                KVCache(self.draft_past_key_values_data[2 * i + 1], self.draft_current_length_data[2 * i + 1])
            ])
        
        self.kv_cache_initialized = True
        
    def get_kv_cache_length(self) -> int:
        """获取当前 KV Cache 的有效长度"""
        if not self.kv_cache_initialized or self.draft_past_key_values is None:
            return 0
        # 所有层的长度应该一致，取第一层的
        return self.draft_past_key_values[0][0].current_length.item()
    
    def set_kv_cache_length(self, length: int):
        """
        设置 KV Cache 的有效长度
        
        用于在 draft tree 生成完成后，丢弃 tree_grow 阶段的临时数据，
        只保留 expand_root 阶段的 cache。
        
        Args:
            length: 目标长度
        """
        if not self.kv_cache_initialized or self.draft_past_key_values is None:
            return
        for layer_kv in self.draft_past_key_values:
            for kv_cache in layer_kv:
                kv_cache.current_length.fill_(length)

    def set_draft_kv_sync_callback(self, callback):
        """
        设置 draft_past_key_values 同步回调函数

        在分布式模式下，当 draft_past_key_values 更新后立即调用此回调以同步到其他 rank

        Args:
            callback: 回调函数，接受 draft_past_key_values 作为参数
        """
        self._draft_kv_sync_callback = callback

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

            if current_src_len > mask_len:
                diff = current_src_len - mask_len
                new_valid = torch.zeros(
                    (bsz, diff), dtype=torch.bool, device=padding_bool.device
                )
                padding_bool = torch.cat([padding_bool, new_valid], dim=1)

            padding_bool = padding_bool[:, None, None, :]
            combined_attention_mask = combined_attention_mask.masked_fill(
                padding_bool, torch.finfo(torch.float32).min
            )

        return combined_attention_mask
