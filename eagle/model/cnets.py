# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor




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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # 修改
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
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
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)


        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)


        # cache_hidden.append(hidden_states)

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

        # Fully Connected
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


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor



def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()
        self.config=config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head=nn.Linear(config.hidden_size,config.draft_vocab_size,bias=False)
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json
            try:
                index_json_path = os.path.join(path, "model.safetensors.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "model.safetensors.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                with safe_open(local_emb_path,
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                index_json_path = os.path.join(path, "pytorch_model.bin.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "pytorch_model.bin.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                weights = torch.load(local_emb_path)
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t=torch.zeros((config.draft_vocab_size),dtype=torch.long)
        t2d=torch.zeros((config.vocab_size),dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        
        # =============== 添加内容 ===============
        if hasattr(self, "cache_padding_mask") and self.cache_padding_mask is not None:
            # 我们需要生成一个 bool mask，其中 True 表示"是Padding/需要被屏蔽"
            padding_bool = (self.cache_padding_mask == 0)

            # 随着树的扩展，维度会变化，新生成的token都是有效的，因此简单拼接即可适应维度增长
            current_src_len = combined_attention_mask.shape[-1] # e.g., 380
            mask_len = padding_bool.shape[1]
            bsz = padding_bool.shape[0]
            if current_src_len > mask_len:
                # 情况：Tree Growth 阶段，序列变长了 (380 > 370)，新生成的 Tree Nodes肯定是效的 (False = 不屏蔽)
                diff = current_src_len - mask_len
                new_valid_part = torch.zeros((bsz, diff), dtype=torch.bool, device=padding_bool.device)
                padding_bool = torch.cat([padding_bool, new_valid_part], dim=1)
            
            # 维度对齐: [bsz, total_seq_len] -> [bsz, 1, 1, total_seq_len]
            # 这样可以广播到 [bsz, heads, tgt_len, src_len]
            padding_bool = padding_bool[:, None, None, :]

            # 将 Padding 位置设为 -inf
            min_value = torch.finfo(torch.float32).min
            combined_attention_mask = combined_attention_mask.masked_fill(padding_bool, min_value)


        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        # hidden_states = self.fc(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]


        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None


    # ============== 添加内容 ================
    def _expand_root(self, hidden_states, input_ids, position_ids=None, prefix_len=-1, active_branch=None):
        """
        Phase 1: Root Expansion
        """
        # 无论何时都要切掉第一个token对齐
        actual_hidden_states = hidden_states

        # 1. 判断当前状态   我觉的这里的分支判断应该往前放，也就是说在输入的时候，就应该输入正确的token和hidden states
        if hasattr(self, "stable_kv") and self.stable_kv is not None:   # 如果是非prefill阶段
            kv_len = self.stable_kv[0][0].shape[2]

            if input_ids.shape[0] == 1: 
                if active_branch is not None:  # para 阶段只剩一个branch
                    actual_input_ids = input_ids 
                else:  # Skeleton Decoding, bsz=1
                    actual_input_ids = input_ids[:, 1:]
                    actual_input_ids = actual_input_ids[:, kv_len:]  # 需要通过cache与hidden states对齐
            elif hidden_states.shape[1] != input_ids.shape[1]:  # Parallel init
                actual_input_ids = input_ids[:, 1:]
            else:  # parallel decoding
                # Parallel 什么都不切，输入的维度是对的
                actual_input_ids = input_ids
        else:  # Skeleton Prefill (First Run)
            # 只需要 Input 切掉第一个，Hidden 保留全量, 对齐
            actual_input_ids = input_ids[:, 1:]

        if hasattr(self, "full_position_ids") and self.full_position_ids is not None:
            position_start = self.stable_kv[0][0].shape[2]
            step = actual_input_ids.shape[1]
            position_end = position_start + step
            position_ids = self.full_position_ids[:, position_start:position_end]
            
        # 3. Forward
        out_hidden, past_key_values = self(
            actual_hidden_states, 
            input_ids=actual_input_ids,
            position_ids=position_ids,   # skeleton阶段可以根据kv长度自己生成，不需要手动传入
            past_key_values=self.stable_kv, 
            use_cache=True
        )
        #  这里更新了kv，但是cache padding mask和position是在前边 parallel init更新的
        self.stable_kv = past_key_values   # 只有这个过程更新stable kv即可，因为此时计算的kv是已经接受的token的，后续tree grow都是不一定接收的，这里传递临时变量即可
        last_hidden = out_hidden[:, -1]  # 第一个轮次root的hidden states需要保存
        #  root生成的logits中，采样前Topk个token 
        last_headout = self.lm_head(self.norm(last_hidden))
        last_p = self.logsoftmax(last_headout)   # 获取采样的概率，作为分数

        top = torch.topk(last_p, self.top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        scores = topk_p
        parents = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        
        if self.config.vocab_size == self.config.draft_vocab_size:
            next_token = topk_index
            next_input_ids = topk_index
        else:
            mapped_tokens = topk_index + self.d2t[topk_index]  # 这里表示词表映射，
            next_token = mapped_tokens
            next_input_ids = mapped_tokens

        return scores, parents, next_token, next_input_ids, last_hidden  # next_input_ids和 next token是一样的东西，删掉一个


    def _grow_tree(self, last_hidden, input_ids, scores, bsz, top_k, depth, len_posi):

        # Save Root Tokens (From Phase 1)
        # 注意：这里需要传入 Phase 1 生成的 tokens，为了保持逻辑连续，我们可以在这里重新 append
        # 或者在主函数里管理列表。为了函数纯粹性，建议返回 incremental lists。
        # 这里为了简化，我们假设主函数已经初始化了 list。
        
        # 我们重新构造这个函数，让它负责整个 Loop 逻辑
        
        input_hidden = last_hidden[:, None, :].repeat(1, top_k, 1)   # 因为一个节点生成十个，所以拼接父节点的 hidden state 要 repeat
        tree_mask = self.tree_mask_init.repeat(bsz, 1, 1, 1)   # 这里是一个对角矩阵，没有root节点  ############################ 这里会不会有问题？最终的tree_mask是否包括root节点
        local_range_index = torch.arange(top_k, device=self.embed_tokens.weight.device)  # 用于保存当前层的索引
        past_key_values = self.stable_kv # 这里是eagle layer更新好的kv，后续的tree grow都是不一定接收的，因此通过临时变量使用即可
        
        # Return containers
        loop_scores = []
        loop_parents = []
        loop_tokens = []
        len_posi -= 1  # 输入topk gen的input已经包括了新生成的一个token，为了对齐，实际上计算的token数要-1，也将就是说，两个模型的输入维度都是len_posi-1
        for i in range(depth):
            self.tree_mask = tree_mask
            # 单branch情况下的 position id 应该是跟kv 的长度是一样的，就是last kv index +1
            if hasattr(self, "full_position_ids") and self.full_position_ids is not None:  # parallel 阶段
                root_position_ids = self.full_position_ids[:, -1]
                current_pos = root_position_ids + i + 1
                position_ids = current_pos.unsqueeze(1).expand(-1, top_k)
            else:  # skeleton阶段
                position_ids = len_posi + self.position_ids  # 此处len_pos是当前stablekv的长度
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
                len_posi += 1
                 
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                              position_ids=position_ids, use_cache=True)
            # Parents
            bias1 = top_k if i > 0 else 0  # 全量的第一层的个数
            bias2 = max(0, i - 1)  # 如果处于第二层及以上，就要考虑前面层的全量的token数量
            bias = 1 + top_k ** 2 * bias2 + bias1  # 根节点，加全量的层token数量，再加全量的第一层的token数量。local_range_index是当前层的索引
            parents = (local_range_index + bias)   #  这里的逻辑其实保存的是当前节点在整棵树中的索引值，而不是父节点的索引
            loop_parents.append(parents.unsqueeze(0).repeat(bsz, 1))  # 在这之前，parents中是空的，没有存根节点

            # Prediction
            last_headout = self.lm_head(self.norm(out_hidden))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)  # 在单个hidden states上，针对logits取前十个
            local_topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, :, None]
            
            # Global Top-K
            topk_cs = torch.topk(cu_scores.view(bsz, -1), top_k, dim=-1)   # 获取该轮次100个token的前10个
            selected_cs_index, selected_scores = topk_cs.indices, topk_cs.values
            scores = selected_scores

            # Backtracking
            out_ids = selected_cs_index // top_k
            out_ids_expanded = out_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            input_hidden = torch.gather(out_hidden, 1, out_ids_expanded)  # 获取选取的前十个token的父节点，并获取相应父节点对应的hidden states

            # Tokens
            flat_topk_index = local_topk_index.view(bsz, -1)
            selected_tokens = torch.gather(flat_topk_index, 1, selected_cs_index)

            if self.config.vocab_size == self.config.draft_vocab_size:
                loop_tokens.append(local_topk_index) # Save full K*K
                flat_source = local_topk_index.view(bsz, -1)
            else:
                mapped_tokens = local_topk_index + self.d2t[local_topk_index]
                loop_tokens.append(mapped_tokens) # Save full K*K
                flat_source = mapped_tokens.view(bsz, -1)
            
            input_ids = torch.gather(flat_source, 1, selected_cs_index) # Next Input

            loop_scores.append(cu_scores)

            # Mask Update
            new_masks = []
            for b in range(bsz):
                current_b_mask = tree_mask[b:b+1]
                selected_cols = current_b_mask[:, :, :, out_ids[b]]
                init_mask = self.tree_mask_init
                new_mask_b = torch.cat((selected_cols, init_mask), dim=3)
                new_masks.append(new_mask_b)
            tree_mask = torch.cat(new_masks, dim=0)
            
        return loop_scores, loop_parents, loop_tokens, tree_mask


    def _post_process_tree(self, bsz, scores_list, ss_token, parents_list, sample_token, total_tokens, top_k):
        draft_tokens_list = []
        retrieve_indices_list = []
        tree_mask_final_list = []
        tree_position_ids_list = []

        for b in range(bsz):
            # Flatten Logic
            b_scores = [scores_list[0][b].view(-1)] + [s[b].view(-1) for s in scores_list[1:]]
            flat_scores = torch.cat(b_scores, dim=0)  # 这里的所有数据中都不含有root   # 710
            
            b_tokens = [ss_token[0][b].view(-1)] + [t[b].view(-1) for t in ss_token[1:]]
            flat_tokens = torch.cat(b_tokens, dim=0)   # 710
            
            b_parents = [parents_list[0][b].view(-1)] + [p[b].view(-1) for p in parents_list[1:]]
            flat_parents = torch.cat(b_parents, dim=0)  # 80,这里的数量对吗

            # Selection
            top_scores = torch.topk(flat_scores, total_tokens, dim=-1) # total_tokens = 59
            top_scores_index = torch.sort(top_scores.indices).values

            draft_tokens_b = flat_tokens[top_scores_index]
            draft_tokens_b = torch.cat((sample_token[b].unsqueeze(0), draft_tokens_b), dim=0)   # 这里draft tokens中已经含有root节点，总节点数60
            
            # Graph & Mask
            raw_parents = flat_parents[top_scores_index // top_k].long()
            mask_index = torch.searchsorted(top_scores_index, raw_parents - 1, right=False)  
            
            # Safety Checks
            # mask_index[mask_index == total_tokens] = total_tokens - 1 
            # mask_index[top_scores_index[mask_index] != (raw_parents - 1)] = -1
            mask_index[raw_parents == 0] = -1
            mask_index = mask_index + 1
            mask_index_list = mask_index.tolist()
            
            tree_mask_b = torch.eye(total_tokens + 1).bool()
            tree_mask_b[:, 0] = True
            for i in range(total_tokens):
                tree_mask_b[i + 1].add_(tree_mask_b[mask_index_list[i]])
                
            tree_position_ids_b = torch.sum(tree_mask_b, dim=1) - 1
            
            # Retrieve Indices
            max_depth = torch.max(tree_position_ids_b) + 1
            noleaf_index = torch.unique(mask_index).tolist()
            leaf_num = total_tokens - (len(noleaf_index) - 1)
            retrieve_indices_b = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
            retrieve_indices_b = retrieve_indices_b.tolist()
            
            rid = 0
            position_ids_list = tree_position_ids_b.tolist()
            for i in range(total_tokens + 1):
                if i not in noleaf_index:
                    cid = i
                    depth_i = position_ids_list[i]
                    for j in reversed(range(depth_i + 1)):
                        retrieve_indices_b[rid][j] = cid
                        cid = mask_index_list[cid - 1]
                    rid += 1
            
            # Sort
            def custom_sort(lst):
                return [lst[i] if lst[i] >= 0 else total_tokens + 5 for i in range(len(lst))]
            retrieve_indices_b = sorted(retrieve_indices_b, key=custom_sort)

            draft_tokens_list.append(draft_tokens_b)
            retrieve_indices_list.append(torch.tensor(retrieve_indices_b, dtype=torch.long))
            tree_mask_final_list.append(tree_mask_b.float())
            tree_position_ids_list.append(tree_position_ids_b)

        # Padding & Stack
        draft_tokens = torch.stack(draft_tokens_list, dim=0)
        tree_mask = torch.stack(tree_mask_final_list, dim=0)[:, None, :, :]
        tree_position_ids = torch.stack(tree_position_ids_list, dim=0).to(sample_token.device)
        
        max_depth = max([ri.shape[1] for ri in retrieve_indices_list])
        max_leaves = max([ri.shape[0] for ri in retrieve_indices_list])
        
        padded_retrieve = []
        for ri in retrieve_indices_list:
            curr_l, curr_d = ri.shape
            pad_d = max_depth - curr_d
            if pad_d > 0:
                ri = torch.cat((ri, torch.full((curr_l, pad_d), -1, dtype=torch.long, device=ri.device)), dim=1)
            pad_l = max_leaves - curr_l
            if pad_l > 0:
                ri = torch.cat((ri, torch.full((pad_l, max_depth), -1, dtype=torch.long, device=ri.device)), dim=0)
            padded_retrieve.append(ri)
            
        retrieve_indices = torch.stack(padded_retrieve, dim=0)
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def topK_generate(self, hidden_states, input_ids, prefix_len=-1, active_branch=None):
        # 1. Prepare
        bsz = input_ids.shape[0]
        input_ids = input_ids.to(hidden_states.device)
        sample_token = input_ids[:, -1]
        len_posi = input_ids.shape[1]   #  这里貌似不太对，hidden state的维度是对的，input的维度不一定准确
        self.reset()

        scores_list = []
        parents_list = []
        ss_token = []

        # 2. Phase 1: Root Expansion
        scores, parents, next_token, next_input_ids, last_hidden = self._expand_root(hidden_states, input_ids, prefix_len=prefix_len, active_branch=active_branch)
        
        scores_list.append(scores) # 这三行是保存了第一个轮次，一个节点生成十个节点的信息
        parents_list.append(parents)
        ss_token.append(next_token)
        
        # 3. Phase 2: Tree Growth 
        # self.depth=7，所以理论上生成80个token，不包括root
        loop_scores, loop_parents, loop_tokens, _ = self._grow_tree(last_hidden, next_input_ids, scores, bsz, self.top_k, self.depth, len_posi)
        
        scores_list.extend(loop_scores) # 这三行是保存了第2个轮次到最后轮次，十个节点生成十个节点的信息
        parents_list.extend(loop_parents)
        ss_token.extend(loop_tokens)
        
        # 4. Phase 3 & 4: Post Process & Stack
        return self._post_process_tree(bsz, scores_list, ss_token, parents_list, sample_token, self.total_tokens, self.top_k)



import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
