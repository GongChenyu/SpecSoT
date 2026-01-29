# coding=utf-8
"""
Eagle Layer3: EAGLE3 实现

EAGLE3 架构特点：
1. 单层 Decoder
2. 注意力输入维度：hidden_size * 2
3. fc 层输入维度：hidden_size * 3
4. 内置 lm_head
5. 支持词表映射 (d2t)
"""

import math
import os
import json
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .configs import EConfig
from .eagle_base import EagleBase
from .eagle_base import (
    _make_causal_mask,
    _expand_mask,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaMLP,
)


# =============================================================================
# Attention Layer
# =============================================================================

class Eagle3Attention(nn.Module):
    """Eagle3 的注意力模块（输入维度是 hidden_size * 2）"""

    def __init__(self, config: EConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # 投影层（输入维度是 hidden_size * 2）
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

        # 检测 past_key_value 的类型（KVCache 类 or tuple）
        use_kv_cache_class = False
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # 检查是否是 KVCache 类（有 shape 属性返回包含 current_length 的 tuple）
            if hasattr(past_key_value[0], 'shape') and hasattr(past_key_value[0], 'cat'):
                use_kv_cache_class = True
                kv_seq_len += past_key_value[0].shape[2]  # KVCache.shape[2] 是 current_length
            else:
                kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if use_kv_cache_class:
                # 使用 KVCache 类的 cat 方法
                key_states = past_key_value[0].cat(key_states, dim=2)
                value_states = past_key_value[1].cat(value_states, dim=2)
                # 注意: cat 方法会更新 current_length 并返回完整的 data[:current_length]
            else:
                # 原始方式：torch.cat
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # 返回的 past_key_value 保持与输入相同的格式
        # 如果使用 KVCache 类，不需要返回新的 tuple，因为 cat 已经原地更新了
        if use_kv_cache_class:
            # 传递 KVCache 引用，后续调用会继续使用同一个 cache
            past_key_value = past_key_value if use_cache else None
        else:
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
# Decoder Layer
# =============================================================================

class Eagle3DecoderLayer(nn.Module):
    """Eagle Layer 的 Decoder 层"""

    def __init__(self, config: EConfig, last: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Eagle3Attention(config)
        self.mlp = LlamaMLP(config)
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

class Eagle3(EagleBase):
    """
    Eagle Layer: 轻量级草稿模型
    
    负责快速生成候选 token 树，供基础模型验证。
    继承自 EagleBase，复用公共方法。
    
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
        draft_past_key_values: 稳定的 KV Cache (已接受的 tokens)
        
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
        self.midlayer = Eagle3DecoderLayer(config)
        
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
        
        # KV Cache 配置（Eagle3 是单层）
        self._num_layers = 1
        self._num_key_value_heads = config.num_key_value_heads
        self._head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

        self.reset_state()

    @classmethod
    def from_pretrained(
        cls,
        ea_model_path: str,
        base_model: nn.Module,
        base_model_name_or_path: str,
        use_eagle3: bool = True,
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
    ):
        """
        从预训练模型加载 Eagle Layer
        
        加载流程：
        1. 加载配置: 从 config.json 加载 Eagle 配置
        2. 加载权重: 从 model.safetensors 或 pytorch_model.bin 加载权重
        3. 创建实例: 初始化 Eagle Layer
        4. 设备管理: 处理跨设备权重
        5. 词表映射: 处理 draft vocab 到 target vocab 的映射 (Eagle3)
        6. 加载权重: 应用预训练权重
        
        Args:
            ea_model_path: Eagle 模型目录路径
            base_model: 基础模型实例 (用于获取设备和 dtype)
            base_model_name_or_path: 基础模型路径 (用于加载 embedding)
            use_eagle3: 是否使用 Eagle3 架构
            total_token: 每次 draft 生成的总 token 数
            depth: draft 树的深度
            top_k: 每层选择的 top-k 数量
            threshold: 接受阈值
            
        Returns:
            初始化好的 EagleLayer 实例
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
        # 3. 创建 Eagle Layer 实例
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
        # 4. 设备管理：处理跨设备权重
        # =====================================================================
        if device != base_model.lm_head.weight.device:
            eagle_layer.diff_device = True
            eagle_layer.headweight = base_model.lm_head.weight.clone().to(device)
        else:
            eagle_layer.diff_device = False

        # =====================================================================
        # 5. Eagle3 词表映射：如果 draft_vocab == target_vocab，删除映射 buffer
        # =====================================================================
        if use_eagle3 and config.vocab_size == config.draft_vocab_size:
            if hasattr(eagle_layer, 'd2t'):
                del eagle_layer.d2t
            if hasattr(eagle_layer, 't2d'):
                del eagle_layer.t2d

        # =====================================================================
        # 6. 加载权重并移动到正确设备
        # =====================================================================
        eagle_layer.load_state_dict(ea_state_dict, strict=False)
        eagle_layer.to(base_model.dtype).to(device)
        
        eagle_layer.reset_state()
        
        return eagle_layer

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
    # 获取 lm_head 输出
    # =========================================================================

    def get_head_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取 lm_head 输出

        EAGLE3 使用内置的 lm_head，需要先经过 norm 层

        Args:
            hidden_states: 输入 hidden states

        Returns:
            logits: lm_head 输出
        """
        return self.lm_head(self.norm(hidden_states))
