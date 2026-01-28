# coding=utf-8
"""
Eagle Layer2: EAGLE2 适配版本

本文件适配 EAGLE2 架构，用于支持 Vicuna 等使用 EAGLE2 的模型。

EAGLE2 与 EAGLE3 的主要区别：
1. Decoder 层数：EAGLE2 使用多层 Decoder Stack，EAGLE3 使用单层
2. 注意力输入维度：EAGLE2 使用 hidden_size，EAGLE3 使用 hidden_size * 2
3. fc 层输入维度：EAGLE2 使用 hidden_size * 2，EAGLE3 使用 hidden_size * 3
4. lm_head：EAGLE2 使用外部传入的 head，EAGLE3 使用内置的 lm_head

核心方法：
- forward(): 前向传播
- get_head_output(): 获取 lm_head 输出

Draft Tree 生成逻辑已移至 Drafter 类。
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

# 复用 eagle_base 中的公共组件
from .eagle_base import (
    EagleBase,
    _make_causal_mask,
    _expand_mask,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaMLP,
)



# =============================================================================
# EAGLE2 Attention Layer
# =============================================================================

class Eagle2Attention(nn.Module):
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

class Eagle2DecoderLayer(nn.Module):
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
        self.self_attn = Eagle2Attention(config)
        self.mlp = LlamaMLP(config)
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

class Eagle2(EagleBase):
    """
    EAGLE2 Layer: 多层 Decoder Stack 版本
    
    继承自 EagleBase，复用公共方法。
    
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
            Eagle2DecoderLayer(config, index) 
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

    def get_head_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取 lm_head 输出

        EAGLE2 使用外部传入的 head（基础模型的 lm_head）

        Args:
            hidden_states: 输入 hidden states

        Returns:
            logits: lm_head 输出
        """
        if self.diff_device:
            # 使用复制的权重
            return nn.functional.linear(hidden_states, self.headweight)
        else:
            return self.head(hidden_states)
