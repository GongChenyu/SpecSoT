"""
分布式版本的Qwen3模型
支持Pipeline Parallel (PP) 和 KV Cache同步
基于modeling_qwen3_kv.py修改
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from modeling_qwen3_kv import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3Config,
    Qwen3PreTrainedModel
)
from cache_sync_manager import create_sync_manager, CacheSyncManager


class Qwen3ModelDistributed(Qwen3Model):
    """支持Pipeline Parallel的Qwen3Model"""
    
    def __init__(self, config: Qwen3Config, rank: int = 0, world_size: int = 1, sync_strategy: str = "pairwise"):
        super().__init__(config)
        
        self.rank = rank
        self.world_size = world_size
        self.sync_strategy = sync_strategy
        
        # Pipeline Parallel配置
        self.pp_start_layer = 0
        self.pp_end_layer = config.num_hidden_layers
        self.is_pipeline_mode = False  # 是否处于pipeline模式
        
        # Cache同步管理器
        if world_size > 1:
            self.cache_sync_manager = create_sync_manager(
                rank=rank,
                world_size=world_size,
                strategy=sync_strategy,
                streaming=False
            )
        else:
            self.cache_sync_manager = None
            
    def set_pipeline_range(self, start_layer: int, end_layer: int):
        """设置当前rank负责的层范围"""
        self.pp_start_layer = start_layer
        self.pp_end_layer = end_layer
        self.is_pipeline_mode = True
        
    def set_full_model_mode(self):
        """设置为全量模型模式（decode阶段使用）"""
        self.pp_start_layer = 0
        self.pp_end_layer = self.config.num_hidden_layers
        self.is_pipeline_mode = False
        
    def forward_pipeline_stage(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> dict:
        """
        Pipeline stage的forward
        只计算当前rank负责的层
        
        Args:
            input_ids: 输入token ids (只有第一个rank需要)
            hidden_states: 从上一个rank接收的hidden states (非第一个rank需要)
            其他参数同Qwen3Model.forward
            
        Returns:
            dict包含:
                - last_hidden_state: 最后一层的hidden state
                - past_key_values: 当前rank计算的层的KV cache
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        batch_size = input_ids.shape[0] if input_ids is not None else hidden_states.shape[0]
        seq_length = input_ids.shape[1] if input_ids is not None else hidden_states.shape[1]
        
        # 第一个rank: 从input_ids获取embeddings
        if self.rank == 0 and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif hidden_states is None:
            raise ValueError("非第一个rank必须提供hidden_states")
            
        # 准备attention mask和position ids
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        seq_length_with_past = seq_length + past_key_values_length
        
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)
            
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )
        
        # 创建position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 只处理当前rank负责的层
        current_kv_caches = []
        
        for layer_idx in range(self.pp_start_layer, self.pp_end_layer):
            decoder_layer = self.layers[layer_idx]
            
            # 获取该层的past_key_value
            layer_past_kv = None
            if past_key_values is not None:
                # past_key_values可能只包含部分层的cache
                local_idx = layer_idx - self.pp_start_layer
                if local_idx < len(past_key_values):
                    layer_past_kv = past_key_values[local_idx]
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                # layer_outputs格式: (hidden_states,) or (hidden_states, attn_weights,) or (hidden_states, past_kv) or (hidden_states, attn_weights, past_kv)
                # 如果output_attentions=True且use_cache=True: (hidden_states, attn_weights, past_kv)
                # 如果output_attentions=False且use_cache=True: (hidden_states, past_kv)
                kv_idx = 2 if output_attentions else 1
                current_kv_caches.append(layer_outputs[kv_idx])
        
        # 最后一个rank需要过norm
        if self.rank == self.world_size - 1:
            hidden_states = self.norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': current_kv_caches if use_cache else None
        }
        
    def sync_kv_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        同步所有设备的KV cache
        
        Args:
            kv_caches: 当前设备的KV cache列表
            
        Returns:
            合并后的完整KV cache列表
        """
        if self.cache_sync_manager is None or self.world_size == 1:
            return kv_caches
            
        return self.cache_sync_manager.sync_all_layers_sync(kv_caches)
    
    def forward_single_layer(
        self,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> dict:
        """
        单层forward计算
        
        Args:
            layer_idx: 层索引（全局索引）
            hidden_states: 输入的hidden states
            其他参数同标准forward
            
        Returns:
            dict包含:
                - hidden_states: 输出的hidden states
                - past_key_value: 该层的KV cache (key_states, value_states)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        # 获取对应的decoder layer
        decoder_layer = self.layers[layer_idx]
        
        # 执行该层的forward
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        # layer_outputs格式: (hidden_states,) or (hidden_states, attn_weights,) or (hidden_states, past_kv) or (hidden_states, attn_weights, past_kv)
        new_hidden_states = layer_outputs[0]
        
        result = {'hidden_states': new_hidden_states}
        
        if use_cache:
            kv_idx = 2 if output_attentions else 1
            result['past_key_value'] = layer_outputs[kv_idx]
        else:
            result['past_key_value'] = None
            
        return result
    
    def sync_single_layer_cache(
        self,
        layer_idx: int,
        kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同步单层的KV cache
        
        Args:
            layer_idx: 层索引
            kv_cache: 单层的(key, value) cache tuple
            
        Returns:
            同步后的(key, value) cache tuple
        """
        if self.cache_sync_manager is None or self.world_size == 1:
            return kv_cache
        
        # 使用sync_all_layers_sync，但只传入单层cache
        synced_caches = self.cache_sync_manager.sync_all_layers_sync([kv_cache])
        return synced_caches[0]


class Qwen3ForCausalLMDistributed(Qwen3ForCausalLM):
    """支持分布式推理的Qwen3ForCausalLM"""
    
    def __init__(self, config, rank: int = 0, world_size: int = 1, sync_strategy: str = "pairwise"):
        # 调用父类的__init__，但需要替换model
        Qwen3PreTrainedModel.__init__(self, config)
        
        # 使用分布式版本的model
        self.model = Qwen3ModelDistributed(config, rank=rank, world_size=world_size, sync_strategy=sync_strategy)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.rank = rank
        self.world_size = world_size
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def set_pipeline_range(self, start_layer: int, end_layer: int):
        """设置Pipeline Parallel的层范围"""
        self.model.set_pipeline_range(start_layer, end_layer)
        
    def set_full_model_mode(self):
        """设置为全量模型模式"""
        self.model.set_full_model_mode()
        
    def forward_pipeline_stage(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> dict:
        """
        Pipeline stage的forward
        
        Returns:
            dict包含:
                - last_hidden_state: 最后一层的hidden state
                - past_key_values: KV cache
                - logits: 如果是最后一个rank，还包含logits
        """
        outputs = self.model.forward_pipeline_stage(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        result = {
            'last_hidden_state': outputs['last_hidden_state'],
            'past_key_values': outputs['past_key_values']
        }
        
        # 只有最后一个rank计算logits
        if self.rank == self.world_size - 1:
            logits = self.lm_head(outputs['last_hidden_state'])
            result['logits'] = logits
            
        return result
        
    def sync_kv_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """同步KV cache"""
        return self.model.sync_kv_caches(kv_caches)
    
    def forward_single_layer(
        self,
        layer_idx: int,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> dict:
        """单层forward，直接调用model的方法"""
        return self.model.forward_single_layer(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs
        )
    
    def sync_single_layer_cache(
        self,
        layer_idx: int,
        kv_cache: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """同步单层KV cache"""
        return self.model.sync_single_layer_cache(layer_idx, kv_cache)

