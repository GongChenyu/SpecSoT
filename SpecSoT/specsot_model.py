# coding=utf-8
"""
SpecSoT Model (Speculative Decoding + Skeleton-of-Thought)

该模型结合了投机推理(Speculative Decoding)和思维骨架(Skeleton-of-Thought)两种技术：
- 投机推理：使用 Draft Model (Eagle Layer) 快速生成候选 token，再由 Base Model 验证
- 思维骨架：先生成回答骨架，再并行填充各分支内容

核心流程分为三个阶段：
1. Skeleton Generation (骨架生成) - 使用 SD 快速生成回答骨架
2. Skeleton Parsing (骨架解析) - 解析骨架，提取并行分支
3. Parallel Branch Decoding (并行分支解码) - 并行解码各分支内容

推理过程中的关键步骤：
- Prefill: 处理输入 prompt，初始化 KV Cache
- Decode Loop: 循环执行 Draft -> Verify -> Update
  - Draft: Eagle Layer 生成候选 token 树
  - Verify: Base Model 验证候选 token
  - Update: 更新状态，准备下一轮

分布式系统适配说明：
- prefill_* 方法：可在不同节点执行初始化
- verify_step: Base Model 前向推理，可独立部署
- draft_step: Eagle Layer 生成，可独立部署
- update_state: 状态更新，协调各组件
"""

import copy
import json
import time
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessorList
import os

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM

from .eagle_layer import EagleLayer
from .kv_cache import initialize_past_key_values
from .configs import EConfig
from .logits_processor import SemanticLogitsProcessor

# 分布式推理支持
from .distributed import DistributedConfig, DistributedPrefillManager

from .utils import (
    prepare_logits_processor,
    build_parallel_prefill_mask,
    stack_with_left_padding,
    parse_skeleton,
    prepare_parallel_branches,
    prepare_skeleton_input,
    create_skeleton_logits_processor,
    check_stop_conditions,
    check_stop_conditions_parallel,
    merge_outputs,
    evaluate_single,
    evaluate_parallel,
)


# =============================================================================
# SpecSoT Model: 主模型类
# =============================================================================

class SpecSoTModel(nn.Module):
    """
    SpecSoT: Speculative Decoding enabled Skeleton-of-Thought Model
    
    该模型包含两个核心组件：
    - base_model: 目标大模型 (如 Qwen3)，负责验证和最终生成
    - eagle_layer: 轻量级草稿模型，负责快速生成候选 token 树
    
    Attributes:
        base_model: 基础大语言模型
        eagle_layer: EAGLE 草稿层
        tokenizer: 分词器
        
        # 状态管理 (State Management)
        past_key_values: Base Model 的 KV Cache
        past_key_values_data: KV Cache 的底层数据
        current_length_data: 当前 Cache 长度
        
        # 并行解码状态 (Parallel Decoding State)
        branch_index_map: 分支索引映射，记录每个 token 属于哪个分支
        full_position_ids: 完整的位置编码
        active_branches: 当前活跃的分支列表
        
        # 输出存储 (Output Storage)
        skeleton_output: 生成的骨架
        parallel_branches_output: 各分支的输出
    """

    def __init__(
        self,
        base_model: nn.Module,
        eagle_layer: nn.Module,
        base_model_name_or_path: str,
        use_eagle3: bool = True,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """
        初始化 SpecSoT 模型
        
        Args:
            base_model: 预加载的基础模型
            eagle_layer: 预加载的 Eagle Layer
            base_model_name_or_path: 基础模型路径，用于加载 tokenizer
            use_eagle3: 是否使用 Eagle3 架构
            distributed_config: 分布式推理配置
        """
        super().__init__()
        
        # =====================================================================
        # 1. Base Model
        # =====================================================================
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.use_eagle3 = use_eagle3
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )
        
        # =====================================================================
        # 2. Eagle Layer
        # =====================================================================
        self.eagle_layer = eagle_layer
        
        # =====================================================================
        # 3. 分布式配置
        # =====================================================================
        self.distributed_config = distributed_config or DistributedConfig()
        self.distributed_prefill_manager: Optional[DistributedPrefillManager] = None
        
        if self.distributed_config.enabled:
            self._init_distributed()
        
        # =====================================================================
        # 4. 推理状态初始化
        # =====================================================================
        self.reset_state()
        self.eagle_layer.reset_state()

    # =========================================================================
    # 类方法：模型加载
    # =========================================================================
    
    @classmethod
    def _load_base_model(
        cls,
        base_model_path: str,
        **kwargs,
    ) -> nn.Module:
        """
        加载 Base Model
        
        根据模型架构类型自动选择对应的 KV Cache 版本模型类。
        
        Args:
            base_model_path: 基础模型路径
            **kwargs: 传递给 from_pretrained 的其他参数
                      (如 torch_dtype, device_map, low_cpu_mem_usage 等)
            
        Returns:
            加载好的基础模型实例
        """
        model_type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        
        model_class_map = {
            'LlamaForCausalLM': KVLlamaForCausalLM,
            'Qwen2ForCausalLM': KVQwen2ForCausalLM,
            'Qwen3ForCausalLM': KVQwen3ForCausalLM,
        }
        
        BaseModelClass = model_class_map.get(model_type, KVMixtralForCausalLM)
        base_model = BaseModelClass.from_pretrained(base_model_path, **kwargs)
        
        return base_model

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        ea_model_path: str,
        use_eagle3: bool = True,
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
        distributed_config: Optional[DistributedConfig] = None,
        **kwargs,
    ) -> "SpecSoTModel":
        """
        从预训练模型加载 SpecSoT 模型
        
        加载流程：
        1. 加载 Base Model: 调用 _load_base_model
        2. 加载 Eagle Layer: 调用 EagleLayer.from_pretrained
        3. 组装 SpecSoT 模型
        
        Args:
            base_model_path: 基础模型路径
            ea_model_path: Eagle 模型路径
            use_eagle3: 是否使用 Eagle3
            total_token: 每次 draft 的总 token 数
            depth: draft 树深度
            top_k: top-k 选择数量
            threshold: 接受阈值
            distributed_config: 分布式推理配置
            **kwargs: 传递给基础模型的其他参数
            
        Returns:
            SpecSoTModel 实例
        """
        # =====================================================================
        # 1. 加载 Base Model
        # =====================================================================
        base_model = cls._load_base_model(base_model_path, **kwargs)

        # =====================================================================
        # 2. 加载 Eagle Layer
        # =====================================================================
        eagle_layer = EagleLayer.from_pretrained(
            ea_model_path=ea_model_path,
            base_model=base_model,
            base_model_name_or_path=base_model_path,
            use_eagle3=use_eagle3,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        # =====================================================================
        # 3. 组装 SpecSoT 模型
        # =====================================================================
        return cls(
            base_model=base_model,
            eagle_layer=eagle_layer,
            base_model_name_or_path=base_model_path,
            use_eagle3=use_eagle3,
            distributed_config=distributed_config,
        )

    def reset_state(self):
        """
        初始化/重置推理状态（在每次推理开始前调用）
        
        重置内容：
        1. Base Model: tree mode
        2. Eagle Layer: KV Cache, tree mask, 并行状态
        3. Base Model KV Cache 引用 (实际分配在 generate 中)
        4. 并行解码状态: BIM, position_ids, active_branches
        5. 输出存储
        
        注意：
        - Base Model KV Cache 在 generate 方法中按需分配
        - Eagle Layer init_tree 只需在模型加载时调用一次
        """
        # Base Model tree mode
        self.base_model.model.tree_mask = None
        self.base_model.model.tree_mode = None 

        # Base Model KV Cache 引用 (实际分配在 generate 中)
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

        # 并行解码状态
        # BIM: -1=共享prefix, -2=空位置, >=0=分支索引
        self.branch_index_map = None
        self.full_position_ids = None
        self.active_branches = None

        # 输出存储
        self.skeleton_output = None
        self.parallel_branches_output = None
        self.instruction_len = None

    def _init_distributed(self):
        """初始化分布式推理组件"""
        if not self.distributed_config.enabled:
            return
        
        device = self.base_model.device
        self.distributed_prefill_manager = DistributedPrefillManager(
            config=self.distributed_config,
            model=self,
            device=str(device),
        )
        print(f"[SpecSoT] 分布式推理已启用: {self.distributed_config}")
    
    def cleanup_distributed(self):
        """清理分布式资源"""
        if self.distributed_prefill_manager is not None:
            self.distributed_prefill_manager.cleanup()
            self.distributed_prefill_manager = None
    
    def is_distributed(self) -> bool:
        """是否启用分布式推理"""
        return self.distributed_config.enabled
    
    def is_main_rank(self) -> bool:
        """是否为主rank（用于输出和统计）"""
        if not self.distributed_config.enabled:
            return True
        return self.distributed_config.is_last_rank()

    # =========================================================================
    # 信息获取 (Getters)
    # =========================================================================

    def get_tokenizer(self):
        """获取 tokenizer"""
        return self.tokenizer

    def get_model_type(self) -> str:
        """
        检测模型类型
        
        Returns:
            'qwen': Qwen 系列模型
            'llama': Llama 系列模型
            'other': 其他模型
        """
        model_name = self.base_model_name_or_path.lower()
        
        if 'qwen' in model_name:
            return 'qwen'
        elif 'llama' in model_name:
            return 'llama'
        else:
            return 'other'

    # =========================================================================
    # 基础前向传播 (Base Forward)
    # =========================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_orig: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Base Model 前向传播
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码
            past_key_values: KV Cache
            position_ids: 位置编码
            output_orig: 是否输出原始 logits
            
        Returns:
            outputs: 模型输出
            orig (可选): 原始 logits
            hidden_states: 隐藏状态
        """
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            hidden_states = outputs[0]
            
            if output_orig:
                orig = self.base_model.lm_head(hidden_states)
                return outputs, orig, hidden_states
            
            return outputs, hidden_states

    # =========================================================================
    # 主生成入口 (Main Generation Entry)
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        task_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        enable_parallel: bool = True,
        para_token_ids: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """
        主生成函数：支持普通模式和并行骨架模式
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            enable_parallel: 是否启用骨架并行模式
            para_token_ids: 特殊 token IDs (骨架解析用)
            
        Returns:
            output_ids: 生成的 token IDs
            avg_accept_len: 平均接受长度
            num_para: 并行分支数量
            avg_draft_time: 平均 draft 时间
            avg_update_time: 平均 update 时间
            avg_verify_time: 平均 verify 时间
        """
        # 公共初始化
        self.reset_state()
        self.eagle_layer.reset_state()
                
        # 准备采样 logits processor (温度、top_p、top_k)
        logits_processor = None
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature, top_p, top_k)

        if not enable_parallel:
            # 普通投机解码模式
            return self.generate_eagle(
                task_prompt, max_new_tokens, logits_processor
            )
        
        # 骨架并行模式
        return self.generate_specsot(
            task_prompt, max_new_tokens, logits_processor, para_token_ids
        )

    def generate_eagle(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """
        标准投机解码生成（不使用骨架）
        
        流程：
        1. 初始化: 准备 input_ids, KV Cache
        2. Prefill: 初始化 KV Cache 和第一轮 Draft Tree (支持分布式)
        3. Decode Loop: 循环执行 decode_step_single 直到停止
        """
        device = self.base_model.device
        input_ids = self.tokenizer([task_prompt], return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]
        # print(f"input_ids: {self.tokenizer.decode(input_ids[0])}")
        
        # =====================================================================
        # 1. 初始化 KV Cache
        # =====================================================================
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # =====================================================================
        # 2. Prefill 阶段 (支持分布式)
        # =====================================================================
        if self.is_distributed():
            # 分布式Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, first_token = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, logits_processor
                )
        else:
            # 普通Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, logits_processor)
        
        # =====================================================================
        # 3. Decode 循环
        # =====================================================================
        # 计时器
        total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
        total_draft_time = 0.0
        total_update_time = 0.0
        total_verify_time = 0.0
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_after_verify = torch.cuda.Event(enable_timing=True)
        evt_after_update = torch.cuda.Event(enable_timing=True)
        evt_after_draft = torch.cuda.Event(enable_timing=True)
        
        stop_token_id = None
        eos_token_id = self.tokenizer.eos_token_id
        
        for step in range(max_new_tokens):
            evt_start.record()
            
            # 执行单步 decode
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                accept_length,
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
                evt_after_verify=evt_after_verify,
                evt_after_update=evt_after_update,
            )
            evt_after_draft.record()
            
            # 统计
            total_accept_len += accept_length.sum()
            
            # 计时统计
            torch.cuda.synchronize()
            total_verify_time += evt_start.elapsed_time(evt_after_verify) / 1000
            total_update_time += evt_after_verify.elapsed_time(evt_after_update) / 1000
            total_draft_time += evt_after_update.elapsed_time(evt_after_draft) / 1000
            
            # 停止条件检查
            if check_stop_conditions(
                input_ids, input_len, stop_token_id, eos_token_id,
                self.current_length_data[0].item(), max_kv_len,
                tokens_per_step=self.eagle_layer.total_tokens + 1
            ):
                break
        
        # 计算平均值
        num_steps = max(step, 1)
        output_ids = input_ids[:, input_len:]
        avg_accept_len = total_accept_len.item() / num_steps
        avg_draft_time = total_draft_time / num_steps
        avg_update_time = total_update_time / num_steps
        avg_verify_time = total_verify_time / num_steps
        
        return output_ids, avg_accept_len, 0, avg_draft_time, avg_update_time, avg_verify_time

    def generate_specsot(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        para_token_ids: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """
        骨架并行生成模式
        
        三阶段流程：
        1. Skeleton Generation - 生成回答骨架（单序列解码）
        2. Skeleton Parsing - 解析骨架，提取分支
        3. Parallel Decoding - 并行解码各分支
        """
        device = self.base_model.device
        model_type = self.get_model_type()
        
        # =====================================================================
        # Stage 1: Skeleton Generation (骨架生成)
        # =====================================================================
        # 准备 skeleton 阶段的 input_ids 和 logits_processor
        input_ids, task_input_ids = prepare_skeleton_input(self.tokenizer, task_prompt, model_type, device)
        input_len = input_ids.shape[1]
        # print(f"input_ids: {self.tokenizer.decode(input_ids[0])}")
        
        skeleton_logits_processor = create_skeleton_logits_processor(
            para_token_ids, input_len, logits_processor
        )
        
        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # ---------------------------------------------------------------------
        # Stage 1.1: Prefill 阶段
        # ---------------------------------------------------------------------
        if self.is_distributed():
            # 分布式Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, first_token = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, skeleton_logits_processor
                )
        else:
            # 普通Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, skeleton_logits_processor)
        
        # ---------------------------------------------------------------------
        # Stage 1.2: Skeleton Decode 循环 (不保留统计，skeleton 阶段仅作为中间步骤)
        # ---------------------------------------------------------------------
        stop_token_id = para_token_ids['para_end_token_id']
        eos_token_id = self.tokenizer.eos_token_id
        max_steps = 150  # 骨架长度限制
        
        for step in range(max_steps):
            # 执行单步 decode
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                _,  # accept_length - skeleton 阶段不需要统计
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=skeleton_logits_processor,
            )
            
            # 停止条件检查
            if check_stop_conditions(
                input_ids, input_len, stop_token_id, eos_token_id,
                self.current_length_data[0].item(), max_kv_len,
                tokens_per_step=self.eagle_layer.total_tokens + 1
            ):
                break
        
        skeleton_ids = input_ids[:, input_len:]
        print("Generated Skeleton:", self.tokenizer.decode(skeleton_ids[0]))
        self.skeleton_output = skeleton_ids.clone()

        # =====================================================================
        # Stage 2: Skeleton Parsing (骨架解析)
        # =====================================================================
        # 2.1: 解析骨架，提取分支标题和预测长度
        branch_headers, predicted_branch_lengths = parse_skeleton(
            self.tokenizer, skeleton_ids, para_token_ids
        )
        
        if not branch_headers:
            print("Parsing failed or no branches found. Returning skeleton.")
            return skeleton_ids, 0.0, 0, 0.0, 0.0, 0.0
        
        # 2.2: 准备并行分支输入（添加上下文指令前缀）
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, branch_headers, model_type, task_prompt
        )

        num_para = len(clean_branches)
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        print(f"Detected {num_para} parallel branches with predicted lengths: {predicted_branch_lengths}")

        # =====================================================================
        # Stage 3: Parallel Decoding (并行分支解码)
        # =====================================================================
        # ---------------------------------------------------------------------
        # Stage 3.1: 前缀复用 - 复用 Skeleton KV Cache + 初始化并行状态
        # ---------------------------------------------------------------------
        input_ids, tips_indices, branch_begins, branch_lengths_actual, draft_input_ids = \
            self.reuse_prefix_for_parallel(task_input_ids, clean_branches, max_new_tokens)
        
        # ---------------------------------------------------------------------
        # Stage 3.2: Prefill 阶段 (并行)
        # ---------------------------------------------------------------------
        prefix_len = task_input_ids.shape[1]
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths_actual, draft_input_ids, logits_processor
            )
        
        # ---------------------------------------------------------------------
        # Stage 3.3: Parallel Decode 循环
        # ---------------------------------------------------------------------
        total_accept_len_parallel = torch.zeros(1, dtype=torch.long, device=device)
        total_verify_time_parallel = 0.0
        total_update_time_parallel = 0.0
        total_draft_time_parallel = 0.0
        
        # 获取 Eagle Layer 每步每分支的最大 token 数（用于峰值 KV cache 计算）
        tokens_per_branch = self.eagle_layer.total_tokens + 1
        
        evt_start_p = torch.cuda.Event(enable_timing=True)
        evt_after_verify_p = torch.cuda.Event(enable_timing=True)
        evt_after_update_p = torch.cuda.Event(enable_timing=True)
        evt_after_draft_p = torch.cuda.Event(enable_timing=True)
        
        for step_parallel in range(max_new_tokens):
            # 峰值 KV cache 检查：确保下一步不会溢出
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                print(f"Incomplete branches due to KV cache limit: {self.active_branches}")
                break
            
            if step_parallel % 50 == 0:
                print(f"Parallel Decoding Step {step_parallel + 1}")

            evt_start_p.record()
            
            # 执行单步并行 decode
            (
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                accept_length,
                all_finished,
            ) = self.decode_step_parallel(
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
                evt_after_verify=evt_after_verify_p,
                evt_after_update=evt_after_update_p,
            )
            evt_after_draft_p.record()
            
            # 统计
            total_accept_len_parallel += accept_length.sum()
            
            # 计时统计
            torch.cuda.synchronize()
            total_verify_time_parallel += evt_start_p.elapsed_time(evt_after_verify_p) / 1000
            total_update_time_parallel += evt_after_verify_p.elapsed_time(evt_after_update_p) / 1000
            total_draft_time_parallel += evt_after_update_p.elapsed_time(evt_after_draft_p) / 1000
            
            # 停止条件检查
            if all_finished:
                print("All branches finished generation.")
                break
        
        # 计算并行阶段统计
        num_steps_parallel = max(step_parallel, 1)
        avg_accept_len = total_accept_len_parallel.item() / num_steps_parallel
        avg_draft_time = total_draft_time_parallel / num_steps_parallel
        avg_update_time = total_update_time_parallel / num_steps_parallel
        avg_verify_time = total_verify_time_parallel / num_steps_parallel
        
        print(f"Avg accepted lengths: {avg_accept_len}, "
              f"Avg draft time: {avg_draft_time:.4f}s, "
              f"Avg update time: {avg_update_time:.4f}s, "
              f"Avg verify time: {avg_verify_time:.4f}s")

        # 合并结果
        merged_ids = merge_outputs(
            skeleton_output=self.skeleton_output,
            parallel_branches_output=self.parallel_branches_output,
            instruction_len=self.instruction_len,
            para_token_ids=para_token_ids,
            num_para=num_para,
            device=self.base_model.device,
        )
        
        return merged_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time

    # =========================================================================
    # Prefill 和 Verify 方法 (Prefill & Verify Methods)
    # =========================================================================

    def prefill_single(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        单序列模式：Prefill 阶段（处理输入 prompt，初始化 KV Cache 和首次 Draft）
        
        流程：
        1. Base Model Prefill: 处理输入，获取最后一个 token 的 logits
        2. Sample Root Token: 采样第一个生成的 token
        3. Generate Draft Tree: 使用 Eagle Layer 生成候选树
        
        Args:
            input_ids: 输入 token IDs [1, seq_len]
            logits_processor: logits 处理器
            
        Returns:
            draft_tokens: 候选 tokens [1, tree_size]
            retrieve_indices: 检索索引 [num_leaves, depth]
            tree_mask: 树掩码 [1, 1, tree_size, tree_size]
            tree_position_ids: 位置编码 [1, tree_size]
            orig: 原始 logits
            hidden_states: 隐藏状态
            token: 采样的 root token
        """
        # Base Model Prefill
        outputs, orig, hidden_states = self(
            input_ids, past_key_values=self.past_key_values, output_orig=True
        )

        # Sample Root Token
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(input_ids, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]
        
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

        # Prepare Hidden States for Eagle Layer
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)

        # Generate Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.eagle_layer.generate_draft_tree(hidden_states, input_ids)

        return (
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            orig, hidden_states, token
        )

    def prefill_parallel(
        self,
        prefix_len: int,
        input_ids: torch.Tensor,
        tips_indices: torch.Tensor,
        branch_begins: List[int],
        branch_lengths: List[int],
        draft_input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        并行模式：Prefill 阶段（处理各分支 prompt，初始化并行状态和首次 Draft）
        
        处理多个分支的并行 Prefill，共享 prefix 的 KV Cache。
        
        Args:
            prefix_len: 共享前缀长度
            input_ids: 打包后的输入 [1, total_len]
            tips_indices: 各分支 tip 位置
            branch_begins: 各分支起始位置
            branch_lengths: 各分支长度
            draft_input_ids: Draft 模型输入 [num_para, max_len]
            logits_processor: logits 处理器
            
        Returns:
            input_ids: 更新后的输入
            draft_tokens: 候选 tokens
            retrieve_indices: 检索索引
            tree_mask: 树掩码
            tree_position_ids: 位置编码
            hidden_states: 隐藏状态
        """
        device = input_ids.device
        num_para = len(branch_lengths)

        # 构建并行 Prefill 的 Attention Mask
        attention_mask = build_parallel_prefill_mask(
            self.branch_index_map,
            prefix_len,
            branch_len=input_ids.shape[1] - prefix_len,
            device=device,
            dtype=torch.float32,
        )

        # 从 KV Cache 中获取已处理的长度
        kv_len = self.past_key_values[0][0].shape[2]
        position_ids = self.full_position_ids

        # Base Model Forward (Prefill)
        outputs, hidden_states = self(
            input_ids=input_ids[:, kv_len:],
            attention_mask=attention_mask,
            position_ids=position_ids[kv_len:],
            past_key_values=self.past_key_values,
            output_orig=False,
        )

        # 提取各分支 Tip 的 Logits
        tips_hidden = hidden_states[:, tips_indices - kv_len, :]
        tips_logits = self.base_model.lm_head(tips_hidden)
        current_logits = tips_logits.squeeze(0)  # [num_para, vocab]

        # Sample Root Tokens
        if logits_processor is not None:
            current_logits = logits_processor(None, current_logits)
            probs = torch.nn.functional.softmax(current_logits, dim=-1)
            root_tokens = torch.multinomial(probs, num_samples=1)
        else:
            root_tokens = torch.argmax(current_logits, dim=-1, keepdim=True)

        # 准备 Draft 模型输入
        draft_input_ids = torch.cat([draft_input_ids, root_tokens], dim=1)

        # 处理 Hidden States for Eagle Layer
        ea_device = self.eagle_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        packed_hidden = torch.cat(outputs["hidden_states"], dim=-1)[0]

        # 提取各分支的 Hidden States
        branch_hidden_list = []
        for i in range(num_para):
            start = branch_begins[i]
            end = start + branch_lengths[i]
            branch_hidden_list.append(packed_hidden[start - prefix_len:end - prefix_len])
        
        batched_hidden = stack_with_left_padding(branch_hidden_list, pad_id=0, device=device)

        # Generate Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.eagle_layer.generate_draft_tree(batched_hidden, draft_input_ids, prefix_len=prefix_len)

        return (
            input_ids, draft_tokens, retrieve_indices, tree_mask,
            tree_position_ids, hidden_states
        )

    # =========================================================================
    # 单步解码 (Decode Step)
    # =========================================================================

    def decode_step_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        evt_after_verify: Optional[torch.cuda.Event] = None,
        evt_after_update: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单序列投机解码单步
        
        执行一轮完整的 Verify -> Evaluate -> Update -> Draft 流程
        
        Args:
            input_ids: 当前输入 token IDs [1, seq_len]
            draft_tokens: 当前 draft tokens
            retrieve_indices: 检索索引
            tree_mask: tree attention mask
            tree_position_ids: tree 位置编码
            logits_processor: logits 处理器列表
            evt_after_verify: verify 完成后记录的 CUDA event (用于计时)
            evt_after_update: update 完成后记录的 CUDA event (用于计时)
            
        Returns:
            input_ids: 更新后的输入
            draft_tokens: 下一轮的 draft tokens
            retrieve_indices: 下一轮的检索索引
            tree_mask: 下一轮的 tree mask
            tree_position_ids: 下一轮的 tree 位置编码
            accept_length: 本轮接受的 token 数量
        """
        device = input_ids.device
        padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
        
        # -----------------------------------------------------------------
        # Step 1: Verify (Base Model Forward)
        # -----------------------------------------------------------------
        self.base_model.model.tree_mask = tree_mask
        draft_tokens = draft_tokens.to(device)
        
        logits, hidden_state_new, _ = self.verify_step_single(
            draft_tokens, self.past_key_values,
            tree_position_ids, input_ids, retrieve_indices
        )
        if evt_after_verify is not None:
            evt_after_verify.record()

        # -----------------------------------------------------------------
        # Step 2: Evaluate (Compare Candidates)
        # -----------------------------------------------------------------
        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]
        
        best_candidate, accept_length, sample_p = evaluate_single(input_ids, logits, candidates, logits_processor)
        # print(f"accept_length: {accept_length.item()}")
        # if accept_length.item() > 0:
        #     best_candidate_token = candidates[0, best_candidate, :accept_length]
        #     print(f"best_candidate: {self.tokenizer.decode(best_candidate_token[0].tolist())}")
        
        # 规范化维度
        if isinstance(accept_length, int):
            accept_length = torch.tensor(accept_length, device=device)
        if best_candidate.ndim == 0:
            best_candidate = best_candidate.unsqueeze(0)
        if accept_length.ndim == 0:
            accept_length = accept_length.unsqueeze(0)

        # -----------------------------------------------------------------
        # Step 3: Update State
        # -----------------------------------------------------------------
        input_ids, draft_input_ids, accept_hidden = self.update_state_single(
            input_ids, candidates, best_candidate, accept_length,
            retrieve_indices, logits_processor, hidden_state_new, sample_p
        )
        if evt_after_update is not None:
            evt_after_update.record()

        # -----------------------------------------------------------------
        # Step 4: Generate Next Draft
        # -----------------------------------------------------------------
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.eagle_layer.generate_draft_tree(accept_hidden, input_ids=draft_input_ids)

        return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, accept_length

    def decode_step_parallel(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        evt_after_verify: Optional[torch.cuda.Event] = None,
        evt_after_update: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        并行分支投机解码单步
        
        执行一轮完整的 Verify -> Evaluate -> Update -> Draft 流程（并行版本）
        
        Args:
            draft_tokens: 当前 draft tokens [num_branches, num_nodes]
            retrieve_indices: 检索索引
            tree_mask: tree attention mask
            tree_position_ids: tree 位置编码
            logits_processor: logits 处理器列表
            evt_after_verify: verify 完成后记录的 CUDA event (用于计时)
            evt_after_update: update 完成后记录的 CUDA event (用于计时)
            
        Returns:
            draft_tokens: 下一轮的 draft tokens
            retrieve_indices: 下一轮的检索索引
            tree_mask: 下一轮的 tree mask
            tree_position_ids: 下一轮的 tree 位置编码
            accept_length: 本轮各分支接受的 token 数量
            all_finished: 是否所有分支都已完成
        """
        device = self.base_model.device
        
        # -----------------------------------------------------------------
        # Step 1: Verify (Parallel Base Model Forward)
        # -----------------------------------------------------------------
        num_nodes = draft_tokens.shape[1]
        
        logits, hidden_states = self.verify_step_parallel(
            draft_tokens, tree_position_ids, tree_mask, num_nodes,
            self.branch_index_map, self.active_branches, self.eagle_layer.full_position_ids
        )
        if evt_after_verify is not None:
            evt_after_verify.record()

        # -----------------------------------------------------------------
        # Step 2: Evaluate (Per-Branch Evaluation)
        # -----------------------------------------------------------------
        best_candidate, accept_length, sample_logits = evaluate_parallel(
            logits, draft_tokens, retrieve_indices, logits_processor
        )

        # 采样 Bonus Token
        if logits_processor is not None:
            sample_tokens = torch.multinomial(sample_logits, 1)
            if sample_tokens.ndim == 1:
                sample_tokens = sample_tokens.unsqueeze(0)
        else:
            sample_tokens = torch.argmax(sample_logits, dim=-1, keepdim=True)

        # -----------------------------------------------------------------
        # Step 3: Update State & Prepare Next Draft
        # -----------------------------------------------------------------
        next_tips_hidden, next_tips_tokens = self.update_state_parallel(
            best_candidate, accept_length, draft_tokens,
            retrieve_indices, hidden_states, sample_tokens, num_nodes
        )
        
        # 检查是否所有分支都已完成
        all_finished = not self.active_branches
        
        if all_finished:
            if evt_after_update is not None:
                evt_after_update.record()
            return None, None, None, None, accept_length, True
        
        if evt_after_update is not None:
            evt_after_update.record()

        # -----------------------------------------------------------------
        # Step 4: Generate Next Draft
        # -----------------------------------------------------------------
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.eagle_layer.generate_draft_tree(
                next_tips_hidden, next_tips_tokens, active_branch=self.active_branches
            )

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, accept_length, False

    def verify_step_single(
        self,
        tree_candidates: torch.Tensor,
        past_key_values,
        tree_position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        retrieve_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        单序列模式：Verify 步骤 (Base Model 验证)
        
        Args:
            tree_candidates: 候选 tokens [1, tree_size]
            past_key_values: KV Cache
            tree_position_ids: 树位置编码 [1, tree_size]
            input_ids: 当前输入 [1, seq_len]
            retrieve_indices: 检索索引 [num_leaves, depth]
            
        Returns:
            logits: 验证后的 logits [num_leaves, depth, vocab]
            hidden_state: 隐藏状态
            outputs: 模型原始输出
        """
        # 计算绝对位置
        position_ids = tree_position_ids + input_ids.shape[1]
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Base Model Forward
        outputs, tree_logits, hidden_state = self(
            tree_candidates,
            output_orig=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # 处理 Hidden States for Eagle Layer (if Eagle3)
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

        # 按检索索引重组 logits
        logits = tree_logits[0, retrieve_indices]
        
        return logits, hidden_state, outputs

    def verify_step_parallel(
        self,
        draft_tokens: torch.Tensor,
        tree_position_ids: torch.Tensor,
        tree_mask: torch.Tensor,
        num_nodes: int,
        branch_index_map: torch.Tensor,
        active_branches: List[int],
        eagle_full_position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行模式：Verify 步骤 (Parallel Base Model 验证)
        
        该函数包含并行验证掩码构建和 Base Model 前向传播两个步骤。
        
        Args:
            draft_tokens: Draft tokens [num_para, num_nodes]
            tree_position_ids: Tree 位置编码 [num_para, num_nodes]
            tree_mask: Draft Tree 的掩码 [num_para, 1, num_nodes, num_nodes]
            num_nodes: 每个分支的 Draft 节点数
            branch_index_map: 分支索引映射
            active_branches: 当前活跃的分支列表
            eagle_full_position_ids: Eagle Layer 的完整位置编码
            
        Returns:
            logits: [num_para, num_nodes, vocab_size]
            hidden_states: 隐藏状态
        """
        device = draft_tokens.device
        num_para = draft_tokens.shape[0]
        current_length = self.current_length_data[0].item()
        
        # =====================================================================
        # Step 1: 构建并行验证的注意力掩码
        # =====================================================================
        history_bim = branch_index_map[:current_length]
        packed_draft_len = num_para * num_nodes
        
        # 初始化 Cross Mask (全部遮蔽)
        cross_mask = torch.full(
            (1, 1, packed_draft_len, current_length),
            torch.finfo(torch.float32).min, device=device
        )
        
        # 计算 Draft tokens 的分支归属
        active_ids_tensor = torch.tensor(active_branches, device=device)
        draft_branch_ids = active_ids_tensor.repeat_interleave(num_nodes)
        
        # Prefix 全部可见 (BIM == -1)
        is_prefix = (history_bim == -1).view(1, 1, 1, -1)
        cross_mask.masked_fill_(is_prefix, 0)
        
        # 同分支可见
        draft_ids_view = draft_branch_ids.view(1, 1, -1, 1)
        hist_ids_view = history_bim.view(1, 1, 1, -1)
        is_same_branch = (draft_ids_view == hist_ids_view)
        cross_mask.masked_fill_(is_same_branch, 0)
        
        # 构建 Draft Block Mask (块对角)
        converted_tree_mask = torch.where(
            tree_mask == 1, 0.0, torch.finfo(torch.float32).min
        )
        draft_block_mask = torch.full(
            (packed_draft_len, packed_draft_len),
            torch.finfo(torch.float32).min, device=device
        )
        for i in range(num_para):
            st, ed = i * num_nodes, (i + 1) * num_nodes
            draft_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :]

        draft_block_mask = draft_block_mask.unsqueeze(0).unsqueeze(0)
        
        # 合并
        combined_mask = torch.cat([cross_mask, draft_block_mask], dim=-1)
        
        # =====================================================================
        # Step 2: Base Model Forward
        # =====================================================================
        flat_draft_tokens = draft_tokens.reshape(1, -1)
        
        # 计算绝对位置
        current_tip_pos = eagle_full_position_ids[:, -1].unsqueeze(-1)
        abs_draft_pos = tree_position_ids + current_tip_pos + 1
        flat_draft_pos = abs_draft_pos.view(1, -1)
        
        # Base Model Forward
        outputs, hidden_states = self(
            flat_draft_tokens,
            past_key_values=self.past_key_values,
            attention_mask=combined_mask,
            position_ids=flat_draft_pos,
            output_orig=False,
        )
        
        # 计算 Logits
        logits = self.base_model.lm_head(hidden_states)
        logits = logits.view(num_para, num_nodes, -1)
        
        # 处理 Hidden States for Eagle Layer
        ea_device = self.eagle_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        
        return logits, hidden_states

    def update_state_single(
        self,
        input_ids: torch.Tensor,
        candidates: torch.Tensor,
        best_candidate: torch.Tensor,
        accept_length: torch.Tensor,
        retrieve_indices: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList],
        hidden_state_new: torch.Tensor,
        sample_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        更新推理状态（单序列模式）
        
        操作：
        1. 将接受的 tokens 添加到 input_ids
        2. 更新 KV Cache (搬运接受的 KV)
        3. 采样 Bonus Token
        4. 提取接受路径的 Hidden States (供上层生成 Draft Tree)
        
        Args:
            input_ids: 当前输入 [1, seq_len]
            candidates: 候选 tokens [num_leaves, depth]
            best_candidate: 最佳候选索引 [1]
            accept_length: 接受长度 [1]
            retrieve_indices: 检索索引 [num_leaves, depth]
            logits_processor: logits 处理器
            hidden_state_new: 新的隐藏状态
            sample_p: 采样概率分布
            
        Returns:
            input_ids: 更新后的输入
            draft_input_ids: Draft 模型的输入 (含 Bonus Token)
            accept_hidden: 接受路径的隐藏状态 (供生成下一轮 Draft Tree)
        """
        prev_input_len = input_ids.shape[1]
        bc = best_candidate[0]
        al = accept_length[0]
        
        # 提取接受的 tokens
        new_tokens = candidates[0, bc, :al + 1].unsqueeze(0)
        select_indices = retrieve_indices[0, bc, :al + 1] + prev_input_len

        # 更新 input_ids
        input_ids = torch.cat([input_ids, new_tokens.to(input_ids.device)], dim=-1)
        # print(f"Updated input_ids: {self.tokenizer.decode(new_tokens[0])}")

        # 更新 KV Cache (搬运接受的 KV 到正确位置)
        for past_kv_data in self.past_key_values_data:
            tgt = past_kv_data.index_select(dim=-2, index=select_indices.to(past_kv_data.device))
            dst = past_kv_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
            self.current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # 采样 Bonus Token
        if logits_processor is not None:
            token = torch.multinomial(sample_p, 1)
            token = token[None] if token.ndim == 1 else token
        else:
            token = torch.argmax(sample_p, dim=-1, keepdim=True)
        
        draft_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

        # 提取接受路径的 Hidden States
        retrieve_hidden = hidden_state_new[:, retrieve_indices[0]]
        accept_hidden = retrieve_hidden[:, best_candidate[0], :accept_length[0] + 1]

        return input_ids, draft_input_ids, accept_hidden

    def update_state_parallel(
        self,
        best_candidate: torch.Tensor,
        accept_length: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        hidden_states: torch.Tensor,
        sample_tokens: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        更新并行解码状态
        
        核心操作：
        1. 将接受的 KV 搬运到 History 末尾
        2. 更新 BIM
        3. 处理分支完成/剪枝
        4. 准备下一轮 Draft 输入
        
        Args:
            best_candidate: 每个分支的最佳候选索引
            accept_length: 每个分支的接受长度
            draft_tokens: Draft tokens
            retrieve_indices: 检索索引
            hidden_states: 隐藏状态
            sample_tokens: 采样的 Bonus tokens
            num_nodes: 节点数
            
        Returns:
            next_tips_hidden: 下一轮的 tip hidden states
            next_tips_tokens: 下一轮的 tip tokens
        """
        device = self.base_model.device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # 获取当前有效历史长度
        valid_history_len = (self.branch_index_map != -2).sum().item()
        dst_ptr = valid_history_len
        last_pos = self.eagle_layer.full_position_ids[:, -1].tolist()

        new_bim_entries = []
        new_pos_list = []
        draft_update_tokens_list = []
        draft_update_hiddens_list = []
        keep_mask_list = [True] * len(self.active_branches)

        # 遍历每个活跃分支
        for i, branch_idx in enumerate(self.active_branches):
            acc_len_i = accept_length[i].item()
            best_idx_i = best_candidate[i].item()

            # 提取接受的 tokens
            select_indices = retrieve_indices[i, best_idx_i, :acc_len_i + 1]
            seq_tokens = draft_tokens[i][select_indices]
            bonus_token = sample_tokens[i].item()
            is_finished = (bonus_token == eos_token_id)

            # 更新分支输出
            tokens_to_add = seq_tokens.tolist()
            if is_finished:
                print(f"Branch {branch_idx} finished with EOS.")
                keep_mask_list[i] = False
                tokens_to_add.append(bonus_token)
            self.parallel_branches_output[branch_idx].extend(tokens_to_add)

            # 更新 KV Cache (搬运接受的 KV)
            cache_indices = select_indices + num_nodes * i
            for past_kv_data in self.past_key_values_data:
                tgt = past_kv_data.index_select(
                    dim=-2,
                    index=(cache_indices + valid_history_len).to(device)
                )
                tgt_len = tgt.shape[-2]
                dst = past_kv_data[..., dst_ptr: dst_ptr + tgt_len, :]
                dst.copy_(tgt, non_blocking=True)
                self.current_length_data.fill_(dst_ptr + tgt_len)
            
            dst_ptr += tgt_len
            new_bim_entries.extend([branch_idx] * tgt_len)

            # 更新位置编码
            pos = torch.arange(
                last_pos[i] + 1, last_pos[i] + 1 + tgt_len,
                device=device, dtype=torch.long
            )
            self.full_position_ids = torch.cat([self.full_position_ids, pos])

            # 准备下一轮 Draft 输入
            if not is_finished:
                hidden_branch = hidden_states[:, i * num_nodes: (i + 1) * num_nodes]
                seq_hiddens = hidden_branch[0, select_indices, :]

                draft_tokens_tensor = torch.cat([
                    seq_tokens[1:],
                    torch.tensor([bonus_token], device=device)
                ])
                draft_update_tokens_list.append(draft_tokens_tensor)
                draft_update_hiddens_list.append(seq_hiddens)
                new_pos_list.append(pos)

        # 更新 BIM
        if new_bim_entries:
            self.branch_index_map[valid_history_len:dst_ptr] = torch.tensor(
                new_bim_entries, device=device
            )
        self.branch_index_map[dst_ptr:] = -2
        self.current_length_data.fill_(dst_ptr)

        # 分支剪枝
        keep_mask_tensor = torch.tensor(keep_mask_list, device=device, dtype=torch.bool)
        if not torch.all(keep_mask_tensor):
            self.eagle_layer.full_position_ids = \
                self.eagle_layer.full_position_ids[keep_mask_tensor]
            self.eagle_layer.cache_padding_mask = \
                self.eagle_layer.cache_padding_mask[keep_mask_tensor]
            self.active_branches = [
                b for b, keep in zip(self.active_branches, keep_mask_list) if keep
            ]
            
            # 剪枝 Eagle Layer KV Cache
            if self.eagle_layer.stable_kv is not None:
                k_stable, v_stable = self.eagle_layer.stable_kv[0]
                k_stable = k_stable[keep_mask_tensor]
                v_stable = v_stable[keep_mask_tensor]
                self.eagle_layer.stable_kv = ((k_stable, v_stable),)

        if not self.active_branches:
            return None, None

        # 更新 Draft 模型状态
        batched_draft_tokens, batch_mask = stack_with_left_padding(
            draft_update_tokens_list, pad_id=pad_token_id, device=device, return_mask=True
        )
        batched_draft_hiddens = stack_with_left_padding(
            draft_update_hiddens_list, pad_id=0, device=device
        )
        batched_new_pos = stack_with_left_padding(
            new_pos_list, pad_id=0, device=device
        )
        
        self.eagle_layer.cache_padding_mask = torch.cat(
            [self.eagle_layer.cache_padding_mask, batch_mask], dim=1
        )
        self.eagle_layer.full_position_ids = torch.cat(
            [self.eagle_layer.full_position_ids, batched_new_pos], dim=1
        )

        return batched_draft_hiddens, batched_draft_tokens

    # =========================================================================
    # 前缀复用：从 Skeleton 到 Parallel 的状态转换
    # =========================================================================

    def reuse_prefix_for_parallel(
        self,
        prefix_ids: torch.Tensor,
        branches_prompts: List[List[int]],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], torch.Tensor]:
        """
        前缀复用：从 Skeleton 模式复用 KV Cache 到 Parallel 模式
        
        该方法在 Skeleton 生成完成后调用，复用 skeleton 的 prefix KV Cache，
        为各分支初始化并行解码状态。
        
        核心步骤：
        1. 复制 KV Cache: 将 skeleton 的 prefix KV 复制为多分支
        2. 构建 BIM: 创建 Branch Index Map，追踪每个 token 的分支归属
        3. 打包输入: 将所有分支的 prompt 打包成一个序列
        4. 位置编码: 为各分支构建正确的位置编码
        
        关键数据结构：
        - Branch Index Map (BIM): 记录每个 token 属于哪个分支
          - -1: 共享 prefix
          - -2: 空位置 (预留空间)
          - >=0: 分支索引
        - Packed Input: 将所有分支的 token 打包成一个序列
          格式: [prefix | branch_0 | branch_1 | ... | branch_n]
        
        Args:
            prefix_ids: 共享前缀 (skeleton) [1, prefix_len]
            branches_prompts: 各分支的 prompt token 列表
            max_new_tokens: 每个分支最大生成长度
            
        Returns:
            input_ids: 打包后的输入 [1, total_len]
            tips_indices: 各分支 tip 位置 (最后一个 token 的索引)
            branch_begins: 各分支起始位置
            branch_lengths: 各分支初始长度
            draft_input_ids: Draft 模型的输入 [num_para, max_len]
        """
        device = self.base_model.device
        num_para = len(branches_prompts)
        prefix_len = prefix_ids.shape[1]
        
        # 初始化活跃分支列表
        self.active_branches = list(range(num_para))

        # ---------------------------------------------------------------------
        # 1. Base Model KV Cache: 重置到 prefix 长度
        # ---------------------------------------------------------------------
        self.current_length_data.fill_(prefix_len)
        
        # ---------------------------------------------------------------------
        # 2. Eagle Layer KV Cache: 复制 prefix KV 为多分支
        # ---------------------------------------------------------------------
        if self.eagle_layer.stable_kv is not None:
            k_draft, v_draft = self.eagle_layer.stable_kv[0]
            k_prefix = k_draft[..., :prefix_len, :].clone()
            v_prefix = v_draft[..., :prefix_len, :].clone()
            k_expanded = k_prefix.expand(num_para, -1, -1, -1).clone()
            v_expanded = v_prefix.expand(num_para, -1, -1, -1).clone()
            self.eagle_layer.stable_kv = ((k_expanded, v_expanded),)

        # ---------------------------------------------------------------------
        # 3. 构建打包输入序列和 Branch Index Map
        # ---------------------------------------------------------------------
        flat_branch_ids = []
        branch_index_list = [-1] * prefix_len  # Prefix 标记为 -1
        
        tips_indices = []
        branch_begins = []
        branch_lengths = []
        pos_ids_list = list(range(prefix_len))
        
        draft_branch_list = []
        draft_pos_list = []
        
        current_offset = prefix_len
        for i, br in enumerate(branches_prompts):
            curr_len = len(br)
            branch_begins.append(current_offset)
            flat_branch_ids.extend(br)
            branch_index_list.extend([i] * curr_len)
            branch_lengths.append(curr_len)
            current_offset += curr_len
            tips_indices.append(current_offset - 1)
            
            # 位置编码: 每个分支从 prefix_len 开始独立计数
            curr_pos = list(range(prefix_len, prefix_len + curr_len))
            pos_ids_list.extend(curr_pos)
            
            draft_branch_list.append(torch.tensor(br, device=device, dtype=torch.long))
            draft_pos_list.append(torch.tensor(curr_pos, device=device, dtype=torch.long))

        # 构建 Tensor
        branches_tensor = torch.tensor([flat_branch_ids], device=device, dtype=torch.long)
        input_ids = torch.cat([prefix_ids, branches_tensor], dim=1)
        tips_indices = torch.tensor(tips_indices, device=device)
        self.full_position_ids = torch.tensor(pos_ids_list, device=device)

        # 初始化 BIM
        total_capacity = input_ids.shape[1] + max_new_tokens + 128
        self.branch_index_map = torch.full(
            (total_capacity,), -2, dtype=torch.long, device=device
        )
        self.branch_index_map[:len(branch_index_list)] = torch.tensor(
            branch_index_list, device=device
        )

        # 构建 Draft 模型的 Batched 输入（左填充）
        draft_input_ids, branch_mask = stack_with_left_padding(
            draft_branch_list, pad_id=self.tokenizer.pad_token_id,
            device=device, return_mask=True
        )
        padded_branch_pos = stack_with_left_padding(draft_pos_list, pad_id=0, device=device)

        prefix_mask = torch.ones((num_para, prefix_len), dtype=torch.long, device=device)
        prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(num_para, -1)
        self.eagle_layer.cache_padding_mask = torch.cat([prefix_mask, branch_mask], dim=1)
        self.eagle_layer.full_position_ids = torch.cat([prefix_pos, padded_branch_pos], dim=1)

        return input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids

    