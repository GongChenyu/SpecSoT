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

from .utils import (
    prepare_logits_processor,
    reset_tree_mode,
    prefill_single,
    prefill_parallel,
    verify_step_single,
    verify_step_parallel,
    evaluate_posterior,
    update_inference_inputs,
    stack_with_left_padding,
    parse_skeleton,
    check_stop_conditions,
)

from .prompts import base_prompt, skeleton_trigger_zh, parallel_trigger_zh


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
        base_model_name_or_path: str,
        ea_model_path: str,
        use_eagle3: bool = True,
        total_token: int = 60,
        depth: int = 7,
        top_k: int = 10,
        threshold: float = 1.0,
        ea_layer_state_dict: dict = None,
    ):
        """
        初始化 SpecSoT 模型
        
        初始化流程：
        1. Base Model 初始化: 设置基础模型和配置
        2. Eagle Layer 初始化: 创建并配置草稿层
        3. 推理状态初始化: 重置所有运行时状态
        
        Args:
            base_model: 预加载的基础模型
            base_model_name_or_path: 基础模型路径，用于加载 tokenizer
            ea_model_path: Eagle 模型配置路径
            use_eagle3: 是否使用 Eagle3 架构
            total_token: 每次 draft 生成的总 token 数
            depth: draft 树的深度
            top_k: 每层选择的 top-k 数量
            threshold: 接受阈值
            ea_layer_state_dict: Eagle 层的预训练权重
        """
        super().__init__()
        
        # =====================================================================
        # 1. Base Model 初始化
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
        # 2. Eagle Layer 初始化
        # =====================================================================
        self._init_eagle_layer(
            ea_model_path, use_eagle3, total_token, depth, top_k, threshold, ea_layer_state_dict
        )
        
        # =====================================================================
        # 3. 推理状态初始化
        # =====================================================================
        self._init_inference_state()

    # =========================================================================
    # 初始化方法 (Initialization Methods)
    # =========================================================================

    def _init_eagle_layer(
        self,
        ea_model_path: str,
        use_eagle3: bool,
        total_token: int,
        depth: int,
        top_k: int,
        threshold: float,
        ea_layer_state_dict: dict,
    ):
        """
        初始化 Eagle Layer (草稿层)
        
        步骤：
        1. 加载配置: 从 config.json 加载 Eagle 配置
        2. 创建 Eagle Layer: 初始化草稿模型
        3. 设备管理: 处理跨设备权重
        4. 词表映射: 配置 draft vocab 到 target vocab 的映射 (Eagle3)
        5. 加载权重: 加载预训练的 Eagle 权重
        6. Tree 初始化: 初始化 draft tree 相关 buffer
        """
        # 1. 加载配置
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        bias = con.get("bias", True)

        # 2. 创建 Eagle Layer
        self.eagle_layer = EagleLayer(
            config=config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            path=self.base_model_name_or_path,
            load_emb=True,
        )

        # 3. 设备管理
        device = self.base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != self.base_model.lm_head.weight.device:
            self.eagle_layer.diff_device = True
            self.eagle_layer.headweight = self.base_model.lm_head.weight.clone().to(device)
        else:
            self.eagle_layer.diff_device = False

        # 4. 词表映射处理 (Eagle3 特有)
        if use_eagle3 and config.vocab_size == config.draft_vocab_size:
            if hasattr(self.eagle_layer, 'd2t'):
                del self.eagle_layer.d2t
            if hasattr(self.eagle_layer, 't2d'):
                del self.eagle_layer.t2d

        # 5. 加载权重
        self.eagle_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.eagle_layer.to(self.base_model.dtype).to(device)
        
        # 6. Tree 初始化
        self.eagle_layer.init_tree()

    def _init_inference_state(self):
        """
        初始化推理状态
        
        包括：
        - Eagle Layer KV Cache
        - Base Model KV Cache
        - 并行解码状态 (BIM, position_ids 等)
        - 输出存储
        """
        # Eagle Layer 状态
        self.eagle_layer.reset_kv()
        
        # Base Model KV Cache (将在 generate 时动态分配)
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

        # 并行解码状态
        # Branch Index Map (BIM): 记录每个 token 属于哪个分支
        # -1: 共享 prefix, -2: 空位置, >=0: 分支索引
        self.branch_index_map = None
        self.full_position_ids = None
        self.active_branches = None
        
        # Eagle Layer 并行状态
        self.eagle_layer.cache_padding_mask = None
        self.eagle_layer.full_position_ids = None

        # 输出存储
        self.skeleton_output = None
        self.parallel_branches_output = None
        self.instruction_len = None

    # =========================================================================
    # 状态管理 (State Management)
    # =========================================================================

    def reset_state(self):
        """
        重置所有推理状态，为新一轮生成做准备
        
        调用时机：
        - 开始新的对话或任务
        - 清空所有历史状态
        """
        self._init_inference_state()

    def get_tokenizer(self):
        """获取 tokenizer"""
        return self.tokenizer

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
        self.reset_state()

        if not enable_parallel:
            # 普通投机解码模式
            return self._generate_standard(
                task_prompt, max_new_tokens, temperature, top_p, top_k
            )
        
        # 骨架并行模式
        return self._generate_with_skeleton(
            task_prompt, max_new_tokens, temperature, top_p, top_k, para_token_ids
        )

    def _generate_standard(
        self,
        task_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """标准投机解码生成（不使用骨架）"""
        input_ids = self.tokenizer([task_prompt], return_tensors="pt").input_ids.to(self.base_model.device)
        
        output_ids, avg_accept_len, avg_draft_time, avg_update_time, avg_verify_time = \
            self._decode_loop_single(
                input_ids, max_new_tokens, temperature, top_p, top_k,
                logits_processor=None, stop_token_id=None
            )
        
        return output_ids, avg_accept_len, 0, avg_draft_time, avg_update_time, avg_verify_time

    def _generate_with_skeleton(
        self,
        task_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        para_token_ids: Dict[str, int],
    ) -> Tuple[torch.Tensor, float, int, float, float, float]:
        """
        骨架并行生成模式
        
        三阶段流程：
        1. Skeleton Generation - 生成回答骨架
        2. Skeleton Parsing - 解析骨架，提取分支
        3. Parallel Decoding - 并行解码各分支
        """
        device = self.base_model.device
        
        # =====================================================================
        # Stage 1: Skeleton Generation (骨架生成)
        # =====================================================================
        task_input = base_prompt.format(user_question=task_prompt)
        task_input_ids = self.tokenizer([task_input], return_tensors="pt").input_ids.to(device)
        skeleton_input_ids = self.tokenizer([skeleton_trigger_zh], return_tensors="pt").input_ids.to(device)
        input_ids = torch.cat([task_input_ids, skeleton_input_ids], dim=-1)

        # 构造语义约束 Logits Processor
        sp_processor = SemanticLogitsProcessor(
            para_end_token_id=para_token_ids['para_end_token_id'],
            ellipsis_token_id=para_token_ids['ellipsis_token_id'],
            line_break_token_id=para_token_ids['line_break_token_id'],
            para_begin_token_id=para_token_ids['para_begin_token_id'],
            colon_token_id=para_token_ids['colon_token_id'],
            cn_colon_token_id=para_token_ids['cn_colon_token_id'],
            colon_new_line_token_id=para_token_ids['colon_new_line_token_id'],
            prefix_len=input_ids.shape[1]
        )
        
        skeleton_ids, avg_accept_len, avg_draft_time, avg_update_time, avg_verify_time = \
            self._decode_loop_single(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                logits_processor=LogitsProcessorList([sp_processor]),
                stop_token_id=para_token_ids['para_end_token_id'],
                max_steps=150,  # 骨架长度限制
            )
        
        print("Generated Skeleton:", self.tokenizer.decode(skeleton_ids[0]))
        self.skeleton_output = skeleton_ids.clone()

        # =====================================================================
        # Stage 2: Skeleton Parsing (骨架解析)
        # =====================================================================
        clean_branches, instruction_len = parse_skeleton(self.tokenizer, skeleton_ids, para_token_ids)
        
        if not clean_branches:
            print("Parsing failed or no branches found. Returning skeleton.")
            return skeleton_ids, avg_accept_len, 0, avg_draft_time, avg_update_time, avg_verify_time

        num_para = len(clean_branches)
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        print(f"Detected {num_para} parallel branches.")

        # =====================================================================
        # Stage 3: Parallel Decoding (并行分支解码)
        # =====================================================================
        avg_accept_len, avg_draft_time, avg_update_time, avg_verify_time = \
            self._decode_loop_parallel(
                prefix_ids=task_input_ids,
                branches_prompts=clean_branches,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        # 合并结果
        merged_ids = self._merge_outputs(para_token_ids, num_para)
        
        return merged_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time

    def _merge_outputs(
        self, para_token_ids: Dict[str, int], num_para: int
    ) -> torch.Tensor:
        """合并骨架和并行分支的输出"""
        skeleton_part = self.skeleton_output[0].tolist()
        skeleton_part.append(para_token_ids['line_break_token_id'])

        parallel_part = []
        for i in range(num_para):
            branch_output = self.parallel_branches_output[i][self.instruction_len[i]:]
            parallel_part.extend(branch_output)
            parallel_part.append(para_token_ids['line_break_token_id'])
            print(f"Branch {i} Length: {len(branch_output)}")

        merged_ids = skeleton_part + parallel_part
        merged_ids.append(para_token_ids['para_end_token_id'])

        return torch.tensor([merged_ids], device=self.base_model.device)

    # =========================================================================
    # 单序列解码循环 (Single Sequence Decode Loop)
    # =========================================================================

    def _decode_loop_single(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stop_token_id: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float, float, float, float]:
        """
        单序列投机解码循环
        
        流程：
        1. Prefill: 初始化 KV Cache 和 Draft Tree
        2. Decode Loop:
           - Draft: Eagle Layer 生成候选树
           - Verify: Base Model 验证
           - Update: 更新状态
        
        Args:
            input_ids: 输入 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            logits_processor: logits 处理器列表
            stop_token_id: 停止 token ID
            max_steps: 最大步数限制
            
        Returns:
            生成的 token IDs 和各项统计指标
        """
        device = input_ids.device
        input_len = input_ids.shape[1]
        
        # 准备 logits processor
        if temperature > 1e-5:
            lp = prepare_logits_processor(temperature, top_p, top_k)
            if logits_processor:
                lp.extend(logits_processor)
            logits_processor = lp

        # ---------------------------------------------------------------------
        # Prefill Phase: 初始化 KV Cache 和第一轮 Draft
        # ---------------------------------------------------------------------
        self.eagle_layer.reset_kv()
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        reset_tree_mode(self)
        
        # 1. Prefill 阶段: Base Model 处理输入 + Eagle Layer 生成首次 Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
            prefill_single(input_ids, self, logits_processor)

        # ---------------------------------------------------------------------
        # Decode Loop: Verify -> Evaluate -> Update -> Draft
        # ---------------------------------------------------------------------
        padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
        total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
        
        # 计时器
        total_draft_time = 0.0
        total_update_time = 0.0
        total_verify_time = 0.0
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_after_verify = torch.cuda.Event(enable_timing=True)
        evt_after_update = torch.cuda.Event(enable_timing=True)
        evt_after_draft = torch.cuda.Event(enable_timing=True)

        max_iterations = max_steps if max_steps else max_new_tokens
        
        for step in range(max_iterations):
            evt_start.record()
            
            # -----------------------------------------------------------------
            # Step 1: Verify (Base Model Forward)
            # -----------------------------------------------------------------
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(device)
            
            logits, hidden_state_new, _ = verify_step_single(
                self, draft_tokens, self.past_key_values,
                tree_position_ids, input_ids, retrieve_indices
            )
            evt_after_verify.record()

            # -----------------------------------------------------------------
            # Step 2: Evaluate (Compare Candidates)
            # -----------------------------------------------------------------
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            
            # 规范化维度
            if isinstance(accept_length, int):
                accept_length = torch.tensor(accept_length, device=device)
            if best_candidate.ndim == 0:
                best_candidate = best_candidate.unsqueeze(0)
            if accept_length.ndim == 0:
                accept_length = accept_length.unsqueeze(0)
            total_accept_len += accept_length.sum()
            evt_after_update.record()

            # -----------------------------------------------------------------
            # Step 3: Update State
            # -----------------------------------------------------------------
            input_ids, draft_input_ids, accept_hidden = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length,
                retrieve_indices, logits_processor, self, hidden_state_new, sample_p
            )
            evt_after_update.record()

            # -----------------------------------------------------------------
            # Step 4: Generate Next Draft
            # -----------------------------------------------------------------
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
                self.eagle_layer.generate_draft_tree(accept_hidden, input_ids=draft_input_ids)
            evt_after_draft.record()

            # 计时统计
            torch.cuda.synchronize()
            total_verify_time += evt_start.elapsed_time(evt_after_verify) / 1000
            total_update_time += evt_after_verify.elapsed_time(evt_after_update) / 1000
            total_draft_time += evt_after_update.elapsed_time(evt_after_draft) / 1000

            # 停止条件检查
            if check_stop_conditions(
                input_ids, input_len, stop_token_id, self.tokenizer.eos_token_id,
                self.current_length_data[0].item(), max_kv_len
            ):
                break

        # 计算平均值
        num_steps = max(step, 1)
        return (
            input_ids[:, input_len:],
            total_accept_len.item() / num_steps,
            total_draft_time / num_steps,
            total_update_time / num_steps,
            total_verify_time / num_steps,
        )

    # =========================================================================
    # 并行序列解码循环 (Parallel Sequence Decode Loop)
    # =========================================================================

    def _decode_loop_parallel(
        self,
        prefix_ids: torch.Tensor,
        branches_prompts: List[List[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[float, float, float, float]:
        """
        并行分支解码循环
        
        特点：
        - 多个分支共享 prefix 的 KV Cache
        - 使用 Branch Index Map (BIM) 管理分支归属
        - 支持动态分支剪枝（分支完成后移除）
        
        Args:
            prefix_ids: 共享前缀的 token IDs
            branches_prompts: 各分支的 prompt token 列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            各项统计指标
        """
        device = self.base_model.device
        prefix_len = prefix_ids.shape[1]

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature, top_p, top_k)

        # ---------------------------------------------------------------------
        # 前缀复用：复用 Skeleton KV Cache + 初始化并行状态
        # ---------------------------------------------------------------------
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self.reuse_prefix_for_parallel(prefix_ids, branches_prompts, max_new_tokens)

        # 1. Prefill 阶段: Base Model 并行处理各分支 + Eagle Layer 生成首次 Draft Tree
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states = \
            prefill_parallel(
                prefix_len, input_ids, self, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )

        # ---------------------------------------------------------------------
        # Parallel Decode Loop
        # ---------------------------------------------------------------------
        total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
        total_verify_time = 0.0
        total_update_time = 0.0
        total_draft_time = 0.0
        
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_after_verify = torch.cuda.Event(enable_timing=True)
        evt_after_update = torch.cuda.Event(enable_timing=True)
        evt_after_draft = torch.cuda.Event(enable_timing=True)

        for step in range(max_new_tokens):
            evt_start.record()
            
            # -----------------------------------------------------------------
            # Step 1: Verify (Parallel Base Model Forward)
            # -----------------------------------------------------------------
            num_nodes = draft_tokens.shape[1]
            
            logits, hidden_states = verify_step_parallel(
                self, draft_tokens, tree_position_ids, tree_mask, num_nodes,
                self.branch_index_map, self.active_branches, self.eagle_layer.full_position_ids
            )
            evt_after_verify.record()

            # -----------------------------------------------------------------
            # Step 2: Evaluate (Per-Branch Evaluation)
            # -----------------------------------------------------------------
            best_candidate, accept_length, sample_logits = self._evaluate_parallel(
                logits, draft_tokens, retrieve_indices, logits_processor
            )
            total_accept_len += accept_length.sum()

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
            next_tips_hidden, next_tips_tokens = self._update_parallel_state(
                best_candidate, accept_length, draft_tokens,
                retrieve_indices, hidden_states, sample_tokens, num_nodes
            )
            
            if not self.active_branches:
                print("All branches finished generation.")
                break
            evt_after_update.record()

            # -----------------------------------------------------------------
            # Step 4: Generate Next Draft
            # -----------------------------------------------------------------
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
                self.eagle_layer.generate_draft_tree(
                    next_tips_hidden, next_tips_tokens, active_branch=self.active_branches
                )
            evt_after_draft.record()

            # 计时统计
            torch.cuda.synchronize()
            total_verify_time += evt_start.elapsed_time(evt_after_verify) / 1000
            total_update_time += evt_after_verify.elapsed_time(evt_after_update) / 1000
            total_draft_time += evt_after_update.elapsed_time(evt_after_draft) / 1000

        # 统计输出
        num_steps = max(step, 1)
        avg_accept_len = total_accept_len.item() / num_steps
        print(f"Avg accepted lengths: {avg_accept_len}, "
              f"Avg draft time: {total_draft_time/num_steps:.4f}s, "
              f"Avg update time: {total_update_time/num_steps:.4f}s, "
              f"Avg verify time: {total_verify_time/num_steps:.4f}s")

        return (
            avg_accept_len,
            total_draft_time / num_steps,
            total_update_time / num_steps,
            total_verify_time / num_steps,
        )

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

    def _evaluate_parallel(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """并行评估：为每个分支选择最佳候选"""
        num_para = logits.shape[0]
        device = logits.device
        
        retrieve_indices = retrieve_indices.to(device)
        draft_tokens = draft_tokens.to(device)
        
        # 处理无效索引
        padding_mask = (retrieve_indices == -1)
        safe_indices = retrieve_indices.clone()
        safe_indices[padding_mask] = 0
        
        # 提取候选 tokens
        candidates = torch.gather(
            draft_tokens.unsqueeze(1).expand(-1, retrieve_indices.size(1), -1),
            2, safe_indices
        )
        candidates.masked_fill_(padding_mask, 0)
        
        # 提取候选 logits
        vocab_size = logits.size(-1)
        flat_indices = safe_indices.view(num_para, -1).unsqueeze(-1).expand(-1, -1, vocab_size)
        candidate_logits = torch.gather(logits, 1, flat_indices)
        candidate_logits = candidate_logits.view(
            num_para, retrieve_indices.size(1), retrieve_indices.size(2), -1
        )
        
        # 评估
        best_candidate, accept_length, sample_logits = evaluate_posterior(
            candidate_logits, candidates, logits_processor
        )
        
        return best_candidate, accept_length, sample_logits

    def _update_parallel_state(
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
    # 类方法：模型加载
    # =========================================================================

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
        **kwargs,
    ) -> "SpecSoTModel":
        """
        从预训练模型加载 SpecSoT 模型
        
        Args:
            base_model_path: 基础模型路径
            ea_model_path: Eagle 模型路径
            use_eagle3: 是否使用 Eagle3
            total_token: 每次 draft 的总 token 数
            depth: draft 树深度
            top_k: top-k 选择数量
            threshold: 接受阈值
            **kwargs: 传递给基础模型的其他参数
            
        Returns:
            SpecSoTModel 实例
        """
        # 根据架构类型加载基础模型
        model_type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        
        model_class_map = {
            'LlamaForCausalLM': KVLlamaForCausalLM,
            'Qwen2ForCausalLM': KVQwen2ForCausalLM,
            'Qwen3ForCausalLM': KVQwen3ForCausalLM,
        }
        
        BaseModelClass = model_class_map.get(model_type, KVMixtralForCausalLM)
        base_model = BaseModelClass.from_pretrained(base_model_path, **kwargs)

        # 加载 Eagle 配置
        config_path = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = hf_hub_download(ea_model_path, "config.json")

        # 加载 Eagle 权重
        try:
            load_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_path):
                load_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_state_dict = torch.load(load_path, map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_path):
                load_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_state_dict = load_file(load_path)

        return cls(
            base_model=base_model,
            base_model_name_or_path=base_model_path,
            ea_model_path=config_path,
            use_eagle3=use_eagle3,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            ea_layer_state_dict=ea_state_dict,
        )
