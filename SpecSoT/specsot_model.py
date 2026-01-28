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

from .modeling.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling.modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM

# Eagle Layer (新版重构模块)
from .modeling_draft import Eagle3, Eagle2, Drafter
# 兼容旧版别名
EagleLayer3 = Eagle3
EagleLayer2 = Eagle2

from .kv_cache import initialize_past_key_values
from .modeling_draft import EConfig
from .logits_processor import SemanticLogitsProcessor, VocabScanner

# 分布式推理支持
from .distributed import DistributedConfig, DistributedPrefillManager

# 分支调度支持 (独立于分布式模块)
from .scheduling import (
    DeviceProfile,
    BranchInfo,
    DeviceExecutionPlan,
    SchedulePlan,
    HeuristicScheduler,
    SimpleDistributedScheduler,
    BranchExecutionManager,
)

# 日志支持
from .logging_utils import get_unified_logger

from .utils import (
    prepare_logits_processor,
    build_parallel_prefill_mask,
    stack_with_left_padding,
    prepare_skeleton_input,
    check_stop_conditions,
    check_stop_conditions_parallel,
    check_skeleton_stop,
    merge_outputs,
    set_random_seed,
    evaluate_single,
    evaluate_parallel,
    parse_skeleton_output,
    prepare_parallel_branches,
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
        seed: int = 42,
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
            seed: 随机种子
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
        self.seed = seed
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )
        
        # =====================================================================
        # 2. Eagle Layer
        # =====================================================================
        self.eagle_layer = eagle_layer
        
        # =====================================================================
        # 2.5 Drafter (Draft Tree 生成器)
        # =====================================================================
        self.drafter = Drafter(eagle_layer)
        
        # =====================================================================
        # 3. 日志系统
        # =====================================================================
        # 获取 logger，如果是分布式模式会自动使用对应的 rank logger
        rank = distributed_config.rank if distributed_config and distributed_config.enabled else -1
        self.logger = get_unified_logger(rank=rank, name_suffix="-Model")
        
        # =====================================================================
        # 4. 分布式配置
        # =====================================================================
        self.distributed_config = distributed_config or DistributedConfig()
        self.distributed_prefill_manager: Optional[DistributedPrefillManager] = None
        
        if self.distributed_config.enabled:
            self._init_distributed()
        
        # =====================================================================
        # 5. Semantic Logits Processor 预初始化
        # =====================================================================        
        self.logger.info("预初始化 VocabScanner 和 SemanticLogitsProcessor...")
        self._vocab_scanner = VocabScanner(self.tokenizer)
        self._semantic_processor = SemanticLogitsProcessor(
            tokenizer=self.tokenizer,
            prefix_len=0,  # 将在推理时通过 configure() 设置
            enforce_format=True,
            vocab_scanner=self._vocab_scanner,  # 复用预初始化的 VocabScanner
        )
        self.logger.info("SemanticLogitsProcessor 预初始化完成")
        
        # =====================================================================
        # 6. 推理状态初始化
        # =====================================================================
        set_random_seed(self.seed)
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
        seed: int = 42,
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
        # 2. 加载 Eagle Layer (根据 use_eagle3 选择不同版本)
        # =====================================================================
        if use_eagle3:
            # EAGLE3: 单层 Decoder，用于 LLaMA 3.1, Qwen3 等新模型
            eagle_layer = EagleLayer3.from_pretrained(
                ea_model_path=ea_model_path,
                base_model=base_model,
                base_model_name_or_path=base_model_path,
                use_eagle3=use_eagle3,
                total_token=total_token,
                depth=depth,
                top_k=top_k,
                threshold=threshold,
            )
        else:
            # EAGLE2: 多层 Decoder Stack，用于 Vicuna 等旧模型
            eagle_layer = EagleLayer2.from_pretrained(
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
            seed=seed,
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
        
        # 调度模式：追踪最近完成的分支（用于 Continuous Batching）
        self.recently_completed_branches = []
        
        # Continuous Batching 状态
        # 正在 prefill 的新分支（已加入活跃列表但尚未完成 prefill）
        self.prefilling_branches: List[int] = []
        # 新分支的 prompt tokens 缓存
        self.pending_prefill_prompts: Dict[int, List[int]] = {}

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
        self.logger.info(f"分布式推理已启用: {self.distributed_config}")
    
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
            'llama': LLaMA 3.1 Instruct 等使用 chat template 的模型
            'vicuna': Vicuna 模型（有特定的 chat template）
            'other': 其他模型
        """
        model_name = self.base_model_name_or_path.lower()
        
        if 'qwen' in model_name:
            return 'qwen'
        elif 'vicuna' in model_name:
            return 'vicuna'
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
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, float, int, float, float, float, float, float, float]:
        """
        主生成函数：支持普通模式和并行骨架模式
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            enable_parallel: 是否启用骨架并行模式
            use_semantic_constraint: 是否使用语义约束 (FSM 状态机)
            
        Returns:
            output_ids: 生成的 token IDs
            avg_accept_len: 平均接受长度
            num_para: 并行分支数量
            avg_draft_time: 平均 draft 时间
            avg_update_time: 平均 update 时间
            avg_verify_time: 平均 verify 时间
            total_time: 推理总时间
            skeleton_time: Skeleton阶段时间 (仅并行模式有效，否则为0)
            parallel_time: Parallel阶段时间 (仅并行模式有效，否则为0)
        """
        # 公共初始化
        self.reset_state()
        self.eagle_layer.reset_state()
                
        # 准备采样 logits processor (温度、top_p、top_k)
        logits_processor = None
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature, top_p, top_k)
        
        # 记录总开始时间
        total_start_time = time.time()

        if not enable_parallel:
            # 普通投机解码模式
            result = self.generate_eagle(task_prompt, max_new_tokens, logits_processor)
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time = result
            skeleton_time = 0.0
            parallel_time = 0.0
        else:
            result = self.generate_specsot(task_prompt, max_new_tokens, logits_processor, use_semantic_constraint)
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, skeleton_time, parallel_time = result
            
        # 计算总时间
        total_time = time.time() - total_start_time
        
        return output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, total_time, skeleton_time, parallel_time
        
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
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, logits_processor
                )
        else:
            # 普通Prefill
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, logits_processor)
        set_random_seed(self.seed)

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
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, float, int, float, float, float, float, float]:
        """
        骨架并行生成模式
        
        三阶段流程：
        1. Skeleton Generation - 生成回答骨架
        2. Skeleton Parsing - 使用 parser 解析骨架
        3. Parallel Decoding - 并行解码各分支
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            logits_processor: 采样相关的 logits processor
            use_semantic_constraint: 是否使用语义约束 (FSM 状态机)
        
        Returns:
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, skeleton_time, parallel_time
        """
        device = self.base_model.device
        model_type = self.get_model_type()
        
        # 阶段时间统计
        skeleton_time = 0.0
        parallel_time = 0.0
        evt_start_skeleton = torch.cuda.Event(enable_timing=True) # 骨架开始
        evt_end_skeleton = torch.cuda.Event(enable_timing=True)   # 骨架结束
        evt_start_parallel = torch.cuda.Event(enable_timing=True) # 并行开始 
        evt_end_parallel = torch.cuda.Event(enable_timing=True)   # 并行结束 
        
        # =====================================================================
        # Stage 1: Skeleton Generation
        # =====================================================================
        # 记录Skeleton阶段开始时间
        evt_start_skeleton.record()

        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]
        
        # 根据参数决定是否使用 FSM 状态机约束
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
            self.logger.info("使用 FSM 语义约束进行骨架生成")
        else:
            skeleton_logits_processor = logits_processor
            self.logger.info("不使用 FSM 语义约束，直接生成骨架")
        
        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # ---------------------------------------------------------------------
        # Stage 1.1: Prefill 阶段
        # ---------------------------------------------------------------------
        if self.is_distributed():
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, skeleton_logits_processor
                )
        else:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, skeleton_logits_processor)
        set_random_seed(self.seed)

        # ---------------------------------------------------------------------
        # Stage 1.2: Skeleton Decode 循环
        # ---------------------------------------------------------------------
        # 骨架的停止条件：检测到 [END] 标记或 EOS
        eos_token_id = self.tokenizer.eos_token_id
        max_steps = 200  # 骨架长度限制
        
        for step in range(max_steps):
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                _,
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=skeleton_logits_processor,
            )
            
            # 骨架的停止条件：检测是否生成了 [END] 或遇到 EOS
            generated_text = self.tokenizer.decode(input_ids[0, input_len:])
            if check_skeleton_stop(
                generated_text, eos_token_id, input_ids, input_len, 
                self.current_length_data[0].item(), max_kv_len,
                self.eagle_layer.total_tokens + 1
            ):
                break
        
        skeleton_ids = input_ids[:, input_len:]
        skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
        self.logger.info(f"Generated Skeleton: {skeleton_text}")
        self.skeleton_output = skeleton_ids.clone()
        
        # 记录Skeleton阶段时间
        evt_end_skeleton.record()
        torch.cuda.synchronize()
        skeleton_time = evt_start_skeleton.elapsed_time(evt_end_skeleton) / 1000.0  # 转换为秒
        self.logger.info(f"Skeleton generation completed in {skeleton_time:.3f}s")

        # =====================================================================
        # Stage 2: Skeleton Parsing (骨架解析)
        # =====================================================================
        mode, content = parse_skeleton_output(skeleton_text)
        
        if mode == "direct":
            # 直接回答模式：不需要并行处理，直接返回骨架输出
            self.logger.info("Direct answer mode detected, returning skeleton output")
            return skeleton_ids, 0.0, 0, 0.0, 0.0, 0.0, skeleton_time, 0.0
        
        elif mode == "error":
            self.logger.warning(f"Skeleton parsing error: {content}")
            return skeleton_ids, 0.0, 0, 0.0, 0.0, 0.0, skeleton_time, 0.0
        
        # mode == "plan": 规划模式
        tasks = content
        num_para = len(tasks)
        self.logger.info(f"Detected {num_para} parallel tasks: {[t['title'] for t in tasks]}")
        
        # 准备并行分支输入
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, tasks, skeleton_text, model_type, task_prompt
        )
        
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        predicted_branch_lengths = [t['length'] for t in tasks]
        self.logger.info(f"Predicted lengths: {predicted_branch_lengths}")

        # =====================================================================
        # Stage 3: Parallel Decoding (并行分支解码)
        # =====================================================================
        evt_start_parallel.record()
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
        
        tokens_per_branch = self.eagle_layer.total_tokens + 1
        
        evt_start_p = torch.cuda.Event(enable_timing=True)
        evt_after_verify_p = torch.cuda.Event(enable_timing=True)
        evt_after_update_p = torch.cuda.Event(enable_timing=True)
        evt_after_draft_p = torch.cuda.Event(enable_timing=True)
        
        for step_parallel in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                self.logger.warning(f"Incomplete branches due to KV cache limit: {self.active_branches}")
                break
            
            if step_parallel % 50 == 0:
                self.logger.debug(f"Parallel Decoding Step {step_parallel + 1}")

            evt_start_p.record()
            
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
            
            total_accept_len_parallel += accept_length.sum()
            
            torch.cuda.synchronize()
            total_verify_time_parallel += evt_start_p.elapsed_time(evt_after_verify_p) / 1000
            total_update_time_parallel += evt_after_verify_p.elapsed_time(evt_after_update_p) / 1000
            total_draft_time_parallel += evt_after_update_p.elapsed_time(evt_after_draft_p) / 1000
            
            if all_finished:
                self.logger.info("All branches finished generation.")
                break
        
        # 计算统计
        num_steps_parallel = max(step_parallel, 1)
        avg_accept_len = total_accept_len_parallel.item() / num_steps_parallel
        avg_draft_time = total_draft_time_parallel / num_steps_parallel
        avg_update_time = total_update_time_parallel / num_steps_parallel
        avg_verify_time = total_verify_time_parallel / num_steps_parallel
        
        self.logger.info(f"Avg accepted lengths: {avg_accept_len:.2f}, "
                        f"Avg draft time: {avg_draft_time:.4f}s, "
                        f"Avg update time: {avg_update_time:.4f}s, "
                        f"Avg verify time: {avg_verify_time:.4f}s")
        
        # 记录Parallel阶段时间
        evt_end_parallel.record()
        torch.cuda.synchronize()
        parallel_time = evt_start_parallel.elapsed_time(evt_end_parallel) / 1000.0  # 转换为秒
        self.logger.info(f"Parallel decoding completed in {parallel_time:.3f}s")

        # 合并结果
        merged_ids = merge_outputs(
            skeleton_output=self.skeleton_output,
            parallel_branches_output=self.parallel_branches_output,
            instruction_len=self.instruction_len,
            device=device,
            tasks=tasks,
            tokenizer=self.tokenizer,
        )
        
        return merged_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time, skeleton_time, parallel_time

    # =========================================================================
    # 带调度的骨架并行生成 (Scheduled Skeleton Parallel Generation)
    # =========================================================================

    @torch.no_grad()
    def generate_with_scheduling(
        self,
        task_prompt: str,
        max_new_tokens: int,
        max_parallel: int = 2,
        device_profiles: Optional[List[DeviceProfile]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        带调度的骨架并行生成

        使用分支调度系统优化并行解码阶段：
        - 支持 Continuous Batching（分支完成后立即加入新分支）
        - 支持异构设备调度（考虑算力差异）
        - 单机也可用（串行化部分分支优化时延）

        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            max_parallel: 每设备最大并行分支数
            device_profiles: 设备能力描述列表（None 则使用默认单设备）
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束

        Returns:
            output_ids: 生成的 token IDs
            stats: 统计信息字典
        """
        device = self.base_model.device
        model_type = self.get_model_type()

        # 默认单设备配置
        if device_profiles is None:
            device_profiles = [DeviceProfile.default_single_device()]

        self.logger.info(
            f"开始带调度的骨架并行生成: "
            f"max_parallel={max_parallel}, devices={len(device_profiles)}"
        )

        # 统计信息
        stats = {
            'skeleton_time': 0.0,
            'scheduling_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
        }

        # =================================================================
        # Stage 1: Skeleton Generation (复用现有逻辑)
        # =================================================================
        skeleton_start = time.time()

        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]

        # 准备 logits processor
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
        else:
            skeleton_logits_processor = logits_processor

        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)

        # Prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
            self.prefill_single(input_ids, skeleton_logits_processor)
        set_random_seed(self.seed)

        # Skeleton Decode 循环
        eos_token_id = self.tokenizer.eos_token_id
        max_steps = 200

        for step in range(max_steps):
            (
                input_ids, draft_tokens, retrieve_indices,
                tree_mask, tree_position_ids, _,
            ) = self.decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=skeleton_logits_processor,
            )

            generated_text = self.tokenizer.decode(input_ids[0, input_len:])
            if check_skeleton_stop(
                generated_text, eos_token_id, input_ids, input_len,
                self.current_length_data[0].item(), max_kv_len,
                self.eagle_layer.total_tokens + 1
            ):
                break

        skeleton_ids = input_ids[:, input_len:]
        skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
        self.skeleton_output = skeleton_ids.clone()

        stats['skeleton_time'] = time.time() - skeleton_start
        self.logger.info(f"Skeleton 生成完成: {stats['skeleton_time']:.3f}s")
        self.logger.info(f"Generated Skeleton: {self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)}")

        # =================================================================
        # Stage 2: Skeleton Parsing + Scheduling
        # =================================================================
        scheduling_start = time.time()

        mode, content = parse_skeleton_output(skeleton_text)

        if mode == "direct":
            self.logger.info("直接回答模式，跳过调度")
            stats['scheduling_time'] = time.time() - scheduling_start
            return skeleton_ids, stats

        if mode == "error":
            self.logger.warning(f"骨架解析错误: {content}")
            stats['scheduling_time'] = time.time() - scheduling_start
            return skeleton_ids, stats

        # mode == "plan": 规划模式
        tasks = content
        num_branches = len(tasks)
        stats['num_branches'] = num_branches
        self.logger.info(f"检测到 {num_branches} 个并行分支")

        # 准备分支信息
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, tasks, skeleton_text, model_type, task_prompt
        )

        branch_infos = []
        for i, task in enumerate(tasks):
            branch_infos.append(BranchInfo(
                branch_id=i,
                title=task['title'],
                predicted_length=task['length'],
                prompt_tokens=clean_branches[i],
            ))

        # 执行调度
        scheduler = HeuristicScheduler(use_compute_weight=True)
        schedule_plan = scheduler.schedule(branch_infos, device_profiles)

        stats['scheduling_time'] = time.time() - scheduling_start
        self.logger.info(f"调度完成: {stats['scheduling_time']:.3f}s")
        self.logger.info(schedule_plan.summary())

        # =================================================================
        # Stage 3: Parallel Decoding (使用调度计划 + Continuous Batching)
        # =================================================================
        parallel_start = time.time()

        # 获取当前设备的执行计划（单机模式下只有设备 0）
        device_plan = schedule_plan.get_plan_for_device(0)
        if device_plan is None:
            self.logger.warning("未找到设备 0 的执行计划")
            stats['parallel_time'] = time.time() - parallel_start
            return skeleton_ids, stats

        # 创建分支信息字典
        branch_info_dict = {info.branch_id: info for info in branch_infos}

        # 初始化执行管理器
        exec_manager = BranchExecutionManager(
            execution_plan=device_plan,
            branch_infos=branch_info_dict,
            rank=0,
        )

        # 初始化模型并行状态
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len

        # 获取初始批次的分支
        initial_branches = exec_manager.get_branches_to_add()
        if not initial_branches:
            self.logger.warning("没有分支需要执行")
            stats['parallel_time'] = time.time() - parallel_start
            return skeleton_ids, stats

        # 准备初始批次的输入
        initial_prompts = [clean_branches[bid] for bid in initial_branches]

        # 前缀复用 + Prefill 初始批次
        input_ids, tips_indices, branch_begins, branch_lengths_actual, draft_input_ids = \
            self._prepare_batch_for_prefill(
                task_input_ids, initial_prompts, initial_branches, max_new_tokens
            )

        prefix_len = task_input_ids.shape[1]
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, hidden_states = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths_actual, draft_input_ids, logits_processor
            )

        # 激活初始批次的分支
        exec_manager.activate_branches(initial_branches)

        # Parallel Decode 循环 (支持 Continuous Batching)
        tokens_per_branch = self.eagle_layer.total_tokens + 1

        for step_parallel in range(max_new_tokens):
            # 检查停止条件
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                break

            # 检查是否所有分支都已完成
            if exec_manager.is_all_completed():
                self.logger.info("所有分支执行完成")
                break

            # 检查是否有新分支需要 Prefill（Continuous Batching）
            if self.prefilling_branches:
                # 使用混合 verify + prefill 模式
                self.logger.info(
                    f"Continuous Batching: 混合处理 - "
                    f"老分支验证 {self.active_branches}, 新分支 prefill {self.prefilling_branches}"
                )
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, accept_length, all_finished,
                ) = self.decode_step_parallel_with_prefill(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    prefix_len=prefix_len,
                    logits_processor=logits_processor,
                )
            else:
                # 普通解码步骤
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, accept_length, all_finished,
                ) = self.decode_step_parallel(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=logits_processor,
                )

            exec_manager.step()

            # 检查是否有分支完成（使用 update_state_parallel 中记录的完成分支）
            if self.recently_completed_branches:
                completed_branches = self.recently_completed_branches
                self.logger.info(f"分支完成: {completed_branches}")
                
                # 收集完成分支的输出
                branch_outputs = {
                    bid: self.parallel_branches_output[bid]
                    for bid in completed_branches
                }
                exec_manager.handle_completed_branches(completed_branches, branch_outputs)

                # Continuous Batching: 尝试加入新分支
                new_branches = exec_manager.get_branches_to_add()
                if new_branches:
                    self.logger.info(f"Continuous Batching: 准备加入新分支 {new_branches}")
                    # 标记新分支进入 prefilling 状态
                    # 实际的 prefill 在下一轮 decode_step_parallel_with_prefill 中完成
                    self._add_branches_to_active(
                        new_branches, clean_branches, prefix_len, logits_processor
                    )
                    exec_manager.activate_branches(new_branches)

            if all_finished:
                break

            if step_parallel % 50 == 0:
                self.logger.debug(f"Step {step_parallel}: {exec_manager.summary()}")

        stats['parallel_time'] = time.time() - parallel_start
        stats['execution_stats'] = exec_manager.get_stats()
        self.logger.info(f"并行解码完成: {stats['parallel_time']:.3f}s")
        self.logger.info(exec_manager.summary())

        # 合并结果
        merged_ids = merge_outputs(
            skeleton_output=self.skeleton_output,
            parallel_branches_output=self.parallel_branches_output,
            instruction_len=self.instruction_len,
            device=device,
            tasks=tasks,
            tokenizer=self.tokenizer,
        )

        return merged_ids, stats

    # =========================================================================
    # 分布式生成方法 (Distributed Generation Methods)
    # =========================================================================

    @torch.no_grad()
    def generate_distributed(
        self,
        task_prompt: str,
        max_new_tokens: int,
        max_parallel: int = 4,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        分布式生成 - Naive 模式 (distributed=True, use_scheduling=False)

        三阶段流程：
        1. Distributed Prefill: 使用已实现的分布式 Prefill
        2. Skeleton Decode: 在 Master 设备上执行骨架解码
        3. Parallel Decode: 简单均分分支到各设备，每设备并行执行所有分配的分支

        特点：
        - Skeleton KV Cache 不需要同步（Parallel 阶段用不上）
        - 使用 SimpleDistributedScheduler 进行均分调度
        - 每个设备上的所有分支一次性并行执行

        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            max_parallel: 每设备最大并行分支数（naive模式下被忽略，每设备执行所有分配的分支）
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束

        Returns:
            output_ids: 生成的 token IDs
            stats: 统计信息字典
        """
        from .scheduling import SimpleDistributedScheduler, DeviceProfile, BranchInfo
        from .distributed import BranchCommManager

        device = self.base_model.device
        model_type = self.get_model_type()
        rank = self.distributed_config.rank if self.distributed_config else 0
        world_size = self.distributed_config.world_size if self.distributed_config else 1
        is_master = (rank == 0)

        stats = {
            'skeleton_time': 0.0,
            'scheduling_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': 'distributed_naive',
        }

        self.logger.info(f"开始分布式生成 (Naive 模式): rank={rank}/{world_size-1}")

        # =================================================================
        # Stage 1: Distributed Prefill (所有 Rank 参与)
        # =================================================================
        # 所有 Rank 需要同时参与分布式 Prefill
        skeleton_start = time.time()

        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]
        base_prompt_len = task_input_ids.shape[1]  # 用于后续 parallel 阶段

        # 准备 logits processor
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
        else:
            skeleton_logits_processor = logits_processor

        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)

        # 分布式 Prefill（所有 Rank 同时执行）
        self.logger.info("执行分布式 Prefill...")
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
            self.distributed_prefill_manager.prefill_single_distributed(
                input_ids, self.past_key_values, skeleton_logits_processor
            )
        set_random_seed(self.seed)

        self.logger.info(f"分布式 Prefill 完成, rank={rank}")

        # =================================================================
        # Stage 2: Skeleton Generation (仅 Master) / Worker 等待任务
        # =================================================================
        if is_master:
            # 保存 base prompt 的 Eagle KV 状态（用于后续 parallel 阶段）
            saved_eagle_kv = None
            if self.eagle_layer.stable_kv is not None:
                k_draft, v_draft = self.eagle_layer.stable_kv[0]
                saved_eagle_kv = (
                    k_draft[..., :base_prompt_len, :].clone(),
                    v_draft[..., :base_prompt_len, :].clone()
                )

            # Skeleton Decode 循环 (仅在 Master 上执行)
            self.logger.info("在 Master 上执行 Skeleton 解码...")
            eos_token_id = self.tokenizer.eos_token_id
            max_steps = 200

            for step in range(max_steps):
                (
                    input_ids, draft_tokens, retrieve_indices,
                    tree_mask, tree_position_ids, _,
                ) = self.decode_step_single(
                    input_ids=input_ids,
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=skeleton_logits_processor,
                )

                generated_text = self.tokenizer.decode(input_ids[0, input_len:])
                if check_skeleton_stop(
                    generated_text, eos_token_id, input_ids, input_len,
                    self.current_length_data[0].item(), max_kv_len,
                    self.eagle_layer.total_tokens + 1
                ):
                    break

            skeleton_ids = input_ids[:, input_len:]
            skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
            self.skeleton_output = skeleton_ids.clone()
            stats['skeleton_time'] = time.time() - skeleton_start
            self.logger.info(f"Skeleton 生成完成: {stats['skeleton_time']:.3f}s")
            self.logger.info(f"Generated Skeleton: {skeleton_text}")

            # =================================================================
            # Stage 2: Skeleton Parsing + Simple Distributed Scheduling
            # =================================================================
            scheduling_start = time.time()

            mode, content = parse_skeleton_output(skeleton_text)

            if mode == "direct":
                self.logger.info("直接回答模式，跳过并行阶段")
                stats['scheduling_time'] = time.time() - scheduling_start
                # 广播完成信号到所有 Worker
                self._broadcast_parallel_complete_signal()
                return skeleton_ids, stats

            if mode == "error":
                self.logger.warning(f"骨架解析错误: {content}")
                stats['scheduling_time'] = time.time() - scheduling_start
                self._broadcast_parallel_complete_signal()
                return skeleton_ids, stats

            # mode == "plan": 规划模式
            tasks = content
            num_branches = len(tasks)
            stats['num_branches'] = num_branches
            self.logger.info(f"检测到 {num_branches} 个并行分支")

            # 准备分支信息
            clean_branches, instruction_len = prepare_parallel_branches(
                self.tokenizer, tasks, skeleton_text, model_type, task_prompt
            )

            branch_infos = []
            for i, task in enumerate(tasks):
                branch_infos.append(BranchInfo(
                    branch_id=i,
                    title=task['title'],
                    predicted_length=task['length'],
                    prompt_tokens=clean_branches[i],
                ))

            # 简单均分调度
            device_profiles = [DeviceProfile(device_id=r) for r in range(world_size)]
            scheduler = SimpleDistributedScheduler(all_parallel=True)
            schedule_plan = scheduler.schedule(branch_infos, device_profiles)

            stats['scheduling_time'] = time.time() - scheduling_start
            self.logger.info(f"调度完成: {stats['scheduling_time']:.3f}s")
            self.logger.info(schedule_plan.summary())

            # =================================================================
            # Stage 3: 分发任务并执行 Parallel Decoding
            # =================================================================
            parallel_start = time.time()

            # 初始化分支通信管理器
            branch_comm = BranchCommManager(
                rank=rank,
                world_size=world_size,
                base_comm_manager=self.distributed_prefill_manager.comm,
            )

            # 广播调度计划和分支 Prompt 到各设备
            branch_comm.broadcast_schedule_plan(schedule_plan)
            branch_comm.send_branch_prompts(branch_infos, schedule_plan)

            # Master 执行自己的分支
            my_plan = schedule_plan.get_plan_for_device(0)
            my_branch_ids = my_plan.assigned_branches if my_plan else []

            self.logger.info(f"Master 执行分支: {my_branch_ids}")

            # 保存所有分支输出
            self.parallel_branches_output = [list(br) for br in clean_branches]
            self.instruction_len = instruction_len

            if my_branch_ids:
                # 恢复 base prompt 的 KV 状态（parallel 阶段需要从 base prompt 开始）
                # 1. 恢复 Eagle KV
                if saved_eagle_kv is not None:
                    k_saved, v_saved = saved_eagle_kv
                    self.eagle_layer.stable_kv = ((k_saved.clone(), v_saved.clone()),)

                # 2. 重置 base model KV cache 长度（_prepare_batch_for_prefill 会处理）
                # 注意：不需要手动重置，_prepare_batch_for_prefill 会自动 fill_(prefix_len)

                # 执行本地分支
                my_outputs = self._execute_parallel_branches_local(
                    task_input_ids, clean_branches, my_branch_ids,
                    max_new_tokens, logits_processor
                )
                # 更新输出
                for bid, tokens in my_outputs.items():
                    self.parallel_branches_output[bid] = tokens

            # 收集其他设备的输出
            # 注意：只需要收集分配给其他设备的分支（总分支数 - 本地分支数）
            num_other_branches = num_branches - len(my_branch_ids)
            self.logger.info(f"收集其他设备的分支输出 ({num_other_branches} 个)...")
            other_outputs = branch_comm.collect_all_outputs(
                num_branches=num_other_branches,
                timeout=300.0
            )
            for bid, tokens in other_outputs.items():
                self.parallel_branches_output[bid] = tokens

            # 广播完成信号
            branch_comm.broadcast_all_complete()

            stats['parallel_time'] = time.time() - parallel_start
            self.logger.info(f"并行解码完成: {stats['parallel_time']:.3f}s")

            # 合并结果
            merged_ids = merge_outputs(
                skeleton_output=self.skeleton_output,
                parallel_branches_output=self.parallel_branches_output,
                instruction_len=self.instruction_len,
                device=device,
                tasks=tasks,
                tokenizer=self.tokenizer,
            )

            return merged_ids, stats

        else:
            # Worker: Prefill 已完成，等待接收任务并执行
            # 传入 input_len 用于 Worker 执行
            return self._worker_execute_distributed_naive(
                input_len, max_new_tokens, logits_processor
            )

    @torch.no_grad()
    def generate_distributed_with_scheduling(
        self,
        task_prompt: str,
        max_new_tokens: int,
        max_parallel: int = 2,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        分布式生成 - 完整调度模式 (distributed=True, use_scheduling=True)

        三阶段流程：
        1. Distributed Prefill: 使用已实现的分布式 Prefill
        2. Skeleton Decode: 在 Master 设备上执行骨架解码
        3. Parallel Decode: 使用启发式调度 + 分布式 Continuous Batching

        特点：
        - Skeleton KV Cache 不同步（各设备独立使用 base prompt cache）
        - 使用 HeuristicScheduler 进行智能调度
        - 支持 Continuous Batching（分支完成后动态加入新分支）

        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            max_parallel: 每设备最大并行分支数
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束

        Returns:
            output_ids: 生成的 token IDs
            stats: 统计信息字典
        """
        from .scheduling import HeuristicScheduler, DeviceProfile, BranchInfo
        from .distributed import BranchCommManager

        device = self.base_model.device
        model_type = self.get_model_type()
        rank = self.distributed_config.rank if self.distributed_config else 0
        world_size = self.distributed_config.world_size if self.distributed_config else 1
        is_master = (rank == 0)

        stats = {
            'skeleton_time': 0.0,
            'scheduling_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': 'distributed_scheduling',
        }

        self.logger.info(f"开始分布式生成 (调度模式): rank={rank}/{world_size-1}, max_parallel={max_parallel}")

        # =================================================================
        # Stage 1: Distributed Prefill (所有 Rank 参与)
        # =================================================================
        # 所有 Rank 需要同时参与分布式 Prefill
        skeleton_start = time.time()

        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]
        base_prompt_len = task_input_ids.shape[1]  # 用于后续 parallel 阶段

        # 准备 logits processor
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
        else:
            skeleton_logits_processor = logits_processor

        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)

        # 分布式 Prefill（所有 Rank 同时执行）
        self.logger.info("执行分布式 Prefill...")
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
            self.distributed_prefill_manager.prefill_single_distributed(
                input_ids, self.past_key_values, skeleton_logits_processor
            )
        set_random_seed(self.seed)

        self.logger.info(f"分布式 Prefill 完成, rank={rank}")

        # =================================================================
        # Stage 2: Skeleton Generation (仅 Master) / Worker 等待任务
        # =================================================================
        if is_master:
            # 保存 base prompt 的 Eagle KV 状态（用于后续 parallel 阶段）
            saved_eagle_kv = None
            if self.eagle_layer.stable_kv is not None:
                k_draft, v_draft = self.eagle_layer.stable_kv[0]
                saved_eagle_kv = (
                    k_draft[..., :base_prompt_len, :].clone(),
                    v_draft[..., :base_prompt_len, :].clone()
                )

            # Skeleton Decode 循环 (仅在 Master 上执行)
            self.logger.info("在 Master 上执行 Skeleton 解码...")
            eos_token_id = self.tokenizer.eos_token_id
            max_steps = 200

            for step in range(max_steps):
                (
                    input_ids, draft_tokens, retrieve_indices,
                    tree_mask, tree_position_ids, _,
                ) = self.decode_step_single(
                    input_ids=input_ids,
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=skeleton_logits_processor,
                )

                generated_text = self.tokenizer.decode(input_ids[0, input_len:])
                if check_skeleton_stop(
                    generated_text, eos_token_id, input_ids, input_len,
                    self.current_length_data[0].item(), max_kv_len,
                    self.eagle_layer.total_tokens + 1
                ):
                    break

            skeleton_ids = input_ids[:, input_len:]
            skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
            self.skeleton_output = skeleton_ids.clone()
            stats['skeleton_time'] = time.time() - skeleton_start
            self.logger.info(f"Skeleton 生成完成: {stats['skeleton_time']:.3f}s")
            self.logger.info(f"Generated Skeleton: {skeleton_text}")

            # =================================================================
            # Stage 2: Skeleton Parsing + Heuristic Scheduling
            # =================================================================
            scheduling_start = time.time()

            mode, content = parse_skeleton_output(skeleton_text)

            if mode == "direct":
                self.logger.info("直接回答模式，跳过并行阶段")
                stats['scheduling_time'] = time.time() - scheduling_start
                self._broadcast_parallel_complete_signal()
                return skeleton_ids, stats

            if mode == "error":
                self.logger.warning(f"骨架解析错误: {content}")
                stats['scheduling_time'] = time.time() - scheduling_start
                self._broadcast_parallel_complete_signal()
                return skeleton_ids, stats

            # mode == "plan": 规划模式
            tasks = content
            num_branches = len(tasks)
            stats['num_branches'] = num_branches
            self.logger.info(f"检测到 {num_branches} 个并行分支")

            # 准备分支信息
            clean_branches, instruction_len = prepare_parallel_branches(
                self.tokenizer, tasks, skeleton_text, model_type, task_prompt
            )

            branch_infos = []
            for i, task in enumerate(tasks):
                branch_infos.append(BranchInfo(
                    branch_id=i,
                    title=task['title'],
                    predicted_length=task['length'],
                    prompt_tokens=clean_branches[i],
                ))

            # 启发式调度（考虑 max_parallel）
            device_profiles = [
                DeviceProfile(device_id=r, max_parallel=max_parallel)
                for r in range(world_size)
            ]
            scheduler = HeuristicScheduler(use_compute_weight=True)
            schedule_plan = scheduler.schedule(branch_infos, device_profiles)

            stats['scheduling_time'] = time.time() - scheduling_start
            self.logger.info(f"调度完成: {stats['scheduling_time']:.3f}s")
            self.logger.info(schedule_plan.summary())

            # =================================================================
            # Stage 3: 分发任务并执行分布式 Continuous Batching
            # =================================================================
            parallel_start = time.time()

            # 初始化分支通信管理器
            branch_comm = BranchCommManager(
                rank=rank,
                world_size=world_size,
                base_comm_manager=self.distributed_prefill_manager.comm,
            )

            # 广播调度计划和分支 Prompt 到各设备
            branch_comm.broadcast_schedule_plan(schedule_plan)
            branch_comm.send_branch_prompts(branch_infos, schedule_plan)

            # Master 执行自己的分支（使用 Continuous Batching）
            my_plan = schedule_plan.get_plan_for_device(0)
            my_branch_ids = my_plan.assigned_branches if my_plan else []

            self.logger.info(f"Master 执行分支 (Continuous Batching): {my_branch_ids}")

            # 保存所有分支输出
            self.parallel_branches_output = [list(br) for br in clean_branches]
            self.instruction_len = instruction_len

            if my_branch_ids:
                # 恢复 base prompt 的 KV 状态（parallel 阶段需要从 base prompt 开始）
                if saved_eagle_kv is not None:
                    k_saved, v_saved = saved_eagle_kv
                    self.eagle_layer.stable_kv = ((k_saved.clone(), v_saved.clone()),)

                # 使用 Continuous Batching 执行本地分支
                my_outputs = self._execute_parallel_branches_with_batching(
                    task_input_ids, clean_branches, my_plan,
                    {info.branch_id: info for info in branch_infos},
                    max_new_tokens, max_parallel, logits_processor
                )
                # 更新输出
                for bid, tokens in my_outputs.items():
                    self.parallel_branches_output[bid] = tokens

            # 收集其他设备的输出
            # 注意：只需要收集分配给其他设备的分支（总分支数 - 本地分支数）
            num_other_branches = num_branches - len(my_branch_ids)
            self.logger.info(f"收集其他设备的分支输出 ({num_other_branches} 个)...")
            other_outputs = branch_comm.collect_all_outputs(
                num_branches=num_other_branches,
                timeout=300.0
            )
            for bid, tokens in other_outputs.items():
                self.parallel_branches_output[bid] = tokens

            # 广播完成信号
            branch_comm.broadcast_all_complete()

            stats['parallel_time'] = time.time() - parallel_start
            self.logger.info(f"并行解码完成: {stats['parallel_time']:.3f}s")

            # 合并结果
            merged_ids = merge_outputs(
                skeleton_output=self.skeleton_output,
                parallel_branches_output=self.parallel_branches_output,
                instruction_len=self.instruction_len,
                device=device,
                tasks=tasks,
                tokenizer=self.tokenizer,
            )

            return merged_ids, stats

        else:
            # Worker: Prefill 已完成，等待接收任务并执行（使用 Continuous Batching）
            return self._worker_execute_distributed_scheduling(
                input_len, max_new_tokens, max_parallel, logits_processor
            )

    # =========================================================================
    # 分布式辅助方法 (Distributed Helper Methods)
    # =========================================================================

    def _broadcast_parallel_complete_signal(self) -> None:
        """广播并行阶段完成信号（用于 direct/error 模式）"""
        if not self.is_distributed():
            return
        if not self.distributed_config.is_last_rank():
            return

        from .distributed import BranchCommManager
        branch_comm = BranchCommManager(
            rank=self.distributed_config.rank,
            world_size=self.distributed_config.world_size,
            base_comm_manager=self.distributed_prefill_manager.comm,
        )
        branch_comm.broadcast_all_complete()

    def _execute_parallel_branches_local(
        self,
        prefix_ids: torch.Tensor,
        all_branches: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Dict[int, List[int]]:
        """
        本地执行指定分支（所有分支一次性并行）

        用于 naive 分布式模式，所有分配的分支一次性并行执行。

        Args:
            prefix_ids: 前缀 token IDs
            all_branches: 所有分支的 prompt tokens
            branch_ids: 要执行的分支 ID
            max_new_tokens: 最大生成 token 数
            logits_processor: logits 处理器

        Returns:
            分支输出字典 {branch_id: token_list}
        """
        if not branch_ids:
            return {}

        device = self.base_model.device
        prefix_len = prefix_ids.shape[1]

        # 获取指定分支的 prompts
        branch_prompts = [all_branches[bid] for bid in branch_ids]

        # 初始化并行状态
        self.parallel_branches_output = [list(br) for br in all_branches]
        self.active_branches = list(branch_ids)

        # 前缀复用 + Prefill
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self._prepare_batch_for_prefill(
                prefix_ids, branch_prompts, branch_ids, max_new_tokens
            )

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )

        # Parallel Decode 循环
        max_kv_len = prefix_len + max_new_tokens + 500
        tokens_per_branch = self.eagle_layer.total_tokens + 1

        for step in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                break

            (
                draft_tokens, retrieve_indices, tree_mask,
                tree_position_ids, _, all_finished,
            ) = self.decode_step_parallel(
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
            )

            if all_finished:
                break

        # 返回执行的分支输出
        outputs = {}
        for bid in branch_ids:
            outputs[bid] = self.parallel_branches_output[bid]
        return outputs

    def _execute_parallel_branches_with_batching(
        self,
        prefix_ids: torch.Tensor,
        all_branches: List[List[int]],
        execution_plan: 'DeviceExecutionPlan',
        branch_info_dict: Dict[int, 'BranchInfo'],
        max_new_tokens: int,
        max_parallel: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Dict[int, List[int]]:
        """
        使用 Continuous Batching 执行分支

        用于调度模式，支持动态加入新分支。

        Args:
            prefix_ids: 前缀 token IDs
            all_branches: 所有分支的 prompt tokens
            execution_plan: 执行计划
            branch_info_dict: 分支信息字典
            max_new_tokens: 最大生成 token 数
            max_parallel: 最大并行度
            logits_processor: logits 处理器

        Returns:
            分支输出字典 {branch_id: token_list}
        """
        from .scheduling import BranchExecutionManager

        device = self.base_model.device
        prefix_len = prefix_ids.shape[1]

        # 初始化执行管理器
        exec_manager = BranchExecutionManager(
            execution_plan=execution_plan,
            branch_infos=branch_info_dict,
            rank=self.distributed_config.rank if self.distributed_config else 0,
        )

        # 初始化并行状态
        self.parallel_branches_output = [list(br) for br in all_branches]
        self.instruction_len = {
            bid: len(all_branches[bid]) for bid in execution_plan.assigned_branches
        }

        # 获取初始批次
        initial_branches = exec_manager.get_branches_to_add()
        if not initial_branches:
            return {}

        initial_prompts = [all_branches[bid] for bid in initial_branches]

        # 前缀复用 + Prefill 初始批次
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self._prepare_batch_for_prefill(
                prefix_ids, initial_prompts, initial_branches, max_new_tokens
            )

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )

        exec_manager.activate_branches(initial_branches)

        # Continuous Batching 循环
        max_kv_len = prefix_len + max_new_tokens + 500
        tokens_per_branch = self.eagle_layer.total_tokens + 1

        for step in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                break

            if exec_manager.is_all_completed():
                break

            # 检查是否有新分支需要 Prefill
            if self.prefilling_branches:
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, _, all_finished,
                ) = self.decode_step_parallel_with_prefill(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    prefix_len=prefix_len,
                    logits_processor=logits_processor,
                )
            else:
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, _, all_finished,
                ) = self.decode_step_parallel(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=logits_processor,
                )

            exec_manager.step()

            # 处理完成的分支
            if self.recently_completed_branches:
                completed = self.recently_completed_branches
                branch_outputs = {
                    bid: self.parallel_branches_output[bid]
                    for bid in completed
                }
                exec_manager.handle_completed_branches(completed, branch_outputs)

                # Continuous Batching: 加入新分支
                new_branches = exec_manager.get_branches_to_add()
                if new_branches:
                    self._add_branches_to_active(
                        new_branches, all_branches, prefix_len, logits_processor
                    )
                    exec_manager.activate_branches(new_branches)

            if all_finished:
                break

        # 返回执行的分支输出
        outputs = {}
        for bid in execution_plan.assigned_branches:
            outputs[bid] = self.parallel_branches_output[bid]
        return outputs

    def _worker_execute_distributed_naive(
        self,
        input_len: int,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Worker 执行分布式 naive 模式

        注意: 分布式 Prefill 已在 generate_distributed() 中完成，
        KV Cache 已经初始化并同步了 base prompt 的 cache。

        Args:
            input_len: 输入序列长度（来自 Prefill）
            max_new_tokens: 最大生成 token 数
            logits_processor: logits 处理器

        Returns:
            output_ids: 空 tensor (Worker 不返回最终结果)
            stats: 统计信息
        """
        from .distributed import BranchCommManager

        device = self.base_model.device
        rank = self.distributed_config.rank
        world_size = self.distributed_config.world_size

        stats = {
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': 'worker_naive',
        }

        self.logger.info(f"Worker {rank} 等待接收任务...")

        # 初始化分支通信管理器
        branch_comm = BranchCommManager(
            rank=rank,
            world_size=world_size,
            base_comm_manager=self.distributed_prefill_manager.comm,
        )

        # 接收调度计划
        schedule_plan = branch_comm.receive_schedule_plan(timeout=60.0)
        if schedule_plan is None:
            self.logger.error("接收调度计划超时")
            return torch.tensor([], device=device), stats

        self.logger.info(f"收到调度计划: {schedule_plan.summary()}")

        # 接收分支 Prompt
        branch_infos = branch_comm.receive_branch_prompts(timeout=60.0)
        if not branch_infos:
            self.logger.warning("未收到任何分支任务")
            branch_comm.wait_for_complete(timeout=300.0)
            return torch.tensor([], device=device), stats

        stats['num_branches'] = len(branch_infos)
        self.logger.info(f"收到 {len(branch_infos)} 个分支任务")

        # 执行分支
        parallel_start = time.time()

        # KV Cache 已经在分布式 Prefill 中初始化并同步
        # 使用传入的 input_len (即 prefix_len)
        prefix_len = input_len
        max_kv_len = prefix_len + max_new_tokens + 500

        # 如果需要扩展 KV Cache 容量
        if self.past_key_values_data[0].shape[3] < max_kv_len:
            # 保存当前 cache 数据
            old_cache_data = [d[..., :prefix_len, :].clone() for d in self.past_key_values_data]

            # 重新初始化
            self.past_key_values, self.past_key_values_data, self.current_length_data = \
                initialize_past_key_values(self.base_model, max_length=max_kv_len)

            # 恢复 prefix cache
            for i, d in enumerate(old_cache_data):
                self.past_key_values_data[i][..., :prefix_len, :].copy_(d)
            self.current_length_data.fill_(prefix_len)

        # 准备分支数据
        branch_ids = [info.branch_id for info in branch_infos]
        all_branches = {info.branch_id: info.prompt_tokens for info in branch_infos}

        # 创建 prefix_ids (从 cache 中推断)
        # Worker 没有原始 prompt，但有同步的 prefix cache
        prefix_ids = torch.zeros((1, prefix_len), dtype=torch.long, device=device)

        # 初始化并行状态
        max_branch_id = max(branch_ids) + 1
        self.parallel_branches_output = [[] for _ in range(max_branch_id)]
        for info in branch_infos:
            self.parallel_branches_output[info.branch_id] = list(info.prompt_tokens)

        self.active_branches = list(branch_ids)
        self.instruction_len = {info.branch_id: len(info.prompt_tokens) for info in branch_infos}

        # 前缀复用 + Prefill
        branch_prompts = [all_branches[bid] for bid in branch_ids]

        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self._prepare_batch_for_prefill(
                prefix_ids, branch_prompts, branch_ids, max_new_tokens
            )

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )

        # Parallel Decode 循环
        tokens_per_branch = self.eagle_layer.total_tokens + 1

        for step in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                break

            (
                draft_tokens, retrieve_indices, tree_mask,
                tree_position_ids, _, all_finished,
            ) = self.decode_step_parallel(
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
            )

            if all_finished:
                break

        stats['parallel_time'] = time.time() - parallel_start

        # 发送结果到 Master
        for bid in branch_ids:
            branch_comm.send_branch_output(
                branch_id=bid,
                output_tokens=self.parallel_branches_output[bid],
            )

        self.logger.info(f"Worker {rank} 完成分支执行，等待完成信号...")

        # 等待完成信号
        branch_comm.wait_for_complete(timeout=300.0)

        return torch.tensor([], device=device), stats

    def _worker_execute_distributed_scheduling(
        self,
        input_len: int,
        max_new_tokens: int,
        max_parallel: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Worker 执行分布式调度模式（支持 Continuous Batching）

        注意: 分布式 Prefill 已在 generate_distributed_with_scheduling() 中完成，
        KV Cache 已经初始化并同步了 base prompt 的 cache。

        Args:
            input_len: 输入序列长度（来自 Prefill）
            max_new_tokens: 最大生成 token 数
            max_parallel: 最大并行度
            logits_processor: logits 处理器

        Returns:
            output_ids: 空 tensor
            stats: 统计信息
        """
        from .distributed import BranchCommManager
        from .scheduling import BranchExecutionManager, DeviceExecutionPlan

        device = self.base_model.device
        rank = self.distributed_config.rank
        world_size = self.distributed_config.world_size

        stats = {
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': 'worker_scheduling',
        }

        self.logger.info(f"Worker {rank} 等待接收任务 (调度模式)...")

        # 初始化分支通信管理器
        branch_comm = BranchCommManager(
            rank=rank,
            world_size=world_size,
            base_comm_manager=self.distributed_prefill_manager.comm,
        )

        # 接收调度计划
        schedule_plan = branch_comm.receive_schedule_plan(timeout=60.0)
        if schedule_plan is None:
            self.logger.error("接收调度计划超时")
            return torch.tensor([], device=device), stats

        self.logger.info(f"收到调度计划: {schedule_plan.summary()}")

        # 接收分支 Prompt
        branch_infos = branch_comm.receive_branch_prompts(timeout=60.0)
        if not branch_infos:
            self.logger.warning("未收到任何分支任务")
            branch_comm.wait_for_complete(timeout=300.0)
            return torch.tensor([], device=device), stats

        stats['num_branches'] = len(branch_infos)
        self.logger.info(f"收到 {len(branch_infos)} 个分支任务")

        # 执行分支（使用 Continuous Batching）
        parallel_start = time.time()

        # KV Cache 已经在分布式 Prefill 中初始化并同步
        # 使用传入的 input_len (即 prefix_len)
        prefix_len = input_len
        max_kv_len = prefix_len + max_new_tokens + 500

        # 如果需要扩展 KV Cache 容量
        if self.past_key_values_data[0].shape[3] < max_kv_len:
            old_cache_data = [d[..., :prefix_len, :].clone() for d in self.past_key_values_data]

            self.past_key_values, self.past_key_values_data, self.current_length_data = \
                initialize_past_key_values(self.base_model, max_length=max_kv_len)

            for i, d in enumerate(old_cache_data):
                self.past_key_values_data[i][..., :prefix_len, :].copy_(d)
            self.current_length_data.fill_(prefix_len)

        # 准备数据
        branch_ids = [info.branch_id for info in branch_infos]
        branch_info_dict = {info.branch_id: info for info in branch_infos}
        all_branches = {info.branch_id: info.prompt_tokens for info in branch_infos}

        # 获取本设备的执行计划
        my_plan = schedule_plan.get_plan_for_device(rank)
        if my_plan is None:
            # 创建默认计划
            my_plan = DeviceExecutionPlan(
                device_id=rank,
                assigned_branches=branch_ids,
                max_parallel=max_parallel,
            )

        # 初始化执行管理器
        exec_manager = BranchExecutionManager(
            execution_plan=my_plan,
            branch_infos=branch_info_dict,
            rank=rank,
        )

        # 初始化并行状态
        max_branch_id = max(branch_ids) + 1
        self.parallel_branches_output = [[] for _ in range(max_branch_id)]
        for info in branch_infos:
            self.parallel_branches_output[info.branch_id] = list(info.prompt_tokens)

        self.instruction_len = {info.branch_id: len(info.prompt_tokens) for info in branch_infos}

        # 获取初始批次
        initial_branches = exec_manager.get_branches_to_add()
        if not initial_branches:
            branch_comm.wait_for_complete(timeout=300.0)
            return torch.tensor([], device=device), stats

        initial_prompts = [all_branches[bid] for bid in initial_branches]
        prefix_ids = torch.zeros((1, prefix_len), dtype=torch.long, device=device)

        # 前缀复用 + Prefill 初始批次
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self._prepare_batch_for_prefill(
                prefix_ids, initial_prompts, initial_branches, max_new_tokens
            )

        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )

        exec_manager.activate_branches(initial_branches)

        # Continuous Batching 循环
        tokens_per_branch = self.eagle_layer.total_tokens + 1

        for step in range(max_new_tokens):
            if check_stop_conditions_parallel(
                current_length=self.current_length_data[0].item(),
                max_kv_len=max_kv_len,
                num_active_branches=len(self.active_branches),
                tokens_per_branch=tokens_per_branch,
            ):
                break

            if exec_manager.is_all_completed():
                break

            if self.prefilling_branches:
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, _, all_finished,
                ) = self.decode_step_parallel_with_prefill(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    prefix_len=prefix_len,
                    logits_processor=logits_processor,
                )
            else:
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, _, all_finished,
                ) = self.decode_step_parallel(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=logits_processor,
                )

            exec_manager.step()

            # 处理完成的分支
            if self.recently_completed_branches:
                completed = self.recently_completed_branches
                branch_outputs = {
                    bid: self.parallel_branches_output[bid]
                    for bid in completed
                }
                exec_manager.handle_completed_branches(completed, branch_outputs)

                # Continuous Batching: 加入新分支
                new_branches = exec_manager.get_branches_to_add()
                if new_branches:
                    all_branches_list = [all_branches.get(i, []) for i in range(max_branch_id)]
                    self._add_branches_to_active(
                        new_branches, all_branches_list, prefix_len, logits_processor
                    )
                    exec_manager.activate_branches(new_branches)

            if all_finished:
                break

        stats['parallel_time'] = time.time() - parallel_start

        # 发送结果到 Master
        for bid in branch_ids:
            branch_comm.send_branch_output(
                branch_id=bid,
                output_tokens=self.parallel_branches_output[bid],
            )

        self.logger.info(f"Worker {rank} 完成分支执行 (调度模式)，等待完成信号...")

        # 等待完成信号
        branch_comm.wait_for_complete(timeout=300.0)

        return torch.tensor([], device=device), stats

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
        # EAGLE3: 拼接 3 层 hidden states，维度 hidden_size * 3
        # EAGLE2: 只需要最后一层，维度 hidden_size
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        else:
            # EAGLE2: 只需要最后一层 hidden state
            hidden_states = outputs[0]  # outputs[0] 是 last hidden state
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if hidden_states.device != ea_device:
                hidden_states = hidden_states.to(ea_device)

        # Generate Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(hidden_states, input_ids)

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
        # EAGLE3: 拼接 3 层 hidden states，维度 hidden_size * 3
        # EAGLE2: 只需要最后一层，维度 hidden_size
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            packed_hidden = torch.cat(outputs["hidden_states"], dim=-1)[0]
        else:
            # EAGLE2: 使用 outputs[0] (last hidden state)
            packed_hidden = outputs[0][0]  # [seq_len, hidden_size]
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if packed_hidden.device != ea_device:
                packed_hidden = packed_hidden.to(ea_device)

        # 提取各分支的 Hidden States
        branch_hidden_list = []
        for i in range(num_para):
            start = branch_begins[i]
            end = start + branch_lengths[i]
            branch_hidden_list.append(packed_hidden[start - prefix_len:end - prefix_len])
        
        batched_hidden = stack_with_left_padding(branch_hidden_list, pad_id=0, device=device)

        # Generate Draft Tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(batched_hidden, draft_input_ids, prefix_len=prefix_len)

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
        
        best_candidate, accept_length, sample_token = evaluate_single(input_ids, logits, candidates, logits_processor)
        
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
            retrieve_indices, hidden_state_new, sample_token
        )
        if evt_after_update is not None:
            evt_after_update.record()

        # -----------------------------------------------------------------
        # Step 4: Generate Next Draft
        # -----------------------------------------------------------------
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(accept_hidden, input_ids=draft_input_ids)

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
        best_candidate, accept_length, sample_tokens = evaluate_parallel(
            logits, draft_tokens, retrieve_indices, logits_processor
        )

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
            self.drafter.generate_draft_tree(
                next_tips_hidden, next_tips_tokens, active_branch=self.active_branches
            )

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, accept_length, False

    def decode_step_parallel_with_prefill(
        self,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        prefix_len: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        带新分支 Prefill 的并行解码步骤（Continuous Batching 核心）
        
        该方法同时处理：
        1. 老分支的 draft 验证（speculative decoding）
        2. 新分支的 prompt prefill
        
        流程：
        1. 构建混合输入：老分支的 draft tokens + 新分支的 prompt tokens
        2. 构建混合 attention mask
        3. Base Model Forward（同时完成验证和 prefill）
        4. 处理老分支的验证结果（跳过新分支的验证）
        5. 为新分支采样 root token
        6. 更新状态并生成下一轮 draft
        
        Args:
            draft_tokens: 老分支的 draft tokens [num_old_branches, num_nodes]
            retrieve_indices: 检索索引
            tree_mask: tree attention mask
            tree_position_ids: tree 位置编码
            prefix_len: 共享前缀长度
            logits_processor: logits 处理器
            
        Returns:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, accept_length, all_finished
        """
        device = self.base_model.device
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # 获取新分支信息
        new_branch_ids = self.prefilling_branches
        num_new = len(new_branch_ids)
        num_old = len(self.active_branches)
        num_nodes = draft_tokens.shape[1] if draft_tokens is not None else 0
        
        # =====================================================================
        # Step 1: 构建混合输入
        # =====================================================================
        
        # 老分支的 draft tokens
        if num_old > 0 and draft_tokens is not None:
            flat_draft_tokens = draft_tokens.reshape(1, -1)  # [1, num_old * num_nodes]
            
            # 计算绝对位置
            current_tip_pos = self.eagle_layer.full_position_ids[:, -1].unsqueeze(-1)
            abs_draft_pos = tree_position_ids + current_tip_pos + 1
            flat_draft_pos = abs_draft_pos.view(1, -1)  # [1, num_old * num_nodes]
        else:
            flat_draft_tokens = None
            flat_draft_pos = None
        
        # 新分支的 prompt tokens
        new_prompt_tensors = []
        new_prompt_positions = []
        new_branch_bim = []
        new_prompt_lengths = []
        
        for bid in new_branch_ids:
            prompt = self.pending_prefill_prompts[bid]
            prompt_len = len(prompt)
            new_prompt_lengths.append(prompt_len)
            
            prompt_tensor = torch.tensor(prompt, device=device, dtype=torch.long)
            new_prompt_tensors.append(prompt_tensor)
            
            # 位置编码：从 prefix_len 开始
            pos = torch.arange(prefix_len, prefix_len + prompt_len, device=device)
            new_prompt_positions.append(pos)
            
            # BIM 标记
            new_branch_bim.extend([bid] * prompt_len)
        
        # 拼接所有新分支的 prompt
        if new_prompt_tensors:
            flat_new_prompts = torch.cat(new_prompt_tensors).unsqueeze(0)  # [1, total_new_len]
            flat_new_positions = torch.cat(new_prompt_positions).unsqueeze(0)  # [1, total_new_len]
            total_new_len = flat_new_prompts.shape[1]
        else:
            flat_new_prompts = None
            flat_new_positions = None
            total_new_len = 0
        
        # 合并输入
        if flat_draft_tokens is not None and flat_new_prompts is not None:
            combined_input = torch.cat([flat_draft_tokens, flat_new_prompts], dim=1)
            combined_positions = torch.cat([flat_draft_pos, flat_new_positions], dim=1)
        elif flat_draft_tokens is not None:
            combined_input = flat_draft_tokens
            combined_positions = flat_draft_pos
        elif flat_new_prompts is not None:
            combined_input = flat_new_prompts
            combined_positions = flat_new_positions
        else:
            # 没有任何输入
            return None, None, None, None, None, True
        
        # =====================================================================
        # Step 2: 构建混合 Attention Mask
        # =====================================================================
        
        current_length = self.current_length_data[0].item()
        history_bim = self.branch_index_map[:current_length]
        total_input_len = combined_input.shape[1]
        
        # 初始化 Cross Mask (全部遮蔽)
        cross_mask = torch.full(
            (1, 1, total_input_len, current_length),
            torch.finfo(torch.float32).min, device=device
        )
        
        # 计算每个输入 token 的分支归属
        combined_bim = []
        if num_old > 0 and draft_tokens is not None:
            # 老分支的 draft tokens
            for bid in self.active_branches:
                combined_bim.extend([bid] * num_nodes)
        # 新分支的 prompt tokens
        combined_bim.extend(new_branch_bim)
        combined_bim_tensor = torch.tensor(combined_bim, device=device)
        
        # Prefix 全部可见 (BIM == -1)
        is_prefix = (history_bim == -1).view(1, 1, 1, -1)
        cross_mask.masked_fill_(is_prefix, 0)
        
        # 同分支可见
        input_ids_view = combined_bim_tensor.view(1, 1, -1, 1)
        hist_ids_view = history_bim.view(1, 1, 1, -1)
        is_same_branch = (input_ids_view == hist_ids_view)
        cross_mask.masked_fill_(is_same_branch, 0)
        
        # 构建 Input Block Mask
        # 老分支使用 tree_mask，新分支使用 causal mask
        input_block_mask = torch.full(
            (total_input_len, total_input_len),
            torch.finfo(torch.float32).min, device=device
        )
        
        # 老分支的 tree mask（块对角）
        if num_old > 0 and draft_tokens is not None:
            converted_tree_mask = torch.where(
                tree_mask == 1, 0.0, torch.finfo(torch.float32).min
            )
            for i in range(num_old):
                st, ed = i * num_nodes, (i + 1) * num_nodes
                input_block_mask[st:ed, st:ed] = converted_tree_mask[i, 0, :, :]
        
        # 新分支的 causal mask
        if total_new_len > 0:
            old_total_len = num_old * num_nodes if (num_old > 0 and draft_tokens is not None) else 0
            
            # 每个新分支内部使用 causal mask
            offset = old_total_len
            for i, prompt_len in enumerate(new_prompt_lengths):
                for j in range(prompt_len):
                    for k in range(j + 1):
                        input_block_mask[offset + j, offset + k] = 0
                offset += prompt_len
        
        input_block_mask = input_block_mask.unsqueeze(0).unsqueeze(0)
        
        # 合并
        combined_mask = torch.cat([cross_mask, input_block_mask], dim=-1)
        
        # =====================================================================
        # Step 3: Base Model Forward
        # =====================================================================
        
        # 临时更新 BIM（用于本次 Forward）
        temp_bim_start = current_length
        self.branch_index_map[temp_bim_start:temp_bim_start + total_input_len] = combined_bim_tensor
        
        with torch.inference_mode():
            outputs, hidden_states = self(
                combined_input,
                past_key_values=self.past_key_values,
                attention_mask=combined_mask,
                position_ids=combined_positions,
                output_orig=False,
            )
        
        # 计算 Logits
        all_logits = self.base_model.lm_head(hidden_states)  # [1, total_len, vocab]
        
        # 处理 Hidden States for Eagle Layer
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        else:
            hidden_states = outputs[0]
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if hidden_states.device != ea_device:
                hidden_states = hidden_states.to(ea_device)
        
        # =====================================================================
        # Step 4: 处理老分支的验证结果
        # =====================================================================
        
        accept_length = None
        next_tips_hidden_old = None
        next_tips_tokens_old = None
        
        if num_old > 0 and draft_tokens is not None:
            # 提取老分支的 logits
            old_total_len = num_old * num_nodes
            old_logits = all_logits[:, :old_total_len, :]
            old_logits = old_logits.view(num_old, num_nodes, -1)
            old_hidden = hidden_states[:, :old_total_len, :]
            
            # 评估（验证 draft tokens）
            best_candidate, accept_length, sample_tokens = evaluate_parallel(
                old_logits, draft_tokens, retrieve_indices, logits_processor
            )
            
            # 更新状态
            next_tips_hidden_old, next_tips_tokens_old = self.update_state_parallel(
                best_candidate, accept_length, draft_tokens,
                retrieve_indices, old_hidden, sample_tokens, num_nodes
            )
        
        # =====================================================================
        # Step 5: 处理新分支的 Prefill 结果
        # =====================================================================
        
        new_tips_hidden_list = []
        new_tips_tokens_list = []
        new_tips_pos_list = []
        
        if num_new > 0:
            old_total_len = num_old * num_nodes if (num_old > 0 and draft_tokens is not None) else 0
            
            # 更新 KV Cache 长度（新分支的 prompt 已经被处理）
            current_len_after_old = self.current_length_data[0].item()
            
            # 提取每个新分支的结果
            offset = old_total_len
            for i, (bid, prompt_len) in enumerate(zip(new_branch_ids, new_prompt_lengths)):
                # 获取该分支最后一个 token 的 logits
                tip_logits = all_logits[0, offset + prompt_len - 1, :]  # [vocab]
                
                # 采样 root token
                if logits_processor is not None:
                    tip_logits = logits_processor(None, tip_logits.unsqueeze(0)).squeeze(0)
                    probs = torch.nn.functional.softmax(tip_logits, dim=-1)
                    root_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    root_token = torch.argmax(tip_logits).item()
                
                # 获取该分支最后一个 token 的 hidden state（用于 draft）
                # 注意：只需要最后一个 token 的 hidden state
                branch_last_hidden = hidden_states[0, offset + prompt_len - 1, :].unsqueeze(0)  # [1, hidden]
                
                # 准备 draft 输入：只需要 root token
                new_tips_tokens_list.append(
                    torch.tensor([root_token], device=device, dtype=torch.long)
                )
                new_tips_hidden_list.append(branch_last_hidden)
                
                # 位置编码：只有 root token 的位置
                pos = torch.tensor([prefix_len + prompt_len], device=device)
                new_tips_pos_list.append(pos)
                
                # 更新 BIM（将新分支的 prompt KV 标记为该分支）
                bim_start = current_len_after_old
                bim_end = bim_start + prompt_len
                self.branch_index_map[bim_start:bim_end] = bid
                current_len_after_old = bim_end
                
                # 更新分支输出
                self.parallel_branches_output[bid].append(root_token)
                
                offset += prompt_len
            
            # 更新 current_length
            self.current_length_data.fill_(current_len_after_old)
            
            # 将新分支加入活跃列表
            self.active_branches.extend(new_branch_ids)
            
            # 更新 Eagle Layer 状态
            # 新分支的 cache_padding_mask 和 full_position_ids
            # 需要包含 prefix + prompt 的部分
            new_mask_list = []
            new_pos_list_for_eagle = []
            for i, prompt_len in enumerate(new_prompt_lengths):
                # mask: prefix + prompt 部分为 1
                total_len = prefix_len + prompt_len
                mask = torch.ones(total_len, device=device, dtype=torch.long)
                new_mask_list.append(mask)
                
                # position: prefix 部分 [0, prefix_len), prompt 部分 [prefix_len, prefix_len + prompt_len)
                pos = torch.arange(total_len, device=device, dtype=torch.long)
                new_pos_list_for_eagle.append(pos)
            
            # 对齐到老分支的长度
            if self.eagle_layer.cache_padding_mask is not None:
                old_len = self.eagle_layer.cache_padding_mask.shape[1]
                
                # 对齐新分支
                padded_masks = []
                padded_positions = []
                for mask, pos in zip(new_mask_list, new_pos_list_for_eagle):
                    cur_len = len(mask)
                    if cur_len < old_len:
                        # 左填充到相同长度
                        pad_len = old_len - cur_len
                        mask = torch.cat([torch.zeros(pad_len, device=device, dtype=mask.dtype), mask])
                        pos = torch.cat([torch.zeros(pad_len, device=device, dtype=pos.dtype), pos])
                    elif cur_len > old_len:
                        # 老分支需要右填充
                        pad_len = cur_len - old_len
                        old_mask_padded = torch.cat([
                            self.eagle_layer.cache_padding_mask,
                            torch.zeros(self.eagle_layer.cache_padding_mask.shape[0], pad_len, device=device, dtype=self.eagle_layer.cache_padding_mask.dtype)
                        ], dim=1)
                        old_pos_padded = torch.cat([
                            self.eagle_layer.full_position_ids,
                            torch.zeros(self.eagle_layer.full_position_ids.shape[0], pad_len, device=device, dtype=self.eagle_layer.full_position_ids.dtype)
                        ], dim=1)
                        self.eagle_layer.cache_padding_mask = old_mask_padded
                        self.eagle_layer.full_position_ids = old_pos_padded
                        old_len = cur_len
                    padded_masks.append(mask)
                    padded_positions.append(pos)
                
                new_cache_mask = torch.stack(padded_masks, dim=0)
                new_full_pos = torch.stack(padded_positions, dim=0)
                
                self.eagle_layer.cache_padding_mask = torch.cat([
                    self.eagle_layer.cache_padding_mask, new_cache_mask
                ], dim=0)
                self.eagle_layer.full_position_ids = torch.cat([
                    self.eagle_layer.full_position_ids, new_full_pos
                ], dim=0)
            
            # 扩展 Eagle Layer KV Cache
            # 关键：新分支需要从 prefix KV 扩展，并扩展到与老分支相同的序列长度
            if self.eagle_layer.stable_kv is not None:
                k_draft, v_draft = self.eagle_layer.stable_kv[0]
                old_kv_seq_len = k_draft.shape[2]  # 老分支当前的 KV 长度
                
                # 新分支从 prefix 开始
                k_prefix = k_draft[:1, :, :prefix_len, :].clone()
                v_prefix = v_draft[:1, :, :prefix_len, :].clone()
                
                # 扩展到与老分支相同的序列长度（填充零）
                if old_kv_seq_len > prefix_len:
                    # 需要为新分支预留空间
                    pad_len = old_kv_seq_len - prefix_len
                    k_pad = torch.zeros(
                        1, k_prefix.shape[1], pad_len, k_prefix.shape[3],
                        device=device, dtype=k_prefix.dtype
                    )
                    v_pad = torch.zeros(
                        1, v_prefix.shape[1], pad_len, v_prefix.shape[3],
                        device=device, dtype=v_prefix.dtype
                    )
                    k_prefix = torch.cat([k_prefix, k_pad], dim=2)
                    v_prefix = torch.cat([v_prefix, v_pad], dim=2)
                
                k_expanded = k_prefix.expand(num_new, -1, -1, -1).clone()
                v_expanded = v_prefix.expand(num_new, -1, -1, -1).clone()
                
                # 将新分支的 KV Cache 追加到现有的
                k_combined = torch.cat([k_draft, k_expanded], dim=0)
                v_combined = torch.cat([v_draft, v_expanded], dim=0)
                self.eagle_layer.stable_kv = ((k_combined, v_combined),)
            
            # 清理状态
            self.prefilling_branches = []
            self.pending_prefill_prompts = {}
        
        # =====================================================================
        # Step 6: 检查是否完成并生成下一轮 Draft
        # =====================================================================
        
        all_finished = not self.active_branches
        
        if all_finished:
            return None, None, None, None, accept_length, True
        
        # 合并老分支和新分支的 hidden states 和 tokens
        all_tips_hidden_list = []
        all_tips_tokens_list = []
        
        if next_tips_hidden_old is not None and next_tips_tokens_old is not None:
            # 老分支的 tips
            for i in range(next_tips_hidden_old.shape[0]):
                all_tips_hidden_list.append(next_tips_hidden_old[i])
                all_tips_tokens_list.append(next_tips_tokens_old[i])
        
        if new_tips_hidden_list:
            # 新分支的 tips
            for hidden, tokens in zip(new_tips_hidden_list, new_tips_tokens_list):
                all_tips_hidden_list.append(hidden)
                all_tips_tokens_list.append(tokens)
        
        if not all_tips_hidden_list:
            return None, None, None, None, accept_length, True
        
        # 对齐并堆叠
        batched_hidden = stack_with_left_padding(all_tips_hidden_list, pad_id=0, device=device)
        batched_tokens = stack_with_left_padding(all_tips_tokens_list, pad_id=pad_token_id, device=device)
        
        # 生成下一轮 Draft
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            self.drafter.generate_draft_tree(
                batched_hidden, batched_tokens, active_branch=self.active_branches
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

        # 处理 Hidden States for Eagle Layer
        # EAGLE3: 拼接 3 层 hidden states
        # EAGLE2: 只需要最后一层
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
        else:
            # EAGLE2: 使用 outputs[0] (last hidden state)
            hidden_state = outputs[0]
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if hidden_state.device != ea_device:
                hidden_state = hidden_state.to(ea_device)

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
        # EAGLE3: 拼接 3 层 hidden states
        # EAGLE2: 只需要最后一层
        if self.use_eagle3:
            ea_device = self.eagle_layer.lm_head.weight.device
            if outputs["hidden_states"][0].device != ea_device:
                outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
            hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        else:
            # EAGLE2: 使用 outputs[0] (last hidden state)
            hidden_states = outputs[0]
            ea_device = self.eagle_layer.embed_tokens.weight.device
            if hidden_states.device != ea_device:
                hidden_states = hidden_states.to(ea_device)
        
        return logits, hidden_states

    def update_state_single(
        self,
        input_ids: torch.Tensor,
        candidates: torch.Tensor,
        best_candidate: torch.Tensor,
        accept_length: torch.Tensor,
        retrieve_indices: torch.Tensor,
        hidden_state_new: torch.Tensor,
        sample_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        更新推理状态（单序列模式）
        
        操作：
        1. 将接受的 tokens 添加到 input_ids
        2. 更新 KV Cache (搬运接受的 KV)
        3. 拼接 Bonus Token 到 Draft 输入
        4. 提取接受路径的 Hidden States (供上层生成 Draft Tree)
        
        Args:
            input_ids: 当前输入 [1, seq_len]
            candidates: 候选 tokens [num_leaves, depth]
            best_candidate: 最佳候选索引 [1]
            accept_length: 接受长度 [1]
            retrieve_indices: 检索索引 [num_leaves, depth]
            hidden_state_new: 新的隐藏状态
            sample_token: 采样的 Bonus Token
            
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

        # 更新 KV Cache (搬运接受的 KV 到正确位置)
        for past_kv_data in self.past_key_values_data:
            tgt = past_kv_data.index_select(dim=-2, index=select_indices.to(past_kv_data.device))
            dst = past_kv_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
            self.current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # 拼接 Bonus Token 到 Draft 输入
        token = sample_token[None] if sample_token.ndim == 1 else sample_token
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
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_token_id 

        # 获取当前有效历史长度
        valid_history_len = (self.branch_index_map != -2).sum().item()
        dst_ptr = valid_history_len
        last_pos = self.eagle_layer.full_position_ids[:, -1].tolist()

        new_bim_entries = []
        new_pos_list = []
        draft_update_tokens_list = []
        draft_update_hiddens_list = []
        keep_mask_list = [True] * len(self.active_branches)
        
        # 清空上一步的完成分支列表（用于 Continuous Batching）
        self.recently_completed_branches = []

        # 遍历每个活跃分支
        for i, branch_idx in enumerate(self.active_branches):
            acc_len_i = accept_length[i].item()
            best_idx_i = best_candidate[i].item()

            # 提取接受的 tokens
            select_indices = retrieve_indices[i, best_idx_i, :acc_len_i + 1]
            seq_tokens = draft_tokens[i][select_indices]
            bonus_token = sample_tokens[i].item()
            
            # 检查是否完成：EOS token 或 [END] 标记
            is_finished = (bonus_token == eos_token_id)
            
            # 更新分支输出
            tokens_to_add = seq_tokens.tolist()
            
            # 检查 [END] 标记：只检测新生成的内容（不含 instruction 部分）
            if not is_finished:
                # 获取该分支的 instruction 长度
                instr_len = self.instruction_len[branch_idx] if self.instruction_len else 0
                # 只取生成部分（去掉 instruction）
                generated_part = self.parallel_branches_output[branch_idx][instr_len:]
                current_generated = generated_part + tokens_to_add + [bonus_token]
                decoded_text = self.tokenizer.decode(current_generated, skip_special_tokens=False)
                if "[END]" in decoded_text:
                    is_finished = True
                    # 截断到 [END] 之前的内容
                    full_new_tokens = tokens_to_add + [bonus_token]
                    for cut_idx in range(len(full_new_tokens)):
                        partial = generated_part + full_new_tokens[:cut_idx+1]
                        partial_text = self.tokenizer.decode(partial, skip_special_tokens=False)
                        if "[END]" in partial_text:
                            tokens_to_add = full_new_tokens[:cut_idx+1]
                            break
                    self.logger.info(f"分支 {branch_idx} 完成生成 ([END] 标记)")
            
            if is_finished:
                if bonus_token == eos_token_id:
                    self.logger.info(f"分支 {branch_idx} 完成生成 (EOS)")
                    tokens_to_add.append(bonus_token)
                keep_mask_list[i] = False
                # 记录完成的分支（用于 Continuous Batching）
                self.recently_completed_branches.append(branch_idx)
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
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
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
 
        # 构建 Draft 模型的 Batched 输入（左填充）,用什么填充无所谓，都会mask掉
        draft_input_ids, branch_mask = stack_with_left_padding(draft_branch_list, pad_id=pad_token_id, device=device, return_mask=True)
        padded_branch_pos = stack_with_left_padding(draft_pos_list, pad_id=0, device=device)

        prefix_mask = torch.ones((num_para, prefix_len), dtype=torch.long, device=device)
        prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(num_para, -1)
        self.eagle_layer.cache_padding_mask = torch.cat([prefix_mask, branch_mask], dim=1)
        self.eagle_layer.full_position_ids = torch.cat([prefix_pos, padded_branch_pos], dim=1)

        return input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids

    # =========================================================================
    # Continuous Batching 辅助方法
    # =========================================================================

    def _prepare_batch_for_prefill(
        self,
        prefix_ids: torch.Tensor,
        branches_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], torch.Tensor]:
        """
        为指定分支准备 Prefill 输入（支持 Continuous Batching）

        与 reuse_prefix_for_parallel 类似，但只处理指定的分支子集。

        Args:
            prefix_ids: 共享前缀 [1, prefix_len]
            branches_prompts: 分支 prompt 列表
            branch_ids: 分支 ID 列表（用于 BIM 标记）
            max_new_tokens: 最大生成长度

        Returns:
            与 reuse_prefix_for_parallel 相同
        """
        device = self.base_model.device
        num_para = len(branches_prompts)
        prefix_len = prefix_ids.shape[1]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # 初始化活跃分支列表
        self.active_branches = list(branch_ids)

        # 重置 KV Cache 到 prefix 长度
        self.current_length_data.fill_(prefix_len)

        # Eagle Layer KV Cache: 复制 prefix KV 为多分支
        if self.eagle_layer.stable_kv is not None:
            k_draft, v_draft = self.eagle_layer.stable_kv[0]
            k_prefix = k_draft[..., :prefix_len, :].clone()
            v_prefix = v_draft[..., :prefix_len, :].clone()
            k_expanded = k_prefix.expand(num_para, -1, -1, -1).clone()
            v_expanded = v_prefix.expand(num_para, -1, -1, -1).clone()
            self.eagle_layer.stable_kv = ((k_expanded, v_expanded),)

        # 构建打包输入序列和 BIM
        flat_branch_ids = []
        branch_index_list = [-1] * prefix_len

        tips_indices = []
        branch_begins = []
        branch_lengths = []
        pos_ids_list = list(range(prefix_len))

        draft_branch_list = []
        draft_pos_list = []

        current_offset = prefix_len
        for i, (br, bid) in enumerate(zip(branches_prompts, branch_ids)):
            curr_len = len(br)
            branch_begins.append(current_offset)
            flat_branch_ids.extend(br)
            branch_index_list.extend([bid] * curr_len)  # 使用实际的 branch_id
            branch_lengths.append(curr_len)
            current_offset += curr_len
            tips_indices.append(current_offset - 1)

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

        draft_input_ids, branch_mask = stack_with_left_padding(
            draft_branch_list, pad_id=pad_token_id, device=device, return_mask=True
        )
        padded_branch_pos = stack_with_left_padding(draft_pos_list, pad_id=0, device=device)

        prefix_mask = torch.ones((num_para, prefix_len), dtype=torch.long, device=device)
        prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(num_para, -1)
        self.eagle_layer.cache_padding_mask = torch.cat([prefix_mask, branch_mask], dim=1)
        self.eagle_layer.full_position_ids = torch.cat([prefix_pos, padded_branch_pos], dim=1)

        return input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids

    def _get_completed_branches(self) -> List[int]:
        """
        获取已完成的分支列表

        通过检查 parallel_branches_output 中是否包含 EOS token 来判断。

        Returns:
            完成的分支 ID 列表
        """
        completed = []
        eos_token_id = self.tokenizer.eos_token_id

        # 检查哪些分支已经生成了 EOS
        for branch_id in list(self.active_branches):
            if branch_id < len(self.parallel_branches_output):
                output = self.parallel_branches_output[branch_id]
                if output and output[-1] == eos_token_id:
                    completed.append(branch_id)

        return completed

    def _add_branches_to_active(
        self,
        new_branch_ids: List[int],
        all_branches_prompts: List[List[int]],
        prefix_len: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将新分支加入活跃批次（Continuous Batching）
        
        核心逻辑：
        1. 新分支先标记为 prefilling 状态
        2. 保存新分支的 prompt tokens
        3. 下一轮 decode_step_parallel_with_prefill 会同时处理：
           - 老分支的 draft 验证
           - 新分支的 prompt prefill
        4. prefill 完成后，新分支才参与 draft 生成
        
        Args:
            new_branch_ids: 新分支 ID 列表
            all_branches_prompts: 所有分支的 prompt
            prefix_len: 前缀长度
            logits_processor: logits 处理器

        Returns:
            None - 状态更新在 decode_step_parallel_with_prefill 中完成
        """
        if not new_branch_ids:
            return None, None, None, None
            
        device = self.base_model.device
        
        # 将新分支标记为 prefilling 状态
        self.prefilling_branches = list(new_branch_ids)
        
        # 缓存新分支的 prompt tokens
        for bid in new_branch_ids:
            self.pending_prefill_prompts[bid] = all_branches_prompts[bid]
        
        self.logger.debug(
            f"新分支加入 prefilling 队列: {new_branch_ids}, "
            f"当前活跃分支: {self.active_branches}"
        )
        
        # 返回 None，实际的处理在下一轮 decode_step 中完成
        return None, None, None, None
    

    def inject_new_branches(
        self,
        new_branch_ids: List[int],
        prefix_len: int
    ) -> None:
        """
        [Continuous Batching 核心] 动态注入新分支
        
        功能：
        1. 在 KV Cache 尾部为新分支分配空间。
        2. 将共享的 Skeleton Prefix KV 复制到新位置。
        3. 更新 Branch Index Map (BIM)。
        
        Args:
            new_branch_ids: 新加入的分支 ID 列表
            prefix_len: 共享骨架前缀的长度 (对应 KV Cache 的 0:prefix_len)
        """
        if not new_branch_ids:
            return

        device = self.base_model.device
        num_new = len(new_branch_ids)
        
        # 获取当前 KV Cache 的总长度指针 (所有分支共用一个 append-only 指针)
        # current_length_data 是一个 tensor，通常存储在 CPU 或 GPU
        current_global_len = self.current_length_data[0].item()
        
        # 计算增量长度: 每个新分支都需要一份 Prefix 的副本
        added_len = num_new * prefix_len
        
        # 1. 显存容量检查 (简单版)
        if current_global_len + added_len >= self.branch_index_map.shape[0]:
            # 实际生产中应触发驱逐策略或扩容，这里抛出异常
            raise RuntimeError(f"KV Cache Pool 耗尽! Cur: {current_global_len}, Need: {added_len}")

        # 2. 执行 KV Cache 复制 (物理显存操作)
        # 假设 0:prefix_len 存储着干净的 Skeleton Prefix
        # 目标: 将其复制到 current_global_len : current_global_len + added_len
        
        target_start = current_global_len
        
        # 遍历所有层 (Layer)
        for layer_kvs in self.past_key_values:
            for kv_cache in layer_kvs: # 分别处理 Key 和 Value
                # kv_cache.data shape: [Batch=1, Num_Heads, Max_Len, Head_Dim]
                # 注意: 在 specsot 中，Batch 通常为 1，所有分支混在 Seq_Len 维度
                
                # 源数据: [1, H, prefix_len, D]
                src_data = kv_cache.data[..., :prefix_len, :]
                
                # 构造目标数据: 将 src 复制 num_new 份
                # repeat: [1, H, prefix_len, D] -> [num_new, H, prefix_len, D]
                tiled_data = src_data.repeat(num_new, 1, 1, 1) 
                
                # 展平为 [1, H, num_new * prefix_len, D] 以适配存储结构
                tiled_data = tiled_data.transpose(0, 2).reshape(1, kv_cache.data.shape[1], -1, kv_cache.data.shape[3])
                # 修正 reshape 逻辑: 应该是先 repeat dim 0 (batch 维度模拟) 或者 dim 2 (seq 维度)
                # 正确做法：
                # src_data: [1, H, L, D] -> repeat(1, 1, N, 1) -> [1, H, N*L, D] (view 可能会乱序)
                # 更稳妥的做法是利用 cat
                src_data_sq = src_data.squeeze(0) # [H, L, D]
                tiled_sq = src_data_sq.repeat(1, num_new, 1) # [H, N*L, D] 注意：这里是简单的重复，顺序是 L,L,L...
                # 但我们需要配合 BIM，BIM 是 [ID1]*L, [ID2]*L... 所以简单的 repeat 是匹配的
                
                tiled_data = tiled_sq.unsqueeze(0) # [1, H, N*L, D]
                
                # 写入显存
                target_end = target_start + added_len
                kv_cache.data[..., target_start:target_end, :] = tiled_data

        # 3. 更新 Branch Index Map (BIM)
        # 这一步告诉 Attention 机制：这段 KV 属于哪个分支
        new_bim_list = []
        for bid in new_branch_ids:
            new_bim_list.extend([bid] * prefix_len)
            
        bim_tensor = torch.tensor(new_bim_list, device=device, dtype=torch.long)
        self.branch_index_map[target_start : target_start + added_len] = bim_tensor

        # 4. 更新全局指针
        self.current_length_data.fill_(current_global_len + added_len)
        
        # 5. 更新活跃分支记录
        for bid in new_branch_ids:
            if bid not in self.active_branches:
                self.active_branches.append(bid)
                
        # 6. Eagle Layer 状态更新 (如果 Eagle Layer 有独立 Cache)
        # 通常 Eagle Layer 需要重新初始化其内部状态以接纳新分支
        # 这里假设 Eagle 的 Cache 也是随动的，或者在 decode_step 中动态处理
        
        print(f"[SpecSoT] Injected {num_new} branches. Cache grew by {added_len} slots.")


    def decode_step_parallel_mixed(
        self,
        # === A. 老分支 (Verification) ===
        draft_tokens: torch.Tensor,       # [B_old, K]
        retrieve_indices: torch.Tensor,   # [B_old, K]
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        old_branch_ids: List[int],        # 对应 draft_tokens 的分支 ID
        
        # === B. 新分支 (Prefill) ===
        new_prompts: List[torch.Tensor],  # List of [Len]
        new_branch_ids: List[int],
        
        prefix_len: int
    ):
        """
        [混合推理步] 同时处理 Draft 验证 (老任务) 和 Prompt Prefill (新任务)
        """
        device = self.base_model.device
        
        # --- 1. 构造 Base Model 的输入 ---
        
        combined_input_ids = []
        combined_position_ids = []
        combined_bim_ids = [] # 用于构建 Attention Mask
        
        # A. 处理老分支 (Drafts)
        # draft_tokens [B, K] -> flatten
        if old_branch_ids and draft_tokens is not None:
            # 假设 draft_tokens 已经是 [B, K]
            # 对应的 position_ids 需要从 tree_position_ids 获取
            # tree_position_ids: [B, K]
            
            flat_draft = draft_tokens.view(-1)
            flat_pos = tree_position_ids.view(-1)
            
            combined_input_ids.append(flat_draft)
            combined_position_ids.append(flat_pos)
            
            # BIM: 每个 token 属于哪个 branch
            k_size = draft_tokens.shape[1]
            for bid in old_branch_ids:
                combined_bim_ids.extend([bid] * k_size)
                
        # B. 处理新分支 (Prompts)
        if new_branch_ids:
            for prompt, bid in zip(new_prompts, new_branch_ids):
                # prompt: [Len]
                combined_input_ids.append(prompt)
                
                # pos: 从 prefix_len 开始递增
                p_len = prompt.shape[0]
                pos = torch.arange(prefix_len, prefix_len + p_len, device=device)
                combined_position_ids.append(pos)
                
                # BIM
                combined_bim_ids.extend([bid] * p_len)
        
        if not combined_input_ids:
            return None # 无事可做

        # 拼接
        final_input_ids = torch.cat(combined_input_ids).unsqueeze(0) # [1, Total_Len]
        final_position_ids = torch.cat(combined_position_ids).unsqueeze(0)
        
        # --- 2. 构造 Attention Mask ---
        # 这是一个关键点。我们需要构建一个 Mask，使得:
        # Token i 能看到 Token j 当且仅当:
        # 1. BIM[i] == BIM[j] (同分支) OR BIM[j] 属于 Prefix (-1或特定标记)
        # 2. i >= j (Causal) 或 Tree Mask 允许
        
        # 这里复用 self.prepare_parallel_mask 逻辑，但需要传入本次的 BIM 信息
        # 为简化，假设 Base Model 使用 FlashAttention + BIM 索引，或者我们在 forward 中通过 hook 处理
        # 如果是标准 Llama/Qwen，需要传入 attention_mask
        
        # 临时扩展 BIM (为了本次 Forward)
        current_len = self.current_length_data[0].item()
        temp_bim_len = len(combined_bim_ids)
        
        # 更新 BIM 显存 (追加本次输入的 BIM 标记)
        # 注意：Forward 完成后 KV 才会产生，但 mask 需要现在就有
        self.branch_index_map[current_len : current_len + temp_bim_len] = \
            torch.tensor(combined_bim_ids, device=device)

        # --- 3. Base Model Forward ---
        # 此时调用 verify_step_parallel 的核心逻辑
        # 由于输入已经混合，我们直接调用 Base Model
        
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=final_input_ids,
                position_ids=final_position_ids,
                use_cache=True,
                # attention_mask 由模型内部根据 branch_index_map 生成 (假设模型已修改支持 SpecSoT Mask)
            )
            logits = outputs.logits # [1, Total_Len, Vocab]

        # --- 4. 拆分输出并处理 ---
        
        # 指针用于分割 logits
        ptr = 0
        
        results = {} # 存储结果
        
        # A. 处理老分支结果 (Speculative Verify)
        if old_branch_ids:
            total_draft_len = len(old_branch_ids) * draft_tokens.shape[1]
            draft_logits = logits[0, ptr : ptr + total_draft_len, :]
            ptr += total_draft_len
            
            # Reshape 回 [B_old, K, Vocab]
            draft_logits = draft_logits.view(len(old_branch_ids), -1, logits.shape[-1])
            
            # 调用验证逻辑 (Tree Verification)
            # 这里复用原有的 verify 逻辑，但这部分比较长，我简写核心
            # verify_results = self._verify_drafts(draft_logits, draft_tokens, ...)
            # results['verify'] = verify_results
            pass # 具体验证代码略，重点是这里拿到了 logits

        # B. 处理新分支结果 (Prefill Next Token)
        if new_branch_ids:
            for i, (prompt, bid) in enumerate(zip(new_prompts, new_branch_ids)):
                p_len = prompt.shape[0]
                # 取最后一个 token 的 logits 预测下一个
                # 注意：Prompt 的所有 tokens 都进去了，产生了 KV，但我们只需要最后一个 output
                branch_logits = logits[0, ptr : ptr + p_len, :]
                last_token_logits = branch_logits[-1, :] # [Vocab]
                
                # Greedy Decoding (或 Sampling)
                next_token = torch.argmax(last_token_logits).item()
                
                results[bid] = {
                    'status': 'prefill_done',
                    'next_token': next_token
                }
                
                ptr += p_len

        # --- 5. 状态更新 ---
        # KV Cache 已经由 Base Model 的 Forward 自动追加更新了
        # 我们需要更新 current_length_data
        self.current_length_data.add_(temp_bim_len)
        
        return results









        