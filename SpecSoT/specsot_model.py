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

from .kv_cache import initialize_past_key_values, initialize_draft_past_key_values
from .logits_processor import SemanticLogitsProcessor, VocabScanner

# 分布式推理支持
from .distributed_config import DistributedConfig
from .distributed_prefill import DistributedPrefillManager

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

from dataclasses import dataclass, field


# =============================================================================
# 流水线数据结构定义 (Pipeline Data Structures)
# =============================================================================

@dataclass
class SkeletonPrefillResult:
    """骨架Prefill阶段结果"""
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    input_ids: torch.Tensor
    task_input_ids: torch.Tensor
    input_len: int
    base_prompt_len: int
    max_kv_len: int
    saved_eagle_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@dataclass
class SkeletonDecodeResult:
    """骨架解码阶段结果"""
    skeleton_ids: torch.Tensor
    skeleton_text: str
    decode_time: float


@dataclass
class SkeletonParseResult:
    """骨架解析阶段结果"""
    mode: str  # 'plan', 'direct', 'error'
    tasks: Optional[List[Dict]] = None
    clean_branches: Optional[List[List[int]]] = None
    instruction_len: Optional[Dict[int, int]] = None
    error_msg: Optional[str] = None


@dataclass
class ScheduleResult:
    """调度阶段结果"""
    schedule_plan: Optional[SchedulePlan] = None
    my_branch_ids: List[int] = field(default_factory=list)
    branch_infos: List[BranchInfo] = field(default_factory=list)


@dataclass
class ParallelPrefillResult:
    """并行Prefill阶段结果"""
    input_ids: torch.Tensor
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    tips_indices: torch.Tensor
    branch_begins: List[int]
    branch_lengths: List[int]


@dataclass
class ParallelDecodeResult:
    """并行解码阶段结果"""
    branch_outputs: Dict[int, List[int]]
    stats: Dict[str, Any] = field(default_factory=dict)


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
        # 设置模型为评估模式（禁用 dropout 等训练时特性）
        self.base_model.eval()
        self.eagle_layer.eval()
        
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
            eagle_layer = Eagle3.from_pretrained(
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
            eagle_layer = Eagle2.from_pretrained(
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
    # Phase 1: Skeleton Prefill (骨架Prefill阶段)
    # =========================================================================

    @torch.no_grad()
    def _skeleton_prefill(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
        distributed: bool = False,
    ) -> SkeletonPrefillResult:
        """
        骨架Prefill阶段
        
        初始化KV Cache，执行首次前向传播，准备骨架解码所需的状态。
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成token数
            logits_processor: 采样相关的logits processor
            use_semantic_constraint: 是否使用FSM语义约束
            distributed: 是否使用分布式Prefill
            
        Returns:
            SkeletonPrefillResult: 包含draft_tokens, tree_mask等
        """
        device = self.base_model.device
        model_type = self.get_model_type()
        
        # 准备输入
        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, device
        )
        input_len = input_ids.shape[1]
        base_prompt_len = task_input_ids.shape[1]
        
        # 准备logits processor
        skeleton_logits_processor = logits_processor
        if use_semantic_constraint:
            self._semantic_processor.configure(prefix_len=input_len, enforce_format=True)
            skeleton_logits_processor = LogitsProcessorList([self._semantic_processor])
            if logits_processor is not None:
                for p in logits_processor:
                    skeleton_logits_processor.append(p)
            self.logger.info("使用 FSM 语义约束进行骨架生成")
        
        # 缓存logits processor供后续阶段使用
        self._skeleton_logits_processor = skeleton_logits_processor
        
        # 初始化 KV Cache
        max_kv_len = input_len + max_new_tokens + 100
        self.past_key_values, self.past_key_values_data, self.current_length_data = \
            initialize_past_key_values(self.base_model, max_length=max_kv_len)
        
        # 初始化 Eagle Layer 的 KV Cache
        eagle_max_len = input_len + max_new_tokens + self.eagle_layer.total_tokens + 100
        initialize_draft_past_key_values(self.eagle_layer, max_length=eagle_max_len)
        
        # Prefill
        if distributed:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.distributed_prefill_manager.prefill_single_distributed(
                    input_ids, self.past_key_values, skeleton_logits_processor
                )
        else:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, _ = \
                self.prefill_single(input_ids, skeleton_logits_processor)
        set_random_seed(self.seed)
        
        # 保存 base prompt 的 Eagle KV 状态（用于后续 parallel 阶段恢复）
        saved_eagle_kv = None
        if self.eagle_layer.draft_past_key_values is not None:
            key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
            if hasattr(key_cache, 'data'):
                k_draft = key_cache.data[:, :, :base_prompt_len, :].clone()
                v_draft = value_cache.data[:, :, :base_prompt_len, :].clone()
            else:
                k_draft = key_cache[..., :base_prompt_len, :].clone()
                v_draft = value_cache[..., :base_prompt_len, :].clone()
            saved_eagle_kv = (k_draft, v_draft)
        
        return SkeletonPrefillResult(
            draft_tokens=draft_tokens,
            retrieve_indices=retrieve_indices,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            input_ids=input_ids,
            task_input_ids=task_input_ids,
            input_len=input_len,
            base_prompt_len=base_prompt_len,
            max_kv_len=max_kv_len,
            saved_eagle_kv=saved_eagle_kv,
        )

    # =========================================================================
    # Phase 2: Skeleton Decode (骨架解码阶段)
    # =========================================================================

    @torch.no_grad()
    def _skeleton_decode(
        self,
        prefill_result: SkeletonPrefillResult,
        max_steps: int = 200,
    ) -> SkeletonDecodeResult:
        """
        骨架解码阶段
        
        执行骨架解码循环，生成回答骨架。
        
        Args:
            prefill_result: Prefill阶段的结果
            max_steps: 最大解码步数
            
        Returns:
            SkeletonDecodeResult: 包含skeleton_ids, skeleton_text
        """
        decode_start = time.time()
        
        input_ids = prefill_result.input_ids
        input_len = prefill_result.input_len
        max_kv_len = prefill_result.max_kv_len
        
        draft_tokens = prefill_result.draft_tokens
        retrieve_indices = prefill_result.retrieve_indices
        tree_mask = prefill_result.tree_mask
        tree_position_ids = prefill_result.tree_position_ids
        
        logits_processor = getattr(self, '_skeleton_logits_processor', None)
        eos_token_id = self.tokenizer.eos_token_id
        
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
                logits_processor=logits_processor,
            )
            
            generated_text = self.tokenizer.decode(input_ids[0, input_len:])
            if check_skeleton_stop(
                generated_text, eos_token_id, input_ids, input_len,
                self.current_length_data[0].item(), max_kv_len,
                self.eagle_layer.total_tokens + 1
            ):
                break
        
        # 提取骨架
        skeleton_ids = input_ids[:, input_len:]
        skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
        self.skeleton_output = skeleton_ids.clone()
        
        decode_time = time.time() - decode_start
        self.logger.info(f"Skeleton 解码完成: {decode_time:.3f}s")
        self.logger.info(f"Generated Skeleton: {skeleton_text}")
        
        return SkeletonDecodeResult(
            skeleton_ids=skeleton_ids,
            skeleton_text=skeleton_text,
            decode_time=decode_time,
        )

    # =========================================================================
    # Phase 3: Skeleton Parse (骨架解析阶段)
    # =========================================================================

    def _skeleton_parse(
        self,
        skeleton_text: str,
        task_prompt: str,
    ) -> SkeletonParseResult:
        """
        骨架解析阶段
        
        解析骨架输出，提取分支任务信息。
        
        Args:
            skeleton_text: 骨架文本
            task_prompt: 原始任务描述
            
        Returns:
            SkeletonParseResult: 包含mode, tasks, clean_branches等
        """
        model_type = self.get_model_type()
        
        mode, content = parse_skeleton_output(skeleton_text)
        
        if mode == "direct":
            self.logger.info("直接回答模式，跳过并行阶段")
            return SkeletonParseResult(mode="direct")
        
        if mode == "error":
            self.logger.warning(f"骨架解析错误: {content}")
            return SkeletonParseResult(mode="error", error_msg=str(content))
        
        # mode == "plan": 规划模式
        tasks = content
        self.logger.info(f"检测到 {len(tasks)} 个并行分支: {[t['title'] for t in tasks]}")
        
        # 准备分支prompt tokens
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, tasks, skeleton_text, model_type, task_prompt
        )
        
        # 初始化输出存储
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        
        return SkeletonParseResult(
            mode="plan",
            tasks=tasks,
            clean_branches=clean_branches,
            instruction_len=instruction_len,
        )

    # =========================================================================
    # Phase 4: Schedule (调度阶段)
    # =========================================================================

    def _schedule_branches(
        self,
        parse_result: SkeletonParseResult,
        use_scheduling: bool,
        distributed: bool,
        max_parallel: int,
    ) -> ScheduleResult:
        """
        分支调度阶段
        
        根据分支信息生成调度计划。
        
        Args:
            parse_result: 骨架解析结果
            use_scheduling: 是否使用启发式调度
            distributed: 是否是分布式模式
            max_parallel: 最大并行数
            
        Returns:
            ScheduleResult: 包含schedule_plan, my_branch_ids
        """
        if parse_result.mode != "plan":
            return ScheduleResult()
        
        tasks = parse_result.tasks
        clean_branches = parse_result.clean_branches
        
        # 构建 BranchInfo
        branch_infos = [
            BranchInfo(
                branch_id=i,
                title=task['title'],
                predicted_length=task['length'],
                prompt_tokens=clean_branches[i],
            )
            for i, task in enumerate(tasks)
        ]
        
        # 确定设备数量
        if distributed:
            world_size = self.distributed_config.world_size
        else:
            world_size = 1
        
        # 选择调度器
        if use_scheduling:
            device_profiles = [
                DeviceProfile(device_id=r, max_parallel=max_parallel)
                for r in range(world_size)
            ]
            scheduler = HeuristicScheduler(use_compute_weight=True)
        else:
            device_profiles = [
                DeviceProfile.default_single_device() if world_size == 1 
                else DeviceProfile(device_id=r)
                for r in range(world_size)
            ]
            if distributed:
                scheduler = SimpleDistributedScheduler(all_parallel=True)
            else:
                # 单机朴素模式：所有分支分配给设备0
                scheduler = SimpleDistributedScheduler(all_parallel=True)
        
        schedule_plan = scheduler.schedule(branch_infos, device_profiles)
        self.logger.info(schedule_plan.summary())
        
        # 获取当前设备的分支
        my_rank = self.distributed_config.rank if distributed else 0
        my_plan = schedule_plan.get_plan_for_device(my_rank)
        my_branch_ids = my_plan.assigned_branches if my_plan else []
        
        return ScheduleResult(
            schedule_plan=schedule_plan,
            my_branch_ids=my_branch_ids,
            branch_infos=branch_infos,
        )

    # =========================================================================
    # Phase 5: Parallel Prefill (并行Prefill阶段)
    # =========================================================================

    @torch.no_grad()
    def _parallel_prefill(
        self,
        prefix_ids: torch.Tensor,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> ParallelPrefillResult:
        """
        并行Prefill阶段
        
        为指定分支执行Prefill。
        
        Args:
            prefix_ids: 前缀token IDs
            branch_prompts: 分支prompt列表
            branch_ids: 分支ID列表
            max_new_tokens: 最大生成token数
            logits_processor: logits处理器
            
        Returns:
            ParallelPrefillResult: 包含draft_tokens等
        """
        prefix_len = prefix_ids.shape[1]
        
        # 准备批次数据
        input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids = \
            self._prepare_parallel_batch(prefix_ids, branch_prompts, max_new_tokens, branch_ids)
        
        # 执行Prefill
        input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = \
            self.prefill_parallel(
                prefix_len, input_ids, tips_indices,
                branch_begins, branch_lengths, draft_input_ids, logits_processor
            )
        
        return ParallelPrefillResult(
            input_ids=input_ids,
            draft_tokens=draft_tokens,
            retrieve_indices=retrieve_indices,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            tips_indices=tips_indices,
            branch_begins=branch_begins,
            branch_lengths=branch_lengths,
        )

    # =========================================================================
    # Phase 6: Parallel Decode (并行解码阶段)
    # =========================================================================

    @torch.no_grad()
    def _parallel_decode(
        self,
        prefill_result: ParallelPrefillResult,
        schedule_result: ScheduleResult,
        clean_branches: List[List[int]],
        max_new_tokens: int,
        max_kv_len: int,
        prefix_len: int,
        use_scheduling: bool,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> ParallelDecodeResult:
        """
        并行解码阶段
        
        执行并行分支解码，支持朴素模式和连续批处理模式。
        
        Args:
            prefill_result: Prefill阶段结果
            schedule_result: 调度阶段结果
            clean_branches: 所有分支的prompt tokens
            max_new_tokens: 最大生成token数
            max_kv_len: KV Cache最大长度
            prefix_len: 前缀长度
            use_scheduling: 是否使用连续批处理
            logits_processor: logits处理器
            
        Returns:
            ParallelDecodeResult: 包含branch_outputs, stats
        """
        device = self.base_model.device
        tokens_per_branch = self.eagle_layer.total_tokens + 1
        
        draft_tokens = prefill_result.draft_tokens
        retrieve_indices = prefill_result.retrieve_indices
        tree_mask = prefill_result.tree_mask
        tree_position_ids = prefill_result.tree_position_ids
        
        stats = {}
        
        if use_scheduling and schedule_result.schedule_plan is not None:
            # ========== 连续批处理模式 ==========
            branch_info_dict = {info.branch_id: info for info in schedule_result.branch_infos}
            my_plan = schedule_result.schedule_plan.get_plan_for_device(0)
            
            if my_plan is None:
                return ParallelDecodeResult(branch_outputs={}, stats={})
            
            exec_manager = BranchExecutionManager(
                execution_plan=my_plan,
                branch_infos=branch_info_dict,
                rank=0,
            )
            exec_manager.activate_branches(schedule_result.my_branch_ids)
            
            for step in range(max_new_tokens):
                if check_stop_conditions_parallel(
                    current_length=self.current_length_data[0].item(),
                    max_kv_len=max_kv_len,
                    num_active_branches=len(self.active_branches),
                    tokens_per_branch=tokens_per_branch,
                ):
                    break
                
                if exec_manager.is_all_completed():
                    self.logger.info("所有分支执行完成")
                    break
                
                # 解码
                if self.prefilling_branches:
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
                
                # 处理完成的分支，动态加入新分支
                if self.recently_completed_branches:
                    completed = self.recently_completed_branches
                    branch_outputs = {
                        bid: self.parallel_branches_output[bid]
                        for bid in completed
                    }
                    exec_manager.handle_completed_branches(completed, branch_outputs)
                    
                    new_branches = exec_manager.get_branches_to_add()
                    if new_branches:
                        self.logger.info(f"Continuous Batching: 加入新分支 {new_branches}")
                        self._add_branches_to_active(
                            new_branches, clean_branches, prefix_len, logits_processor
                        )
                        exec_manager.activate_branches(new_branches)
                
                if all_finished:
                    break
            
            stats['execution_stats'] = exec_manager.get_stats()
            
        else:
            # ========== 朴素模式：所有分支同时解码 ==========
            total_accept_len = torch.zeros(1, dtype=torch.long, device=device)
            total_draft_time = 0.0
            total_update_time = 0.0
            total_verify_time = 0.0
            
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_after_verify = torch.cuda.Event(enable_timing=True)
            evt_after_update = torch.cuda.Event(enable_timing=True)
            evt_after_draft = torch.cuda.Event(enable_timing=True)
            step_parallel = 0
            
            for step_parallel in range(max_new_tokens):
                if check_stop_conditions_parallel(
                    current_length=self.current_length_data[0].item(),
                    max_kv_len=max_kv_len,
                    num_active_branches=len(self.active_branches),
                    tokens_per_branch=tokens_per_branch,
                ):
                    self.logger.warning(f"KV cache 限制，未完成分支: {self.active_branches}")
                    break
                
                evt_start.record()
                
                (
                    draft_tokens, retrieve_indices, tree_mask,
                    tree_position_ids, accept_length, all_finished,
                ) = self.decode_step_parallel(
                    draft_tokens=draft_tokens,
                    retrieve_indices=retrieve_indices,
                    tree_mask=tree_mask,
                    tree_position_ids=tree_position_ids,
                    logits_processor=logits_processor,
                    evt_after_verify=evt_after_verify,
                    evt_after_update=evt_after_update,
                )
                evt_after_draft.record()
                
                total_accept_len += accept_length.sum()
                
                torch.cuda.synchronize()
                total_verify_time += evt_start.elapsed_time(evt_after_verify) / 1000
                total_update_time += evt_after_verify.elapsed_time(evt_after_update) / 1000
                total_draft_time += evt_after_update.elapsed_time(evt_after_draft) / 1000
                
                if all_finished:
                    self.logger.info("所有分支解码完成")
                    break
            
            num_steps = max(step_parallel, 1)
            stats = {
                'avg_accept_len': total_accept_len.item() / num_steps,
                'avg_draft_time': total_draft_time / num_steps,
                'avg_update_time': total_update_time / num_steps,
                'avg_verify_time': total_verify_time / num_steps,
            }
        
        # 收集输出
        branch_outputs = {}
        for bid in schedule_result.my_branch_ids:
            branch_outputs[bid] = self.parallel_branches_output[bid]
        
        return ParallelDecodeResult(branch_outputs=branch_outputs, stats=stats)

    # =========================================================================
    # Phase 7: Result Merge (结果合并阶段)
    # =========================================================================

    def _merge_results(
        self,
        skeleton_ids: torch.Tensor,
        tasks: List[Dict],
    ) -> torch.Tensor:
        """
        结果合并阶段
        
        合并骨架和分支输出。
        
        Args:
            skeleton_ids: 骨架token IDs
            tasks: 分支任务列表
            
        Returns:
            merged_ids: 合并后的token IDs
        """
        device = self.base_model.device
        
        merged_ids = merge_outputs(
            skeleton_output=self.skeleton_output,
            parallel_branches_output=self.parallel_branches_output,
            instruction_len=self.instruction_len,
            device=device,
            tasks=tasks,
            tokenizer=self.tokenizer,
        )
        
        return merged_ids

    # =========================================================================
    # 流水线控制器 (Pipeline Controllers)
    # =========================================================================

    @torch.no_grad()
    def _run_specsot_pipeline(
        self,
        task_prompt: str,
        max_new_tokens: int,
        max_parallel: int,
        logits_processor: Optional[LogitsProcessorList],
        use_semantic_constraint: bool,
        use_scheduling: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        SpecSoT 主流水线控制器
        
        协调执行所有阶段的流水线。单机和分布式Master都使用此方法。
        
        Args:
            task_prompt: 任务描述
            max_new_tokens: 最大生成token数
            max_parallel: 最大并行数
            logits_processor: logits处理器
            use_semantic_constraint: 是否使用语义约束
            use_scheduling: 是否使用调度
            
        Returns:
            output_ids, stats
        """
        device = self.base_model.device
        distributed = self.is_distributed()
        is_master = not distributed or self.distributed_config.rank == 0
        
        mode_name = f"{'distributed' if distributed else 'local'}_{'scheduling' if use_scheduling else 'naive'}"
        stats = {
            'skeleton_time': 0.0,
            'scheduling_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': mode_name,
            'avg_accept_len': 0.0,
            'avg_draft_time': 0.0,
            'avg_update_time': 0.0,
            'avg_verify_time': 0.0,
        }
        
        self.logger.info(f"开始 SpecSoT 流水线: distributed={distributed}, use_scheduling={use_scheduling}")
        
        # =====================================================================
        # Phase 1: Skeleton Prefill
        # =====================================================================
        skeleton_start = time.time()
        prefill_result = self._skeleton_prefill(
            task_prompt=task_prompt,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            use_semantic_constraint=use_semantic_constraint,
            distributed=distributed,
        )
        
        # 分布式Worker在Prefill后进入独立流程
        if distributed and not is_master:
            return self._worker_pipeline(
                prefill_result=prefill_result,
                max_new_tokens=max_new_tokens,
                max_parallel=max_parallel,
                use_scheduling=use_scheduling,
                logits_processor=logits_processor,
            )
        
        # =====================================================================
        # Phase 2: Skeleton Decode (仅Master)
        # =====================================================================
        decode_result = self._skeleton_decode(prefill_result)
        
        stats['skeleton_time'] = time.time() - skeleton_start
        
        # =====================================================================
        # Phase 3: Skeleton Parse
        # =====================================================================
        parse_result = self._skeleton_parse(
            skeleton_text=decode_result.skeleton_text,
            task_prompt=task_prompt,
        )
        
        # 处理 direct/error 模式
        if parse_result.mode != "plan":
            if distributed:
                self._notify_workers_complete()
            return decode_result.skeleton_ids, stats
        
        stats['num_branches'] = len(parse_result.tasks)
        
        # =====================================================================
        # Phase 4: Schedule
        # =====================================================================
        scheduling_start = time.time()
        schedule_result = self._schedule_branches(
            parse_result=parse_result,
            use_scheduling=use_scheduling,
            distributed=distributed,
            max_parallel=max_parallel,
        )
        stats['scheduling_time'] = time.time() - scheduling_start
        
        # =====================================================================
        # 分布式: 分发任务给Worker
        # =====================================================================
        if distributed:
            self._distribute_tasks_to_workers(schedule_result, parse_result)
        
        # =====================================================================
        # Phase 5: Parallel Prefill
        # =====================================================================
        parallel_start = time.time()
        
        # 恢复Eagle KV到base prompt状态
        self._restore_eagle_kv(prefill_result.saved_eagle_kv)
        
        # 获取当前设备要执行的分支
        my_branch_ids = schedule_result.my_branch_ids
        if not my_branch_ids:
            self.logger.warning("当前设备没有分配分支")
            if distributed:
                # 收集Worker结果
                all_outputs = self._collect_worker_results(
                    schedule_result, len(parse_result.tasks) - len(my_branch_ids)
                )
                for bid, tokens in all_outputs.items():
                    self.parallel_branches_output[bid] = tokens
            stats['parallel_time'] = time.time() - parallel_start
            merged_ids = self._merge_results(decode_result.skeleton_ids, parse_result.tasks)
            return merged_ids, stats
        
        my_prompts = [parse_result.clean_branches[bid] for bid in my_branch_ids]
        
        parallel_prefill_result = self._parallel_prefill(
            prefix_ids=prefill_result.task_input_ids,
            branch_prompts=my_prompts,
            branch_ids=my_branch_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        
        # =====================================================================
        # Phase 6: Parallel Decode
        # =====================================================================
        parallel_decode_result = self._parallel_decode(
            prefill_result=parallel_prefill_result,
            schedule_result=schedule_result,
            clean_branches=parse_result.clean_branches,
            max_new_tokens=max_new_tokens,
            max_kv_len=prefill_result.max_kv_len,
            prefix_len=prefill_result.task_input_ids.shape[1],
            use_scheduling=use_scheduling,
            logits_processor=logits_processor,
        )
        
        # 更新本地输出
        for bid, tokens in parallel_decode_result.branch_outputs.items():
            self.parallel_branches_output[bid] = tokens
        
        stats.update(parallel_decode_result.stats)
        
        # =====================================================================
        # 分布式: 收集Worker结果
        # =====================================================================
        if distributed:
            num_other = len(parse_result.tasks) - len(my_branch_ids)
            if num_other > 0:
                other_outputs = self._collect_worker_results(schedule_result, num_other)
                for bid, tokens in other_outputs.items():
                    self.parallel_branches_output[bid] = tokens
            
            # 通知Worker完成
            self._notify_workers_complete()
        
        stats['parallel_time'] = time.time() - parallel_start
        self.logger.info(f"并行解码完成: {stats['parallel_time']:.3f}s")
        
        # =====================================================================
        # Phase 7: Result Merge
        # =====================================================================
        merged_ids = self._merge_results(decode_result.skeleton_ids, parse_result.tasks)
        
        return merged_ids, stats

    @torch.no_grad()
    def _worker_pipeline(
        self,
        prefill_result: SkeletonPrefillResult,
        max_new_tokens: int,
        max_parallel: int,
        use_scheduling: bool,
        logits_processor: Optional[LogitsProcessorList],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        分布式Worker流水线
        
        Worker在Skeleton Prefill后进入此流程，等待Master分发任务。
        
        Args:
            prefill_result: Prefill阶段结果
            max_new_tokens: 最大生成token数
            max_parallel: 最大并行数
            use_scheduling: 是否使用调度
            logits_processor: logits处理器
            
        Returns:
            空tensor和统计信息（Worker不返回最终结果）
        """
        device = self.base_model.device
        rank = self.distributed_config.rank
        
        stats = {
            'parallel_time': 0.0,
            'num_branches': 0,
            'mode': f'worker_{"scheduling" if use_scheduling else "naive"}',
        }
        
        self.logger.info(f"Worker {rank} 等待接收任务...")
        
        # 接收任务
        task_data = self._receive_tasks_from_master()
        if task_data is None:
            self.logger.info(f"Worker {rank} 收到完成信号，退出")
            return torch.tensor([], device=device), stats
        
        schedule_plan, branch_infos = task_data
        branch_ids = [info.branch_id for info in branch_infos]
        branch_prompts = [info.prompt_tokens for info in branch_infos]
        
        stats['num_branches'] = len(branch_ids)
        self.logger.info(f"Worker {rank} 收到 {len(branch_ids)} 个分支任务")
        
        # 初始化输出存储
        max_branch_id = max(branch_ids) + 1
        self.parallel_branches_output = [[] for _ in range(max_branch_id)]
        for info in branch_infos:
            self.parallel_branches_output[info.branch_id] = list(info.prompt_tokens)
        self.instruction_len = {info.branch_id: len(info.prompt_tokens) for info in branch_infos}
        
        # 准备执行
        parallel_start = time.time()
        prefix_len = prefill_result.base_prompt_len
        max_kv_len = prefix_len + max_new_tokens + 500
        
        # 扩展KV Cache如果需要
        if self.past_key_values_data[0].shape[3] < max_kv_len:
            old_cache = [d[..., :prefix_len, :].clone() for d in self.past_key_values_data]
            self.past_key_values, self.past_key_values_data, self.current_length_data = \
                initialize_past_key_values(self.base_model, max_length=max_kv_len)
            for i, d in enumerate(old_cache):
                self.past_key_values_data[i][..., :prefix_len, :].copy_(d)
            self.current_length_data.fill_(prefix_len)
        
        # 创建虚拟prefix_ids
        prefix_ids = torch.zeros((1, prefix_len), dtype=torch.long, device=device)
        
        # Phase 5: Parallel Prefill
        parallel_prefill_result = self._parallel_prefill(
            prefix_ids=prefix_ids,
            branch_prompts=branch_prompts,
            branch_ids=branch_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        
        # 构建调度结果
        my_plan = schedule_plan.get_plan_for_device(rank)
        schedule_result = ScheduleResult(
            schedule_plan=schedule_plan,
            my_branch_ids=branch_ids,
            branch_infos=branch_infos,
        )
        
        # Phase 6: Parallel Decode
        parallel_decode_result = self._parallel_decode(
            prefill_result=parallel_prefill_result,
            schedule_result=schedule_result,
            clean_branches={info.branch_id: info.prompt_tokens for info in branch_infos},
            max_new_tokens=max_new_tokens,
            max_kv_len=max_kv_len,
            prefix_len=prefix_len,
            use_scheduling=use_scheduling,
            logits_processor=logits_processor,
        )
        
        stats['parallel_time'] = time.time() - parallel_start
        
        # 发送结果给Master
        self._send_results_to_master(parallel_decode_result.branch_outputs)
        
        self.logger.info(f"Worker {rank} 完成，等待完成信号...")
        
        # 等待完成信号
        comm = self.distributed_prefill_manager.comm
        comm.recv_complete_signal(timeout=300.0)
        
        return torch.tensor([], device=device), stats

    # =========================================================================
    # 分布式通信辅助方法 (Distributed Communication Helpers)
    # =========================================================================

    def _notify_workers_complete(self) -> None:
        """通知所有Worker完成"""
        if not self.is_distributed():
            return
        comm = self.distributed_prefill_manager.comm
        comm.broadcast_parallel_complete_signal()

    def _distribute_tasks_to_workers(
        self,
        schedule_result: ScheduleResult,
        parse_result: SkeletonParseResult,
    ) -> None:
        """分发任务给Worker"""
        if not self.is_distributed():
            return
        
        comm = self.distributed_prefill_manager.comm
        schedule_plan = schedule_result.schedule_plan
        branch_infos = schedule_result.branch_infos
        
        # 序列化调度计划
        schedule_plan_data = {
            'num_branches': schedule_plan.num_branches,
            'num_devices': schedule_plan.num_devices,
            'branch_to_device': schedule_plan.branch_to_device,
            'device_plans': {
                did: {
                    'device_id': plan.device_id,
                    'execution_batches': plan.execution_batches,
                    'assigned_branches': plan.assigned_branches,
                    'max_parallel': plan.max_parallel,
                }
                for did, plan in schedule_plan.device_plans.items()
            },
            'scheduler_type': schedule_plan.scheduler_type,
        }
        
        # 广播调度计划
        comm.send_schedule_plan_async(schedule_plan_data, dst_rank=-1)
        
        # 发送分支Prompt到各Worker
        device_branches = {}
        for info in branch_infos:
            device_id = schedule_plan.get_device_for_branch(info.branch_id)
            if device_id not in device_branches:
                device_branches[device_id] = []
            device_branches[device_id].append(info)
        
        for device_id, infos in device_branches.items():
            if device_id == 0:
                continue  # Master不发给自己
            branch_data = [
                {
                    'branch_id': info.branch_id,
                    'title': info.title,
                    'predicted_length': info.predicted_length,
                    'prompt_tokens': info.prompt_tokens,
                }
                for info in infos
            ]
            comm.send_branch_prompt_async(branch_data, device_id)

    def _collect_worker_results(
        self,
        schedule_result: ScheduleResult,
        num_branches: int,
    ) -> Dict[int, List[int]]:
        """收集Worker结果"""
        if not self.is_distributed() or num_branches == 0:
            return {}
        
        comm = self.distributed_prefill_manager.comm
        self.logger.info(f"收集其他设备的分支输出 ({num_branches} 个)...")
        
        outputs = comm.collect_all_branch_outputs(
            num_branches=num_branches,
            timeout=300.0
        )
        return outputs

    def _receive_tasks_from_master(self) -> Optional[Tuple[SchedulePlan, List[BranchInfo]]]:
        """Worker接收Master分发的任务"""
        comm = self.distributed_prefill_manager.comm
        
        # 接收调度计划
        schedule_plan_data = comm.recv_schedule_plan(timeout=60.0)
        if schedule_plan_data is None:
            return None
        
        # 反序列化
        device_plans = {}
        for did, pdata in schedule_plan_data['device_plans'].items():
            device_plans[int(did)] = DeviceExecutionPlan(
                device_id=pdata['device_id'],
                execution_batches=pdata['execution_batches'],
                assigned_branches=pdata['assigned_branches'],
                max_parallel=pdata['max_parallel'],
            )
        schedule_plan = SchedulePlan(
            num_branches=schedule_plan_data['num_branches'],
            num_devices=schedule_plan_data['num_devices'],
            branch_to_device=schedule_plan_data['branch_to_device'],
            device_plans=device_plans,
            scheduler_type=schedule_plan_data['scheduler_type'],
        )
        
        # 接收分支Prompt
        branch_data_list = comm.recv_branch_prompts(timeout=60.0)
        if not branch_data_list:
            return None
        
        branch_infos = [
            BranchInfo(
                branch_id=data['branch_id'],
                title=data['title'],
                predicted_length=data['predicted_length'],
                prompt_tokens=data['prompt_tokens'],
            )
            for data in branch_data_list
        ]
        
        return schedule_plan, branch_infos

    def _send_results_to_master(self, branch_outputs: Dict[int, List[int]]) -> None:
        """Worker发送结果给Master"""
        comm = self.distributed_prefill_manager.comm
        for bid, tokens in branch_outputs.items():
            comm.send_branch_output_async(
                branch_id=bid,
                output_tokens=tokens,
                dst_rank=0,
            )

    def _restore_eagle_kv(self, saved_eagle_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """恢复Eagle KV Cache到base prompt状态"""
        if saved_eagle_kv is None:
            return
        k_saved, v_saved = saved_eagle_kv
        key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
        if hasattr(key_cache, 'restore_from_tensor'):
            key_cache.restore_from_tensor(k_saved.clone())
            value_cache.restore_from_tensor(v_saved.clone())
        else:
            self.eagle_layer.draft_past_key_values = ((k_saved.clone(), v_saved.clone()),)

    # =========================================================================
    # 并行解码辅助方法 (Parallel Decoding Helpers)
    # =========================================================================

    def _prepare_parallel_batch(
        self,
        prefix_ids: torch.Tensor,
        branches_prompts: List[List[int]],
        max_new_tokens: int,
        branch_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int], torch.Tensor]:
        """
        准备并行解码的批次数据（统一入口）
        
        该方法合并了原来的 reuse_prefix_for_parallel 和 _prepare_batch_for_prefill，
        用于在 Skeleton 生成完成后，为各分支准备并行解码的输入数据。
        
        核心步骤：
        1. 复制 KV Cache: 将 skeleton 的 prefix KV 复制为多分支
        2. 构建 BIM: 创建 Branch Index Map，追踪每个 token 的分支归属
        3. 打包输入: 将所有分支的 prompt 打包成一个序列
        4. 位置编码: 为各分支构建正确的位置编码
        
        Args:
            prefix_ids: 共享前缀 (skeleton) [1, prefix_len]
            branches_prompts: 各分支的 prompt token 列表
            max_new_tokens: 每个分支最大生成长度
            branch_ids: 分支 ID 列表（可选，用于 Continuous Batching）
                        如果为 None，则使用 0, 1, 2, ... 作为 ID
            
        Returns:
            input_ids: 打包后的输入 [1, total_len]
            tips_indices: 各分支 tip 位置
            branch_begins: 各分支起始位置
            branch_lengths: 各分支初始长度
            draft_input_ids: Draft 模型的输入 [num_para, max_len]
        """
        device = self.base_model.device
        num_para = len(branches_prompts)
        prefix_len = prefix_ids.shape[1]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # 如果没有指定 branch_ids，使用默认的 0, 1, 2, ...
        if branch_ids is None:
            branch_ids = list(range(num_para))
        
        # 初始化活跃分支列表
        self.active_branches = list(branch_ids)
        
        # ---------------------------------------------------------------------
        # 1. Base Model KV Cache: 重置到 prefix 长度
        # ---------------------------------------------------------------------
        self.current_length_data.fill_(prefix_len)
        
        # ---------------------------------------------------------------------
        # 2. Eagle Layer KV Cache: 复制 prefix KV 为多分支
        # ---------------------------------------------------------------------
        if self.eagle_layer.draft_past_key_values is not None:
            key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
            if hasattr(key_cache, 'data'):
                k_draft = key_cache.data[:, :, :key_cache.shape[2], :]
                v_draft = value_cache.data[:, :, :value_cache.shape[2], :]
            else:
                k_draft = key_cache
                v_draft = value_cache
            k_prefix = k_draft[..., :prefix_len, :].clone()
            v_prefix = v_draft[..., :prefix_len, :].clone()
            k_expanded = k_prefix.expand(num_para, -1, -1, -1).clone()
            v_expanded = v_prefix.expand(num_para, -1, -1, -1).clone()
            # 批量模式下使用普通 tensor 格式
            self.eagle_layer.draft_past_key_values = ((k_expanded, v_expanded),)
            self.eagle_layer.kv_cache_initialized = False
        
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
        for i, (br, bid) in enumerate(zip(branches_prompts, branch_ids)):
            curr_len = len(br)
            branch_begins.append(current_offset)
            flat_branch_ids.extend(br)
            branch_index_list.extend([bid] * curr_len)
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
            draft_branch_list, pad_id=pad_token_id, device=device, return_mask=True
        )
        padded_branch_pos = stack_with_left_padding(draft_pos_list, pad_id=0, device=device)
        
        prefix_mask = torch.ones((num_para, prefix_len), dtype=torch.long, device=device)
        prefix_pos = torch.arange(prefix_len, device=device).unsqueeze(0).expand(num_para, -1)
        self.eagle_layer.cache_padding_mask = torch.cat([prefix_mask, branch_mask], dim=1)
        self.eagle_layer.full_position_ids = torch.cat([prefix_pos, padded_branch_pos], dim=1)
        
        return input_ids, tips_indices, branch_begins, branch_lengths, draft_input_ids

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
        use_scheduling: bool = False,
        max_parallel: int = 2,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        统一生成入口函数
        
        执行逻辑：
        1. enable_parallel=False: 使用纯投机解码 (generate_eagle)
        2. enable_parallel=True:
           - distributed=True: 调用 generate_distributed
           - distributed=False: 调用 generate_local
        3. 在 generate_local/generate_distributed 内部：
           - use_scheduling=False: 朴素并行分配
           - use_scheduling=True: Continuous Batching 调度
        
        Args:
            task_prompt: 用户输入的任务描述
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            enable_parallel: 是否启用骨架并行模式 (SpecSoT)
            use_scheduling: 是否启用 Continuous Batching 调度
            max_parallel: 最大并行分支数 (仅调度模式有效)
            use_semantic_constraint: 是否使用 FSM 语义约束
            
        Returns:
            output_ids: 生成的 token IDs
            stats: 统计信息字典，包含:
                - total_time: 总推理时间
                - skeleton_time: 骨架生成时间 (仅并行模式)
                - parallel_time: 并行解码时间 (仅并行模式)
                - num_branches: 分支数量 (仅并行模式)
                - avg_accept_len: 平均接受长度
                - mode: 执行模式名称
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

        # =====================================================================
        # 分支1: 不启用并行 -> 纯投机解码 (EAGLE)
        # =====================================================================
        if not enable_parallel:
            output_ids, stats = self.generate_eagle(task_prompt, max_new_tokens, logits_processor)
            stats['mode'] = 'eagle'
            stats['total_time'] = time.time() - total_start_time
            return output_ids, stats

        # =====================================================================
        # 分支2: 启用并行 -> SpecSoT Pipeline (统一入口)
        # =====================================================================
        output_ids, stats = self._run_specsot_pipeline(
            task_prompt=task_prompt,
            max_new_tokens=max_new_tokens,
            max_parallel=max_parallel,
            logits_processor=logits_processor,
            use_semantic_constraint=use_semantic_constraint,
            use_scheduling=use_scheduling,
        )
        
        stats['total_time'] = time.time() - total_start_time
        return output_ids, stats
    
    # 非并行解码
    @torch.no_grad()
    def generate_eagle(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        纯投机解码生成（不使用骨架并行）
        
        流程：
        1. 初始化: 准备 input_ids, KV Cache
        2. Prefill: 初始化 KV Cache 和第一轮 Draft Tree (支持分布式)
        3. Decode Loop: 循环执行 decode_step_single 直到停止
        
        Returns:
            output_ids: 生成的 token IDs
            stats: 统计信息字典
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
        
        # 初始化 Eagle Layer 的 KV Cache
        # Eagle Layer 的 max_length 由 input_len + top_k * depth 决定
        # top_k * depth 是单步 draft tree 生成的最大 token 数
        eagle_max_len = input_len + max_new_tokens + self.eagle_layer.total_tokens + 100
        initialize_draft_past_key_values(self.eagle_layer, max_length=eagle_max_len)
        
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
        step = 0
        
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
        
        # 计算平均值并构建统计字典
        num_steps = max(step, 1)
        output_ids = input_ids[:, input_len:]
        
        stats = {
            'skeleton_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
            'avg_accept_len': total_accept_len.item() / num_steps,
            'avg_draft_time': total_draft_time / num_steps,
            'avg_update_time': total_update_time / num_steps,
            'avg_verify_time': total_verify_time / num_steps,
        }
        
        return output_ids, stats

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
            # 注意：批量并行模式下使用普通 tensor 格式，不使用 KVCache 类
            if self.eagle_layer.draft_past_key_values is not None:
                key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
                # 获取实际的 tensor 数据（兼容 KVCache 类和普通 tensor）
                if hasattr(key_cache, 'data'):
                    k_draft = key_cache.data[:, :, :key_cache.shape[2], :]
                    v_draft = value_cache.data[:, :, :value_cache.shape[2], :]
                else:
                    k_draft = key_cache
                    v_draft = value_cache
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
                # 批量模式下使用普通 tensor 格式
                self.eagle_layer.draft_past_key_values = ((k_combined, v_combined),)
                self.eagle_layer.kv_cache_initialized = False  # 标记退出 KVCache 模式
            
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
            # 批量模式下使用普通 tensor 格式
            if self.eagle_layer.draft_past_key_values is not None:
                key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
                # 获取实际的 tensor 数据
                if hasattr(key_cache, 'data'):
                    k_stable = key_cache.data[:, :, :key_cache.shape[2], :][keep_mask_tensor]
                    v_stable = value_cache.data[:, :, :value_cache.shape[2], :][keep_mask_tensor]
                else:
                    k_stable = key_cache[keep_mask_tensor]
                    v_stable = value_cache[keep_mask_tensor]
                self.eagle_layer.draft_past_key_values = ((k_stable, v_stable),)
                self.eagle_layer.kv_cache_initialized = False  # 批量模式下退出 KVCache 模式

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
    # Continuous Batching 辅助方法
    # =========================================================================

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
    






        