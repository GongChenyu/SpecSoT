# coding=utf-8
"""
SpecSoT 生成器 (Generator)

协调整个 SpecSoT 流水线的执行：
1. Skeleton Generation (骨架生成)
2. Skeleton Parsing (骨架解析)
3. Parallel Branch Decoding (并行分支解码)
4. Result Merge (结果合并)

设计原则：
1. 清晰的阶段分离
2. 统一的数据流
3. 支持单机和分布式模式
"""

import time
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessorList

from .datatypes import (
    SkeletonPrefillResult,
    SkeletonDecodeResult,
    SkeletonParseResult,
    ScheduleResult,
    ParallelPrefillResult,
    ParallelDecodeResult,
    GenerateResult,
)

from ..core import (
    BranchStateManager,
    KVCache,
    Drafter,
)

from .inference import InferenceEngine

from ..models.base_model import (
    LlamaForCausalLMKV as KVLlamaForCausalLM,
    Qwen2ForCausalLMKV as KVQwen2ForCausalLM,
    Qwen3ForCausalLMKV as KVQwen3ForCausalLM,
    MixtralForCausalLMKV as KVMixtralForCausalLM,
)

from ..models.draft_model import Eagle3, Eagle2

from ..processing.prompts import (
    prepare_skeleton_input,
    prepare_parallel_branches,
    parse_skeleton_output,
)

from ..core.scheduling import (
    BranchInfo,
    DeviceProfile,
    HeuristicScheduler,
    SimpleDistributedScheduler,
    BranchExecutionManager,
)


from ..core.distributed.distributed_prefill import DistributedPrefillManager
from ..core.communication.task_coordinator import DistributedTaskCoordinator

from ..utils.utils import stack_with_left_padding


class SpecSoTGenerator:
    """
    SpecSoT 生成器 (Generator)
    
    协调整个 SpecSoT 流水线的执行。
    
    Attributes:
        base_model: Base Model
        eagle_layer: Eagle Layer
        drafter: Draft Tree 生成器
        tokenizer: 分词器
        inference_engine: 推理引擎
        state_manager: 状态管理器
        distributed_config: 分布式配置
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        eagle_layer: nn.Module,
        drafter: Any,
        tokenizer: Any,
        distributed_config: Optional[Any] = None,
        logger: Optional[Any] = None,
        use_bim_mode: bool = True,
        seed: int = 42,
    ):
        """
        初始化编排器

        Args:
            base_model: Base Model
            eagle_layer: Eagle Layer
            drafter: Draft Tree 生成器
            tokenizer: 分词器
            distributed_config: 分布式配置
            logger: 日志记录器
            use_bim_mode: 是否使用 BIM 模式
            seed: 随机种子
        """
        self.base_model = base_model
        self.eagle_layer = eagle_layer
        self.drafter = drafter
        self.tokenizer = tokenizer
        self.distributed_config = distributed_config
        self.logger = logger
        self.use_bim_mode = use_bim_mode
        self.seed = seed

        # =====================================================================
        # 1. Base Model 配置提取
        # =====================================================================
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]

        # 设备
        self.device = next(base_model.parameters()).device

        # =====================================================================
        # 2. 推理引擎初始化
        # =====================================================================
        self.inference_engine = InferenceEngine(
            base_model=base_model,
            eagle_layer=eagle_layer,
            drafter=drafter,
            device=self.device,
        )
        
        # 状态管理器（延迟初始化）
        self.state_manager: Optional[BranchStateManager] = None
        
        # KV Cache 引用
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None
        
        # 输出存储
        self.skeleton_output = None
        self.parallel_branches_output = None
        self.instruction_len = None
        
        # 活跃分支
        self.active_branches: List[int] = []
        
        # Continuous Batching 状态（用于调度模式）
        self.recently_completed_branches: List[int] = []
        self.prefilling_branches: List[int] = []  # 正在 prefill 的新分支
        self.pending_prefill_prompts: Dict[int, List[int]] = {}  # 新分支的 prompt tokens 缓存
        
        # Branch Index Map 和 Position IDs（用于并行解码）
        self.branch_index_map: Optional[torch.Tensor] = None
        self.full_position_ids: Optional[torch.Tensor] = None

        # =====================================================================
        # 3. Semantic Logits Processor 预初始化
        # =====================================================================
        from ..processing.logits_processor import SemanticLogitsProcessor, VocabScanner
        self._vocab_scanner = VocabScanner(tokenizer)
        self._semantic_processor = SemanticLogitsProcessor(
            tokenizer=tokenizer,
            prefix_len=0,
            enforce_format=True,
            vocab_scanner=self._vocab_scanner,
        )

        # =====================================================================
        # 4. 推理状态初始化
        # =====================================================================
        from ..utils.utils import set_random_seed
        base_model.eval()
        eagle_layer.eval()
        set_random_seed(seed)
        self._reset_state()
        eagle_layer.reset_state()

        # =====================================================================
        # 5. 分布式初始化
        # =====================================================================
        if distributed_config and getattr(distributed_config, 'enabled', False):
            self._init_distributed()

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
        distributed_config: Optional[Any] = None,
        logger: Optional[Any] = None,
        use_bim_mode: bool = True,
        **kwargs,
    ) -> "SpecSoTGenerator":
        """
        从预训练模型加载 SpecSoT 生成器
        
        加载流程：
        1. 加载 Base Model: 调用 _load_base_model
        2. 加载 Eagle Layer: 调用 Eagle.from_pretrained
        3. 组装 SpecSoT Generator
        
        Args:
            base_model_path: 基础模型路径
            ea_model_path: Eagle 模型路径
            use_eagle3: 是否使用 Eagle3
            total_token: 每次 draft 的总 token 数
            depth: draft 树深度
            top_k: top-k 选择数量
            threshold: 接受阈值
            seed: 随机种子
            distributed_config: 分布式推理配置
            logger: 日志记录器
            use_bim_mode: 是否使用 BIM 模式
            **kwargs: 传递给基础模型的其他参数
            
        Returns:
            SpecSoTGenerator 实例
        """
        # =====================================================================
        # 1. 加载 Base Model
        # =====================================================================
        base_model = cls._load_base_model(base_model_path, **kwargs)
        
        # =====================================================================
        # 2. 加载 Tokenizer
        # =====================================================================
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, use_fast=False
        )

        # =====================================================================
        # 3. 加载 Eagle Layer (根据 use_eagle3 选择不同版本)
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
        # 4. 创建 Drafter
        # =====================================================================
        drafter = Drafter(eagle_layer)

        # =====================================================================
        # 5. 组装 SpecSoT Generator
        # =====================================================================
        return cls(
            base_model=base_model,
            eagle_layer=eagle_layer,
            drafter=drafter,
            tokenizer=tokenizer,
            distributed_config=distributed_config,
            logger=logger,
            use_bim_mode=use_bim_mode,
            seed=seed,
        )

    def _reset_state(self):
        """
        初始化/重置推理状态（在每次推理开始前调用）

        重置内容：
        1. Base Model: tree mode
        2. Base Model KV Cache 引用
        3. 并行解码状态: BIM, position_ids, active_branches
        4. Continuous Batching 状态
        5. 输出存储
        """
        # Base Model tree mode
        self.base_model.model.tree_mask = None
        self.base_model.model.tree_mode = None

        # Base Model KV Cache 引用 (实际分配在 generate 中)
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

        # 并行解码状态
        self.branch_index_map = None
        self.full_position_ids = None
        self.active_branches = []

        # Continuous Batching 状态
        self.recently_completed_branches = []
        self.prefilling_branches = []
        self.pending_prefill_prompts = {}

        # 输出存储
        self.skeleton_output = None
        self.parallel_branches_output = None
        self.instruction_len = None

    def _init_distributed(self):
        """初始化分布式推理组件"""
        if not self.distributed_config or not getattr(self.distributed_config, 'enabled', False):
            return

        self.distributed_prefill_manager = DistributedPrefillManager(
            config=self.distributed_config,
            model=self,
            device=str(self.device),
        )

        # 初始化任务协调器
        self.task_coordinator = DistributedTaskCoordinator(
            comm=self.distributed_prefill_manager.comm,
            logger=self.logger,
            rank=self.distributed_config.rank,
            is_master=(self.distributed_config.rank == 0),
        )

        self._log(f"分布式推理已启用: {self.distributed_config}")

    def cleanup_distributed(self):
        """清理分布式资源"""
        if hasattr(self, 'distributed_prefill_manager') and self.distributed_prefill_manager is not None:
            self.distributed_prefill_manager.cleanup()
            self.distributed_prefill_manager = None

    def _log(self, message: str, level: str = "info"):
        """日志记录"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def _get_model_type(self) -> str:
        """检测模型类型"""
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, '_name_or_path'):
            model_name = self.base_model.config._name_or_path.lower()
        else:
            model_name = ""
        
        if 'qwen' in model_name:
            return 'qwen'
        elif 'vicuna' in model_name:
            return 'vicuna'
        elif 'llama' in model_name:
            return 'llama'
        else:
            return 'other'
    
    # =========================================================================
    # 主入口
    # =========================================================================
    
    @torch.no_grad()
    def generate(
        self,
        task_prompt: str,
        max_new_tokens: int = 512,
        max_parallel: int = 4,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
        use_scheduling: bool = False,
        enable_parallel: bool = True,
    ) -> GenerateResult:
        """
        统一生成入口
        
        Args:
            task_prompt: 任务描述
            max_new_tokens: 最大生成 token 数
            max_parallel: 最大并行分支数
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束
            use_scheduling: 是否使用调度
            enable_parallel: 是否启用并行解码
            
        Returns:
            GenerateResult: 生成结果
        """
        start_time = time.time()
        
        if enable_parallel:
            output_ids, stats = self.generate_specsot(
                task_prompt=task_prompt,
                max_new_tokens=max_new_tokens,
                max_parallel=max_parallel,
                logits_processor=logits_processor,
                use_semantic_constraint=use_semantic_constraint,
                use_scheduling=use_scheduling,
            )
        else:
            output_ids, stats = self.generate_eagle(
                task_prompt=task_prompt,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                use_semantic_constraint=use_semantic_constraint,
            )
        
        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0] if output_ids.dim() > 1 else output_ids)
        
        total_time = time.time() - start_time
        total_tokens = output_ids.shape[-1] if output_ids.numel() > 0 else 0
        
        return GenerateResult(
            output_text=output_text,
            skeleton_text=stats.get('skeleton_text', ''),
            total_tokens=total_tokens,
            total_time=total_time,
            stats=stats,
        )
    
    @torch.no_grad()
    def generate_eagle(
        self,
        task_prompt: str,
        max_new_tokens: int = 512,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        纯投机解码模式（无并行分支）
        
        Args:
            task_prompt: 任务描述
            max_new_tokens: 最大生成 token 数
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束
            
        Returns:
            output_ids, stats
        """
        self._log(f"开始 Eagle 推理模式")
        
        # Phase 1: Prefill
        prefill_result = self._skeleton_prefill(
            task_prompt=task_prompt,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            use_semantic_constraint=use_semantic_constraint,
        )
        
        # Phase 2: Decode Loop
        input_ids = prefill_result.input_ids
        input_len = prefill_result.input_len
        
        draft_tokens = prefill_result.draft_tokens
        retrieve_indices = prefill_result.retrieve_indices
        tree_mask = prefill_result.tree_mask
        tree_position_ids = prefill_result.tree_position_ids
        
        eos_token_id = self.tokenizer.eos_token_id
        total_accept_len = 0
        
        for step in range(max_new_tokens):
            # Decode step
            (
                input_ids, draft_tokens, retrieve_indices,
                tree_mask, tree_position_ids, accept_length,
            ) = self._decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                logits_processor=logits_processor,
            )
            
            total_accept_len += accept_length
            
            # 检查停止条件
            if self._check_stop_single(input_ids, input_len, eos_token_id):
                break
        
        output_ids = input_ids[:, input_len:]
        
        stats = {
            'total_tokens': output_ids.shape[-1],
            'avg_accept_len': total_accept_len / max(step, 1),
        }
        
        return output_ids, stats
    
    @torch.no_grad()
    def generate_specsot(
        self,
        task_prompt: str,
        max_new_tokens: int = 512,
        max_parallel: int = 4,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
        use_scheduling: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        SpecSoT 主流水线
        
        Args:
            task_prompt: 任务描述
            max_new_tokens: 最大生成 token 数
            max_parallel: 最大并行分支数
            logits_processor: logits 处理器
            use_semantic_constraint: 是否使用语义约束
            use_scheduling: 是否使用调度
            
        Returns:
            output_ids, stats
        """
        stats = {
            'skeleton_time': 0.0,
            'scheduling_time': 0.0,
            'parallel_time': 0.0,
            'num_branches': 0,
            'skeleton_text': '',
        }
        
        self._log(f"开始 SpecSoT 流水线: use_scheduling={use_scheduling}")
        
        # =====================================================================
        # Phase 1: Skeleton Prefill
        # =====================================================================
        skeleton_start = time.time()
        
        prefill_result = self._skeleton_prefill(
            task_prompt=task_prompt,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            use_semantic_constraint=use_semantic_constraint,
        )
        
        # =====================================================================
        # Phase 2: Skeleton Decode
        # =====================================================================
        decode_result = self._skeleton_decode(prefill_result)
        
        stats['skeleton_time'] = time.time() - skeleton_start
        stats['skeleton_text'] = decode_result.skeleton_text
        
        # =====================================================================
        # Phase 3: Skeleton Parse
        # =====================================================================
        parse_result = self._skeleton_parse(
            skeleton_text=decode_result.skeleton_text,
            task_prompt=task_prompt,
        )
        
        # 处理 direct/error 模式
        if parse_result.mode != "plan":
            self._log(f"骨架解析: mode={parse_result.mode}")
            return decode_result.skeleton_ids, stats
        
        stats['num_branches'] = parse_result.num_branches
        
        # =====================================================================
        # Phase 4: Schedule
        # =====================================================================
        scheduling_start = time.time()
        
        schedule_result = self._schedule_branches(
            parse_result=parse_result,
            use_scheduling=use_scheduling,
            max_parallel=max_parallel,
        )
        
        stats['scheduling_time'] = time.time() - scheduling_start
        
        # =====================================================================
        # Phase 5: Parallel Prefill
        # =====================================================================
        parallel_start = time.time()
        
        my_branch_ids = schedule_result.my_branch_ids
        if not my_branch_ids:
            self._log("没有分配分支")
            return decode_result.skeleton_ids, stats
        
        branch_prompts = [parse_result.clean_branches[bid] for bid in my_branch_ids]
        
        parallel_prefill_result = self._parallel_prefill(
            prefix_ids=prefill_result.task_input_ids,
            branch_prompts=branch_prompts,
            branch_ids=my_branch_ids,
            max_new_tokens=max_new_tokens,
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
        )
        
        stats['parallel_time'] = time.time() - parallel_start
        stats.update(parallel_decode_result.stats)
        
        # DEBUG: 输出分支结果
        self._log(f"[DEBUG] branch_outputs keys: {list(parallel_decode_result.branch_outputs.keys())}")
        for bid, tokens in parallel_decode_result.branch_outputs.items():
            self._log(f"[DEBUG] branch {bid}: {len(tokens)} tokens")
        
        # =====================================================================
        # Phase 7: Result Merge
        # =====================================================================
        merged_ids = self._merge_results(
            skeleton_ids=decode_result.skeleton_ids,
            tasks=parse_result.tasks,
            branch_outputs=parallel_decode_result.branch_outputs,
        )
        
        self._log(f"[DEBUG] merged_ids shape: {merged_ids.shape}")
        
        return merged_ids, stats
    
    # =========================================================================
    # Phase 实现
    # =========================================================================
    
    def _skeleton_prefill(
        self,
        task_prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[LogitsProcessorList] = None,
        use_semantic_constraint: bool = False,
    ) -> SkeletonPrefillResult:
        """Phase 1: 骨架 Prefill"""
        model_type = self._get_model_type()

        # 准备输入
        input_ids, task_input_ids = prepare_skeleton_input(
            self.tokenizer, task_prompt, model_type, self.device
        )
        input_len = input_ids.shape[1]
        base_prompt_len = task_input_ids.shape[1]

        # 使用 inference engine 执行带初始化的 prefill
        (
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            hidden_states, logits, sample_token
        ) = self.inference_engine.prefill_single_with_init(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )

        # 同步 KV Cache 引用到 generator
        self.past_key_values = self.inference_engine.past_key_values
        self.past_key_values_data = self.inference_engine.past_key_values_data
        self.current_length_data = self.inference_engine.current_length_data

        # 计算 max_kv_len
        tree_buffer = self.eagle_layer.total_tokens if hasattr(self, 'eagle_layer') else 100
        max_kv_len = input_len + max_new_tokens + tree_buffer + 200

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
            hidden_states=hidden_states,
        )
    
    def _skeleton_decode(
        self,
        prefill_result: SkeletonPrefillResult,
        max_steps: int = 200,
    ) -> SkeletonDecodeResult:
        """Phase 2: 骨架解码"""
        decode_start = time.time()
        
        input_ids = prefill_result.input_ids
        input_len = prefill_result.input_len
        
        draft_tokens = prefill_result.draft_tokens
        retrieve_indices = prefill_result.retrieve_indices
        tree_mask = prefill_result.tree_mask
        tree_position_ids = prefill_result.tree_position_ids
        
        eos_token_id = self.tokenizer.eos_token_id
        
        for step in range(max_steps):
            (
                input_ids, draft_tokens, retrieve_indices,
                tree_mask, tree_position_ids, accept_length,
            ) = self._decode_step_single(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
            )
            
            # 检查骨架停止条件
            generated_text = self.tokenizer.decode(input_ids[0, input_len:])
            if self._check_skeleton_stop(generated_text, input_ids, input_len, eos_token_id):
                break
        
        skeleton_ids = input_ids[:, input_len:]
        skeleton_text = self.tokenizer.decode(skeleton_ids[0], skip_special_tokens=False)
        self.skeleton_output = skeleton_ids.clone()
        
        decode_time = time.time() - decode_start
        self._log(f"Skeleton 解码完成: {decode_time:.3f}s")
        
        return SkeletonDecodeResult(
            skeleton_ids=skeleton_ids,
            skeleton_text=skeleton_text,
            decode_time=decode_time,
        )
    
    def _skeleton_parse(
        self,
        skeleton_text: str,
        task_prompt: str,
    ) -> SkeletonParseResult:
        """Phase 3: 骨架解析"""
        model_type = self._get_model_type()
        
        mode, content = parse_skeleton_output(skeleton_text)
        
        if mode == "direct":
            self._log("直接回答模式")
            return SkeletonParseResult(mode="direct")
        
        if mode == "error":
            self._log(f"骨架解析错误: {content}", level="warning")
            return SkeletonParseResult(mode="error", error_msg=str(content))
        
        # mode == "plan"
        tasks = content
        self._log(f"检测到 {len(tasks)} 个并行分支")
        
        clean_branches, instruction_len = prepare_parallel_branches(
            self.tokenizer, tasks, skeleton_text, model_type, task_prompt
        )
        
        self.parallel_branches_output = [list(br) for br in clean_branches]
        self.instruction_len = instruction_len
        
        return SkeletonParseResult(
            mode="plan",
            tasks=tasks,
            clean_branches=clean_branches,
            instruction_len=instruction_len,
        )
    
    def _schedule_branches(
        self,
        parse_result: SkeletonParseResult,
        use_scheduling: bool,
        max_parallel: int,
    ) -> ScheduleResult:
        """Phase 4: 调度"""
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
        
        # 单机模式
        world_size = 1
        device_profiles = [DeviceProfile(device_id=0, max_parallel=max_parallel)]
        
        if use_scheduling:
            scheduler = HeuristicScheduler(use_compute_weight=True)
        else:
            scheduler = SimpleDistributedScheduler(all_parallel=True)
        
        schedule_plan = scheduler.schedule(branch_infos, device_profiles)
        
        my_plan = schedule_plan.get_plan_for_device(0)
        my_branch_ids = my_plan.assigned_branches if my_plan else []
        
        return ScheduleResult(
            schedule_plan=schedule_plan,
            my_branch_ids=my_branch_ids,
            branch_infos=branch_infos,
        )
    
    def _parallel_prefill(
        self,
        prefix_ids: torch.Tensor,
        branch_prompts: List[List[int]],
        branch_ids: List[int],
        max_new_tokens: int,
    ) -> ParallelPrefillResult:
        """
        Phase 5: 并行 Prefill

        关键设计：
        - 复用 prefix 的 KV Cache（不重新计算）
        - 只 prefill 各分支的 parallel prompt
        - 根据 use_bim_mode 选择 BIM 或 Batching 模式
        """
        prefix_len = prefix_ids.shape[1]
        num_branches = len(branch_prompts)

        # 初始化状态管理器
        if self.state_manager is None:
            self.state_manager = BranchStateManager(
                max_seq_len=4096,
                max_branches=num_branches,
                device=self.device,
                use_bim_mode=self.use_bim_mode,
            )
        else:
            self.state_manager.reset()

        self.state_manager.init_prefix(prefix_len)
        self.active_branches = list(branch_ids)

        # 调用 inference engine 执行并行 prefill
        result = self.inference_engine.prefill_parallel_branches(
            prefix_len=prefix_len,
            branch_prompts=branch_prompts,
            branch_ids=branch_ids,
            max_new_tokens=max_new_tokens,
        )

        # 更新状态管理器
        branch_lengths = result.get('branch_lengths', [len(p) for p in branch_prompts])
        branch_begins = result.get('branch_begins', [])

        if self.use_bim_mode and branch_begins:
            for branch_id, begin, length in zip(branch_ids, branch_begins, branch_lengths):
                self.state_manager.add_branch(
                    branch_id, length, start_position=prefix_len + begin
                )

        # 保存 branch_index_map 引用
        self.branch_index_map = result.get('branch_index_map')

        return ParallelPrefillResult(
            input_ids=result['input_ids'],
            draft_tokens=result['draft_tokens'],
            retrieve_indices=result['retrieve_indices'],
            tree_mask=result['tree_mask'],
            tree_position_ids=result['tree_position_ids'],
            tips_indices=result.get('tips_indices'),
            branch_begins=branch_begins,
            branch_lengths=branch_lengths,
            branch_index_map=result.get('branch_index_map'),
            position_ids=result.get('position_ids'),
        )
    
    def _parallel_decode(
        self,
        prefill_result: ParallelPrefillResult,
        schedule_result: ScheduleResult,
        clean_branches: List[List[int]],
        max_new_tokens: int,
        max_kv_len: int,
        prefix_len: int,
        use_scheduling: bool,
    ) -> ParallelDecodeResult:
        """
        Phase 6: 并行解码
        
        执行多分支的 Decode Loop：Draft -> Verify -> Update
        """
        decode_start = time.time()
        
        draft_tokens = prefill_result.draft_tokens
        retrieve_indices = prefill_result.retrieve_indices
        tree_mask = prefill_result.tree_mask
        tree_position_ids = prefill_result.tree_position_ids
        branch_index_map = prefill_result.branch_index_map
        
        branch_outputs = {bid: [] for bid in schedule_result.my_branch_ids}
        tokens_per_branch = self.eagle_layer.total_tokens + 1
        eos_token_id = self.tokenizer.eos_token_id
        
        total_accept_len = 0
        
        # DEBUG: 初始状态
        self._log(f"[DEBUG] _parallel_decode 开始: active_branches={self.active_branches}")
        self._log(f"[DEBUG] draft_tokens shape: {draft_tokens.shape}")
        self._log(f"[DEBUG] max_new_tokens={max_new_tokens}, max_kv_len={max_kv_len}")
        
        for step in range(max_new_tokens):
            # 检查停止条件
            current_len = self.current_length_data[0].item()
            if current_len + tokens_per_branch * len(self.active_branches) > max_kv_len:
                self._log(f"KV Cache 限制，提前结束", level="warning")
                break
            
            if not self.active_branches:
                self._log(f"[DEBUG] 无活跃分支，step={step}")
                break
            
            # DEBUG: step开始
            if step < 5:
                self._log(f"[DEBUG] step={step}, active_branches={self.active_branches}, current_len={current_len}")
            
            # 执行解码步骤
            (
                draft_tokens, retrieve_indices, tree_mask,
                tree_position_ids, accept_lengths, all_finished,
            ) = self.inference_engine.decode_step_parallel(
                input_ids=prefill_result.input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_mask=tree_mask,
                tree_position_ids=tree_position_ids,
                branch_index_map=branch_index_map,
                active_branches=self.active_branches,
                prefix_len=prefix_len,
            )
            
            # DEBUG: step结果
            if step < 5:
                self._log(f"[DEBUG] step={step}, accept_lengths={accept_lengths.tolist()}, all_finished={all_finished}")
            
            total_accept_len += accept_lengths.sum().item()
            
            # 更新分支输出和检查完成状态
            finished_branches = []
            for i, branch_id in enumerate(self.active_branches):
                accept_len = accept_lengths[i].item()
                if accept_len > 0:
                    # 记录接受的 tokens
                    accepted = draft_tokens[i, :accept_len + 1].tolist()
                    branch_outputs[branch_id].extend(accepted)
                    
                    # 检查是否生成了 EOS
                    if eos_token_id in accepted:
                        finished_branches.append(branch_id)
            
            # 移除完成的分支
            for bid in finished_branches:
                self.active_branches.remove(bid)
                self.state_manager.finish_branch(bid)
                self._log(f"分支 {bid} 完成")
            
            if all_finished or not self.active_branches:
                self._log("所有分支解码完成")
                break
        
        decode_time = time.time() - decode_start
        num_steps = max(step, 1)
        
        return ParallelDecodeResult(
            branch_outputs=branch_outputs,
            decode_time=decode_time,
            stats={
                'avg_accept_len': total_accept_len / num_steps,
                'num_steps': num_steps,
            },
        )
    
    def _merge_results(
        self,
        skeleton_ids: torch.Tensor,
        tasks: List[Dict],
        branch_outputs: Optional[Dict[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Phase 7: 结果合并
        
        将骨架和各分支的并行生成结果合并成最终输出。
        
        Args:
            skeleton_ids: 骨架 token ids [1, seq_len]
            tasks: 解析出的任务列表，包含位置信息
            branch_outputs: 各分支的输出 {branch_id: [token_ids]}
            
        Returns:
            merged_ids: 合并后的 token ids
        """
        if branch_outputs is None or not branch_outputs:
            return skeleton_ids
        
        # 将骨架转换为列表便于操作
        skeleton_list = skeleton_ids[0].tolist()
        
        # 收集所有分支输出
        all_branch_tokens = []
        for branch_id in sorted(branch_outputs.keys()):
            branch_tokens = branch_outputs[branch_id]
            if branch_tokens:
                all_branch_tokens.append(branch_tokens)
        
        if not all_branch_tokens:
            return skeleton_ids
        
        # 合并策略：骨架 + 分支内容
        # 找到 [END] 标记的位置，在其后插入分支内容
        # 如果没有 [END]，直接追加在末尾
        
        # 查找特殊标记
        end_tokens = self.tokenizer.encode('[END]', add_special_tokens=False)
        
        # 寻找 [END] 在骨架中的位置
        end_pos = -1
        for i in range(len(skeleton_list) - len(end_tokens) + 1):
            if skeleton_list[i:i+len(end_tokens)] == end_tokens:
                end_pos = i + len(end_tokens)
                break
        
        # 构建合并结果
        merged_list = []
        
        if end_pos > 0:
            # 在 [END] 后插入分支内容
            merged_list = skeleton_list[:end_pos]
            
            # 添加换行
            newline_token = self.tokenizer.encode('\n\n', add_special_tokens=False)
            merged_list.extend(newline_token)
            
            # 添加各分支内容
            for i, branch_tokens in enumerate(all_branch_tokens):
                if i > 0:
                    merged_list.extend(newline_token)
                merged_list.extend(branch_tokens)
        else:
            # 没有 [END]，直接追加
            merged_list = skeleton_list.copy()
            newline_token = self.tokenizer.encode('\n\n', add_special_tokens=False)
            merged_list.extend(newline_token)
            for i, branch_tokens in enumerate(all_branch_tokens):
                if i > 0:
                    merged_list.extend(newline_token)
                merged_list.extend(branch_tokens)
        
        # 转换回 tensor
        merged_ids = torch.tensor([merged_list], dtype=skeleton_ids.dtype, device=skeleton_ids.device)
        
        return merged_ids
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _decode_step_single(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        retrieve_indices: torch.Tensor,
        tree_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """单步解码"""
        return self.inference_engine.decode_step_single(
            input_ids=input_ids,
            draft_tokens=draft_tokens,
            retrieve_indices=retrieve_indices,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
            logits_processor=logits_processor,
        )
    
    def _check_stop_single(
        self,
        input_ids: torch.Tensor,
        input_len: int,
        eos_token_id: int,
    ) -> bool:
        """检查单序列停止条件"""
        if eos_token_id in input_ids[0, input_len:]:
            return True
        return False
    
    def _check_skeleton_stop(
        self,
        generated_text: str,
        input_ids: torch.Tensor,
        input_len: int,
        eos_token_id: int,
    ) -> bool:
        """检查骨架停止条件"""
        # 检查 EOS
        if eos_token_id in input_ids[0, input_len:]:
            return True
        
        # 检查骨架结束标记
        if '[END]' in generated_text or '</plan>' in generated_text:
            return True

        return False

    # =========================================================================
    # 并行解码辅助方法 (从 specsot_model.py 移植)
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
        
        关键设计：
        =========
        在 Skeleton 阶段，已经 prefill 了 prefix (system prompt) 的 KV cache。
        在 Parallel 阶段，我们需要：
          1. **复用** prefix 的 KV cache（不重新计算）
          2. 只 prefill **各分支的 parallel prompt**（新的 user turn 部分）
        
        因此，返回的 input_ids **只包含各分支的 prompt，不包含 prefix**。
        
        核心步骤：
        1. 重置 KV Cache 长度到 prefix_len（保留 prefix cache）
        2. 复制 Eagle Layer KV Cache 为多分支
        3. 构建 BIM: 追踪每个 token 的分支归属（prefix 标记为 -1）
        4. 打包输入: **只包含各分支的 prompt**（不包含 prefix）
        5. 位置编码: 各分支从 prefix_len 开始独立计数
        
        Args:
            prefix_ids: 共享前缀 [1, prefix_len]（仅用于获取长度）
            branches_prompts: 各分支的 prompt token 列表（不包含 prefix）
            max_new_tokens: 每个分支最大生成长度
            branch_ids: 分支 ID 列表（可选，用于 Continuous Batching）
            
        Returns:
            input_ids: 打包后的输入 [1, branches_total_len]（仅分支 prompts）
            tips_indices: 各分支 tip 位置（相对于 prefix_len 的偏移）
            branch_begins: 各分支起始位置（相对于 prefix_len 的偏移）
            branch_lengths: 各分支初始长度
            draft_input_ids: Draft 模型的输入 [num_para, max_len]
        """
        device = self.device
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
            # 获取第一层的 KV Cache
            key_cache, value_cache = self.eagle_layer.draft_past_key_values[0]
            # 使用 KVCache 类的 .data 属性统一访问
            k_draft = key_cache.data[:, :, :key_cache.shape[2], :]
            v_draft = value_cache.data[:, :, :value_cache.shape[2], :]
            # 复制 prefix 部分并扩展到多分支
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
        # 关键：只构建分支部分，不包含 prefix
        flat_branch_ids = []
        # BIM: prefix 部分标记为 -1（共享），分支部分标记为对应的 branch_id
        branch_index_list = [-1] * prefix_len
        
        tips_indices = []
        branch_begins = []
        branch_lengths = []
        # 位置编码：从 prefix_len 开始
        pos_ids_list = []
        
        draft_branch_list = []
        draft_pos_list = []
        
        # 当前偏移从 prefix_len 开始（相对于整个序列的绝对位置）
        current_offset = prefix_len
        for i, (br, bid) in enumerate(zip(branches_prompts, branch_ids)):
            curr_len = len(br)
            branch_begins.append(current_offset - prefix_len)  # 相对于分支起始的偏移
            flat_branch_ids.extend(br)
            branch_index_list.extend([bid] * curr_len)
            branch_lengths.append(curr_len)
            current_offset += curr_len
            tips_indices.append(current_offset - prefix_len - 1)  # 相对于分支起始的偏移
            
            # 位置编码: 每个分支从 prefix_len 开始独立计数
            curr_pos = list(range(prefix_len, prefix_len + curr_len))
            pos_ids_list.extend(curr_pos)
            
            draft_branch_list.append(torch.tensor(br, device=device, dtype=torch.long))
            draft_pos_list.append(torch.tensor(curr_pos, device=device, dtype=torch.long))
        
        # 构建 Tensor
        # 关键修改：input_ids 只包含分支 prompts，不包含 prefix
        branches_tensor = torch.tensor([flat_branch_ids], device=device, dtype=torch.long)
        input_ids = branches_tensor  # 不再拼接 prefix_ids
        tips_indices = torch.tensor(tips_indices, device=device)
        self.full_position_ids = torch.tensor(pos_ids_list, device=device)
        
        # 初始化 BIM（包含 prefix 的标记，用于 attention mask 构建）
        total_capacity = prefix_len + input_ids.shape[1] + max_new_tokens + 128
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
    
    def _add_branches_to_active(
        self,
        new_branch_ids: List[int],
        all_branches_prompts: List[List[int]],
        prefix_len: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
            
        # 将新分支标记为 prefilling 状态
        self.prefilling_branches = list(new_branch_ids)
        
        # 缓存新分支的 prompt tokens
        for bid in new_branch_ids:
            self.pending_prefill_prompts[bid] = all_branches_prompts[bid]
        
        self._log(
            f"新分支加入 prefilling 队列: {new_branch_ids}, "
            f"当前活跃分支: {self.active_branches}",
            level="debug"
        )
        
        # 返回 None，实际的处理在下一轮 decode_step 中完成
        return None, None, None, None
