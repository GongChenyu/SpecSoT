# SpecSoT 系统架构设计文档 (v5.0)

> **文档更新日期**: 2026-02-01  
> **架构版本**: v5.0 - 重构版本  
> **重构重点**: 引擎与组件分离 + 统一状态管理 + 双模式推理 + 异构设备支持

---

## 一、项目概述

### 1.1 核心理念

SpecSoT (Speculative Decoding + Skeleton-of-Thought) 是一个将**语义并行**与**投机解码**结合的高性能推理系统：

- **Skeleton-of-Thought (SoT)**: 先生成回答骨架（规划），再并行填充各分支内容
- **Speculative Decoding (SD)**: 使用轻量级 Draft Model (Eagle) 快速生成候选 token，再由 Base Model 验证

### 1.2 核心推理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SpecSoT 推理流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Skeleton Generation (骨架生成)                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Prefill   │ --> │ SD Decode   │ --> │   Parse     │                   │
│  │  (初始化)   │     │ (投机解码)   │     │ (解析骨架)  │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                │                            │
│                                                v                            │
│  Phase 2: Parallel Branch Decoding (并行分支解码)                           │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  Schedule   │ --> │  Prefill    │ --> │  SD Decode  │                   │
│  │ (分支调度)  │     │ (并行Prefill)│     │(多分支并行) │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                │                            │
│                                                v                            │
│  Phase 3: Result Merge (结果合并)                                           │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │   合并骨架 + 各分支输出 --> 最终响应                  │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 投机解码循环 (SD Decode Loop)

```
每个 Decode Step:
┌──────────────────────────────────────────────────────────────┐
│  Draft Phase        Verify Phase        Update Phase        │
│  ┌──────────┐      ┌──────────┐        ┌──────────┐        │
│  │  Eagle   │ -->  │  Base    │  -->   │  State   │        │
│  │  Layer   │      │  Model   │        │  Update  │        │
│  │(生成Tree)│      │ (验证)   │        │(接受Token)│        │
│  └──────────┘      └──────────┘        └──────────┘        │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、架构概览

新架构采用**引擎-组件分离设计**，将系统划分为：

- **Engine (引擎层)**: 流程控制 + 分布式协调 (Master/Worker)
- **Generate (生成层)**: SpecSoT 主生成流程编排
- **Core (核心组件层)**: 投机解码逻辑 (Drafter) + 统一状态管理 + 推理引擎
- **Models (模型层)**: Base Model + Draft Model (Eagle)
- **Scheduling (调度层)**: 分支调度策略
- **Processing (处理层)**: Prompt 模板 + 骨架解析 + Logits 处理
- **Distributed (分布式层)**: 通信管理 + 分布式配置
- **Config (配置层)**: 设备配置 + 系统配置

### 2.1 架构层次图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer (可选)                               │
│                    server.py / protocol.py                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Engine Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     engine/                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │    master    │  │    worker    │  │       utils              │   │   │
│  │  │   (主控)     │  │   (计算节点)  │  │ (DeviceConfig, 端口清理) │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Generate Layer                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       generate/                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │              SpecSoTOrchestrator (生成编排器)                   │  │   │
│  │  │  ├── generate()              # 统一生成入口                    │  │   │
│  │  │  ├── generate_eagle()        # 纯投机解码模式                  │  │   │
│  │  │  ├── generate_specsot()      # SpecSoT 主流水线                │  │   │
│  │  │  │   ├── _skeleton_prefill()   # Phase 1                       │  │   │
│  │  │  │   ├── _skeleton_decode()    # Phase 2                       │  │   │
│  │  │  │   ├── _skeleton_parse()     # Phase 3                       │  │   │
│  │  │  │   ├── _schedule_branches()  # Phase 4                       │  │   │
│  │  │  │   ├── _parallel_prefill()   # Phase 5                       │  │   │
│  │  │  │   ├── _parallel_decode()    # Phase 6                       │  │   │
│  │  │  │   └── _merge_results()      # Phase 7                       │  │   │
│  │  │  └── datatypes.py            # 流水线数据结构                  │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Core Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       core/                                          │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │     drafter      │  │  state_manager   │  │ inference_engine │   │   │
│  │  │  (Draft Tree生成) │  │  (统一状态管理)   │  │  (推理引擎)      │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                         │   │
│  │  │    kv_cache      │  │   eval_utils     │                         │   │
│  │  │   (KV Cache)     │  │   (评估工具)      │                         │   │
│  │  └──────────────────┘  └──────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Models Layer                                      │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐    │
│  │        modeling/           │  │         modeling_draft/            │    │
│  │  modeling_llama_kv.py      │  │   eagle_base.py                    │    │
│  │  modeling_qwen2_kv.py      │  │   eagle2.py                        │    │
│  │  modeling_qwen3_kv.py      │  │   eagle3.py                        │    │
│  │  modeling_mixtral_kv.py    │  │   configs.py                       │    │
│  └────────────────────────────┘  └────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│         Scheduling          │     Processing      │     Distributed        │
│  ┌───────────────────────┐  │ ┌─────────────────┐ │ ┌───────────────────┐  │
│  │  scheduling/          │  │ │ processing/     │ │ │ distributed/      │  │
│  │  branch_scheduler     │  │ │ prompts         │ │ │ comm_manager      │  │
│  │  branch_manager       │  │ │ logits_processor│ │ │ task_coordinator  │  │
│  │  branch_config        │  │ │                 │ │ │ prefill_manager   │  │
│  └───────────────────────┘  │ └─────────────────┘ │ └───────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Config Layer                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  config/                                                             │   │
│  │  ├── device_config.py      # 设备配置（异构设备支持）                 │   │
│  │  ├── distributed_config.py # 分布式配置                              │   │
│  │  └── system_config.py      # 系统配置                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Utils Layer                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  utils/                                                              │   │
│  │  ├── gpu_monitor.py        # GPUMemoryMonitor                        │   │
│  │  ├── logging_utils.py      # 日志工具                                │   │
│  │  └── tensor_utils.py       # 通用张量工具                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计理念

1. **引擎-组件分离**: Engine 负责流程编排和分布式协调，Generate 负责生成逻辑，Core 负责推理核心组件
2. **Drafter 独立**: 投机解码的 Draft Tree 生成逻辑从 Model 中独立出来，作为核心组件
3. **统一状态管理**: BranchStateManager 统一管理 BIM、Position IDs、分支状态，支持双模式切换
4. **异构设备支持**: 通过设备配置文件支持不同算力、显存的设备
5. **双模式推理**: 支持 Batching 模式和 In-One-Sequence (BIM) 模式切换

---

## 三、新架构文件结构

```
SpecSoT/
│
├── main.py                          # 统一启动入口
│
├── api/                             # [可选] API 服务层
│   ├── __init__.py
│   ├── server.py                    # FastAPI/gRPC 服务入口
│   └── protocol.py                  # 请求/响应协议定义
│
├── config/                          # 【配置层】系统配置
│   ├── __init__.py
│   ├── device_config.py             # 设备配置（异构设备支持）
│   │   ├── DeviceProfile            # 设备能力描述
│   │   │   ├── device_id: int       # 设备 ID
│   │   │   ├── ip: str              # 设备 IP 地址
│   │   │   ├── gpu_id: int          # GPU 索引
│   │   │   ├── compute_capability   # 算力指标 (TFLOPS)
│   │   │   ├── memory_capacity      # 显存容量 (GB)
│   │   │   ├── memory_bandwidth     # 显存带宽 (GB/s)
│   │   │   ├── roofline_inflection  # Roofline 拐点
│   │   │   └── max_parallel         # 最大并行分支数
│   │   ├── parse_devices()          # 解析设备列表 "ip#gpu_id,..."
│   │   ├── load_device_profiles()   # 从配置文件加载设备参数
│   │   └── DeviceConfigValidator    # 设备配置验证器
│   │
│   ├── distributed_config.py        # 分布式配置
│   │   └── DistributedConfig        # 分布式推理配置
│   │       ├── enabled: bool        # 是否启用分布式
│   │       ├── rank: int            # 当前 rank
│   │       ├── world_size: int      # 总设备数（从设备列表计算）
│   │       ├── layer_splits         # 层拆分策略
│   │       ├── base_port: int       # 基础通信端口
│   │       └── comm_mode: str       # 通信模式 (p2p/ring)
│   │
│   └── system_config.py             # 系统配置
│       └── SystemConfig
│           ├── enable_parallel      # 是否启用并行解码
│           ├── use_scheduling       # 是否使用调度
│           ├── use_bim_mode         # 是否使用 BIM 模式
│           ├── max_parallel         # 最大并行分支数
│           └── max_new_tokens       # 最大生成 token 数
│
├── engine/                          # 【引擎层】流程控制 + 分布式协调
│   ├── __init__.py
│   ├── master.py                    # Master 主控引擎
│   │   ├── MasterConfig             # Master 配置
│   │   └── MasterEngine             # 启动/管理 Worker 子进程
│   │       ├── run()                # 主流程 (自动选择单机/分布式)
│   │       ├── _run_single()        # 单机模式入口
│   │       └── _run_distributed()   # 分布式模式入口
│   │
│   ├── worker.py                    # Worker 计算节点引擎
│   │   └── WorkerEngine             # Worker 主类
│   │       ├── run()                # Worker 主循环
│   │       ├── _run_inference_loop()# 推理循环
│   │       └── _run_single_inference()  # 单样本推理
│   │
│   └── utils.py                     # 引擎工具函数
│       ├── cleanup_ports()          # 清理端口
│       └── str2bool()               # 字符串转布尔
│
├── generate/                        # 【生成层】SpecSoT 主生成逻辑
│   ├── __init__.py
│   ├── orchestrator.py              # 【核心】SpecSoT 生成编排器
│   │   └── SpecSoTOrchestrator      # 协调整个 SpecSoT 流水线
│   │       ├── generate()           # 统一生成入口
│   │       ├── generate_eagle()     # 纯投机解码模式
│   │       └── generate_specsot()   # SpecSoT 主流水线
│   │           ├── _skeleton_prefill()   # Phase 1: Skeleton Prefill
│   │           ├── _skeleton_decode()    # Phase 2: Skeleton Decode
│   │           ├── _skeleton_parse()     # Phase 3: Skeleton Parse
│   │           ├── _schedule_branches()  # Phase 4: Schedule
│   │           ├── _parallel_prefill()   # Phase 5: Parallel Prefill
│   │           ├── _parallel_decode()    # Phase 6: Parallel Decode
│   │           └── _merge_results()      # Phase 7: Result Merge
│   │
│   └── datatypes.py                 # 流水线数据结构
│       ├── SkeletonPrefillResult    # 骨架 Prefill 结果
│       ├── SkeletonDecodeResult     # 骨架解码结果
│       ├── SkeletonParseResult      # 骨架解析结果
│       ├── ScheduleResult           # 调度结果
│       ├── ParallelPrefillResult    # 并行 Prefill 结果
│       └── ParallelDecodeResult     # 并行解码结果
│
├── core/                            # 【核心组件层】
│   ├── __init__.py
│   │
│   ├── drafter.py                   # 【关键】Draft Tree 生成器
│   │   └── Drafter                  # 独立于 Eagle Layer 的 Draft 逻辑
│   │       ├── generate_draft_tree()     # 主入口
│   │       ├── _expand_root()            # Phase 1: 根节点扩展
│   │       ├── _grow_tree()              # Phase 2: 树生长
│   │       ├── _post_process_tree()      # Phase 3: 后处理
│   │       └── _apply_vocab_mapping()    # 词表映射 (Eagle2/3 兼容)
│   │
│   ├── state_manager.py             # 【关键】统一状态管理器
│   │   └── BranchStateManager       # 统一 BIM 状态管理
│   │       ├── 状态变量:
│   │       │   ├── bim: torch.Tensor           # Branch Index Map
│   │       │   ├── position_ids: torch.Tensor  # 位置编码
│   │       │   ├── active_branches: List[int]  # 活跃分支
│   │       │   ├── branch_tips: Dict[int, int] # 分支 tip 位置
│   │       │   └── branch_outputs: Dict        # 分支输出
│   │       ├── 核心方法:
│   │       │   ├── reset()                  # 重置状态
│   │       │   ├── init_prefix()            # 初始化共享前缀
│   │       │   ├── add_branch()             # 添加分支
│   │       │   ├── extend_branch()          # 扩展分支
│   │       │   ├── get_attention_mask()     # 构建 attention mask
│   │       │   └── update_after_verify()    # Verify 后更新
│   │       ├── BIM 工具函数 (内置):
│   │       │   ├── build_bim_attention_mask()
│   │       │   ├── find_branch_tips()
│   │       │   └── flatten_branches_to_sequence()
│   │       └── 模式支持:
│   │           ├── use_bim_mode: bool       # True: In-One-Sequence
│   │           └── batching_mode            # False: Batching
│   │
│   ├── kv_cache.py                  # KV Cache 定义和管理
│   │   ├── KVCache                  # KV Cache 类
│   │   │   ├── data                 # 底层数据张量
│   │   │   ├── current_length       # 当前长度
│   │   │   ├── copy()               # 索引复制
│   │   │   ├── cat()                # 拼接
│   │   │   └── restore_from_tensor()# 从张量恢复
│   │   ├── initialize_past_key_values()  # 初始化 KV Cache
│   │   └── reset_past_key_values()  # 重置 KV Cache
│   │
│   ├── inference_engine.py          # 推理引擎 (prefill/decode/verify 执行)
│   │   └── InferenceEngine          # 底层推理执行器
│   │       ├── prefill_single()     # 单序列 Prefill
│   │       ├── prefill_parallel()   # 并行 Prefill
│   │       ├── decode_step_single() # 单序列 Decode Step
│   │       ├── decode_step_parallel()    # 并行 Decode Step
│   │       ├── verify_step_single()      # 单序列 Verify
│   │       ├── verify_step_parallel()    # 并行 Verify
│   │       ├── update_state_single()     # 单序列 State Update
│   │       └── update_state_parallel()   # 并行 State Update
│   │
│   └── eval_utils.py                # 评估工具
│       ├── evaluate_single()        # 单序列评估
│       ├── evaluate_parallel()      # 并行评估
│       ├── greedy_sampling()        # Greedy 采样
│       ├── rejection_sampling()     # Rejection Sampling
│       └── logits_sampling()        # 核心采样逻辑
│
├── models/                          # 【模型层】Base Model + Draft Model
│   ├── __init__.py
│   │
│   ├── modeling/                    # Base Model (目标大模型)
│   │   ├── __init__.py
│   │   ├── modeling_llama_kv.py     # LLaMA 模型 (带 KV Cache)
│   │   ├── modeling_qwen2_kv.py     # Qwen2 模型
│   │   ├── modeling_qwen3_kv.py     # Qwen3 模型
│   │   ├── modeling_mixtral_kv.py   # Mixtral MoE 模型
│   │   └── global_recorder.py       # 全局记录器
│   │
│   └── modeling_draft/              # Draft Model (Eagle Layer)
│       ├── __init__.py
│       ├── configs.py               # Draft 模型配置
│       │   └── DraftConfig          # top_k, depth, total_tokens 等
│       ├── eagle_base.py            # Eagle 基类
│       │   └── EagleBase            # Eagle Layer 公共逻辑
│       │       ├── forward()        # 单步前向
│       │       ├── init_kv_cache()  # KV Cache 初始化
│       │       └── _prepare_attention_mask()  # 准备 attention mask
│       ├── eagle2.py                # Eagle-2 实现
│       │   └── Eagle2               # 多层 Decoder Stack
│       └── eagle3.py                # Eagle-3 实现
│           └── Eagle3               # 单层 Decoder
│
├── scheduling/                      # 【调度层】分支调度策略
│   ├── __init__.py
│   ├── branch_config.py             # 调度数据结构
│   │   ├── BranchInfo               # 分支信息
│   │   ├── DeviceExecutionPlan      # 单设备执行计划
│   │   └── SchedulePlan             # 全局调度计划
│   │
│   ├── branch_scheduler.py          # 调度算法
│   │   ├── BranchScheduler (ABC)    # 调度器基类
│   │   ├── HeuristicScheduler       # 启发式调度 (LPT + 轮询)
│   │   └── SimpleDistributedScheduler  # 简单分布式调度
│   │
│   └── branch_manager.py            # 分支执行管理
│       └── BranchExecutionManager   # 管理分支执行状态
│           ├── add_batch()          # 添加执行批次
│           ├── handle_completed_branches()  # 处理完成分支
│           └── get_branches_to_add()  # 获取待加入分支
│
├── processing/                      # 【处理层】Prompt 和 Logits 处理
│   ├── __init__.py
│   ├── prompts.py                   # Prompt 模板和解析
│   │   ├── prepare_skeleton_input() # 准备骨架输入
│   │   ├── parse_skeleton_output()  # 解析骨架输出
│   │   └── prepare_parallel_branches()  # 准备并行分支
│   │
│   └── logits_processor.py          # Logits 处理器
│       ├── SemanticLogitsProcessor  # 语义约束处理器
│       └── VocabScanner             # 词表扫描器
│
├── distributed/                     # 【分布式层】通信管理
│   ├── __init__.py
│   ├── distributed_prefill.py       # 分布式 Prefill 管理器
│   │   └── DistributedPrefillManager
│   │       ├── prefill_single_distributed()
│   │       ├── _split_into_chunks()
│   │       └── _process_received_caches_nonblocking()
│   │
│   └── communication/               # 通信子模块
│       ├── __init__.py
│       ├── base_comm_manager.py     # 通信管理器基类
│       ├── comm_manager.py          # 通信管理器
│       ├── p2p_comm_manager.py      # P2P 通信
│       ├── ring_comm_manager.py     # Ring 通信
│       ├── task_coordinator.py      # 任务协调器
│       └── comm_utils.py            # 通信工具
│
└── utils/                           # 【工具层】通用工具
    ├── __init__.py
    ├── gpu_monitor.py               # GPU 显存监控器
    │   └── GPUMemoryMonitor
    │       ├── __enter__()
    │       ├── __exit__()
    │       └── peak_usage
    │
    ├── logging_utils.py             # 日志工具
    │   ├── get_unified_logger()
    │   ├── FlushingStreamHandler
    │   └── FlushingFileHandler
    │
    └── tensor_utils.py              # 张量工具
        ├── stack_with_left_padding()
        ├── generate_candidates()
        └── pad_path()
```

---

## 四、代码映射关系

### 4.1 现有文件到新架构的映射

| 现有文件 | 新位置 | 说明 |
|---------|--------|------|
| `specsot_model.py` | `generate/orchestrator.py` + `core/inference_engine.py` | 生成逻辑拆分 |
| `specsot_model.py` (数据结构) | `generate/datatypes.py` | 流水线数据结构 |
| `specsot_model.py` (prefill/verify/update) | `core/inference_engine.py` | 推理引擎 |
| `utils.py` (evaluate_*) | `core/eval_utils.py` | 评估工具 |
| `utils.py` (mask building) | `core/state_manager.py` | BIM 相关移入状态管理 |
| `utils.py` (stop conditions) | `generate/orchestrator.py` | 停止条件内联 |
| `utils.py` (merge_outputs) | `generate/orchestrator.py` | 合并逻辑内联 |
| `utils.py` (tensor utils) | `utils/tensor_utils.py` | 通用张量工具 |
| `utils.py` (logits processor) | `processing/logits_processor.py` | 已存在 |
| `kv_cache.py` | `core/kv_cache.py` | 保持不变 |
| `prompts.py` | `processing/prompts.py` | 保持不变 |
| `logits_processor.py` | `processing/logits_processor.py` | 保持不变 |
| `distributed_config.py` | `config/distributed_config.py` | 配置层 |
| `distributed_prefill.py` | `distributed/distributed_prefill.py` | 保持不变 |
| `communication/*` | `distributed/communication/*` | 保持不变 |
| `scheduling/*` | `scheduling/*` | 保持不变 |
| `modeling/*` | `models/modeling/*` | 保持文件名不变 |
| `modeling_draft/*` | `models/modeling_draft/*` | 保持不变 |
| `engine/worker.py` (GPUMemoryMonitor) | `utils/gpu_monitor.py` | 移动 |
| `engine/utils.py` (DeviceConfig) | `config/device_config.py` | 设备配置 |

### 4.2 关键函数映射

| 现有函数 | 所在文件 | 新位置 | 说明 |
|---------|----------|--------|------|
| `SpecSoTModel.generate_specsot()` | specsot_model.py:1449-1641 | `generate/orchestrator.py` | 主生成流程 |
| `SpecSoTModel.prefill_single()` | specsot_model.py:1776-1836 | `core/inference_engine.py` | 推理引擎 |
| `SpecSoTModel.prefill_parallel()` | specsot_model.py:1848-1959 | `core/inference_engine.py` | 推理引擎 |
| `SpecSoTModel.decode_step_single()` | specsot_model.py:1976-2047 | `core/inference_engine.py` | 推理引擎 |
| `SpecSoTModel.verify_step_parallel()` | specsot_model.py:2613-2716 | `core/inference_engine.py` | 推理引擎 |
| `SpecSoTModel.update_state_parallel()` | specsot_model.py:2789-2963 | `core/inference_engine.py` | 推理引擎 |
| `_prepare_parallel_batch()` | specsot_model.py:1107-1236 | `core/state_manager.py` | BIM 初始化 |
| `evaluate_single()` | utils.py:615-660 | `core/eval_utils.py` | 评估工具 |
| `evaluate_parallel()` | utils.py:669-722 | `core/eval_utils.py` | 评估工具 |
| `greedy_sampling()` | utils.py:291-411 | `core/eval_utils.py` | 采样算法 |
| `rejection_sampling()` | utils.py:413-513 | `core/eval_utils.py` | 采样算法 |
| `build_parallel_prefill_mask()` | utils.py:102-154 | `core/state_manager.py` | BIM mask |
| `build_continuous_decode_mask()` | utils.py:157-251 | `core/state_manager.py` | BIM mask |
| `GPUMemoryMonitor` | engine/worker.py:29-63 | `utils/gpu_monitor.py` | 显存监控 |
| `parse_devices()` | engine/utils.py:24-47 | `config/device_config.py` | 设备解析 |

---

## 五、设备管理与异构支持

### 5.1 设备输入格式

设备使用 `ip#gpu_id` 格式指定，支持多设备：

```bash
# 单设备
--devices "127.0.0.1#0"

# 多设备（同机）
--devices "127.0.0.1#0,127.0.0.1#1,127.0.0.1#2"

# 多设备（跨机）
--devices "192.168.1.100#0,192.168.1.101#0,192.168.1.102#0"
```

### 5.2 设备配置文件

异构设备参数通过 JSON 配置文件指定：

```json
// config/devices.json
{
  "devices": [
    {
      "ip": "192.168.1.100",
      "gpu_id": 0,
      "compute_capability": 19.5,
      "memory_capacity": 80,
      "memory_bandwidth": 2000,
      "max_parallel": 4
    },
    {
      "ip": "192.168.1.101",
      "gpu_id": 0,
      "compute_capability": 31.2,
      "memory_capacity": 24,
      "memory_bandwidth": 600,
      "max_parallel": 2
    }
  ]
}
```

### 5.3 设备配置验证

```python
class DeviceConfigValidator:
    """设备配置验证器"""
    
    @staticmethod
    def validate(devices_str: str, enable_parallel: bool, distributed: bool) -> Tuple[bool, str]:
        """
        验证设备配置
        
        规则：
        1. 如果 enable_parallel=False，则 distributed 必须为 False
        2. 如果 distributed=True，设备数必须 >= 2
        3. world_size 从设备列表自动计算，无需输入
        """
        device_list = parse_devices(devices_str)
        world_size = len(device_list)
        
        if not enable_parallel and distributed:
            return False, "enable_parallel=False 时不支持分布式模式"
        
        if distributed and world_size < 2:
            return False, f"分布式模式需要至少 2 个设备，当前: {world_size}"
        
        return True, ""
```

### 5.4 默认配置

```python
# 默认单设备
DEFAULT_SINGLE_DEVICE = "127.0.0.1#0"

# 默认多设备（本机 3 卡）
DEFAULT_MULTI_DEVICE = "127.0.0.1#0,127.0.0.1#1,127.0.0.1#2"
```

---

## 六、内存管理核心理念

### 6.1 核心原则

**程序初始化后，在固定内存地址上操作，避免各种内存复制（除非必要）。**

| 操作类型 | 推荐 | 说明 |
|---------|------|------|
| KV Cache 追加 | ✅ 原地切片写入 | `cache[:, :, start:end, :] = new_kv` |
| Cache 搬运 | ✅ 索引搬运 | `KVCache.copy(src_indices, dst_start)` |
| Cache 裁剪 | ✅ 只更新长度 | `truncate(length)` 不释放内存 |
| Prefix 复制 | ⚠️ 仅 Batching 初始化时 | 一次性复制，不重复 |
| 分布式传输 | ⚠️ 必要的复制 | 通信需要 |
| 每次 forward 后 clone | ❌ 禁止 | 严重浪费内存 |
| expand() + clone() | ❌ 禁止 | 使用 BIM 索引代替 |

### 6.2 KVCache 类设计

```python
class KVCache:
    """支持原地操作的 KV Cache"""
    
    def append(self, new_kv):
        """原地追加（无复制）"""
        self.data[:, :, self.current_length:end, :] = new_kv
        self.current_length = end
    
    def copy(self, src_indices, dst_start):
        """原地索引搬运（verify 后更新）"""
        self.data[:, :, dst_start:dst_start+len(src_indices), :] = \
            self.data[:, :, src_indices, :]
    
    def truncate(self, length):
        """裁剪长度（不释放内存）"""
        self.current_length = length
    
    def get_valid(self):
        """获取有效部分视图（无复制）"""
        return self.data[:, :, :self.current_length, :]
```

---

## 七、双模式推理设计

详见 [STATE_MANAGER.MD](STATE_MANAGER.MD) 文档。

### 7.1 模式概述

| 特性 | In-One-Sequence (BIM) 模式 | Batching 模式 |
|------|---------------------------|--------------|
| **输入形状** | `[1, total_len]` | `[num_branches, max_len]` |
| **KV Cache batch_size** | 1 | num_branches |
| **Cache 管理** | 共享 prefix，原地操作 | 复制 prefix N 份 |
| **Padding** | 无需 padding | 需要左填充对齐 |
| **Attention Mask** | BIM-based mask | 标准 causal + padding mask |
| **内存效率** | 高（无复制） | 较低（prefix 复制 + padding） |
| **实现复杂度** | 较高（BIM 管理） | 较低 |

### 7.2 BIM 模式 Draft Tree 关键处理

**这是 BIM 模式的核心难点：**

```
expand_root 后：5 个分支 × top_k=10 = 50 个 token
↓
这 50 个 token 构成一个 sequence 输入 Draft Model
↓
需要临时状态：draft_bim, draft_positions, draft_attention_mask
↓
grow_tree 的 cache 是临时的，结束后 truncate 回 expand_root 长度
```

### 7.3 Batching 模式对齐机制

**关键点在于对齐：**

1. **Verify 后对齐**：不同分支接收长度不同（如 1, 2, 3），需要左填充对齐
2. **Continuous Decoding**：新分支 prefill (200 tokens) + 老分支 decode (2 tokens)
   - 新分支需要复制 prefix cache
   - 老分支 cache 长度需要"概念上"对齐
   - position/mask 需要精确管理

### 7.4 模式切换

```python
# 使用 BIM 模式
python run_specsot.py --use_bim_mode True

# 使用 Batching 模式（默认）
python run_specsot.py --use_bim_mode False
```

---

## 八、inference_engine 与 generate 流程的关系

### 7.1 职责划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SpecSoTOrchestrator (generate/)                     │
│  负责：流程编排、停止条件、结果合并                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  generate_specsot():                                                        │
│    ├── _skeleton_prefill()  ──────────────────────────────────────┐        │
│    │     └── 调用 InferenceEngine.prefill_single()                │        │
│    ├── _skeleton_decode()  ───────────────────────────────────────┤        │
│    │     └── while 循环                                            │        │
│    │         └── 调用 InferenceEngine.decode_step_single()        │        │
│    ├── _skeleton_parse()     # 纯 Python 逻辑                      │        │
│    ├── _schedule_branches()  # 调用 scheduling/                    │        │
│    ├── _parallel_prefill()  ──────────────────────────────────────┤        │
│    │     └── 调用 InferenceEngine.prefill_parallel()              │        │
│    ├── _parallel_decode()  ───────────────────────────────────────┤        │
│    │     └── while 循环                                            │        │
│    │         └── 调用 InferenceEngine.decode_step_parallel()      ▼        │
│    └── _merge_results()      # 纯 Python 逻辑                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         InferenceEngine (core/)                             │
│  负责：底层推理操作、评估、状态更新                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  单序列操作 (Skeleton 阶段):                                                 │
│    ├── prefill_single()      # Base Model + Eagle Layer 初始化              │
│    ├── decode_step_single()  # Draft -> Verify -> Update                   │
│    │     ├── drafter.generate_draft_tree()                                  │
│    │     ├── verify_step_single()                                          │
│    │     │     └── eval_utils.evaluate_single()                            │
│    │     └── update_state_single()                                         │
│    └── verify_step_single() / update_state_single() (可独立调用)            │
│                                                                             │
│  并行操作 (Branch 阶段):                                                     │
│    ├── prefill_parallel()    # 并行 Prefill                                 │
│    ├── decode_step_parallel()# Draft -> Verify -> Update (多分支)          │
│    │     ├── drafter.generate_draft_tree()                                  │
│    │     ├── verify_step_parallel()                                        │
│    │     │     └── eval_utils.evaluate_parallel()                          │
│    │     └── update_state_parallel()                                       │
│    └── continuous_decode_step_parallel() (Continuous Batching 专用)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 调用关系示例

```python
# SpecSoTOrchestrator (高层)
class SpecSoTOrchestrator:
    def __init__(self, model, state_manager, inference_engine, ...):
        self.model = model
        self.state_manager = state_manager
        self.inference_engine = inference_engine
    
    def generate_specsot(self, task_prompt, ...):
        # Phase 1: Skeleton Prefill
        prefill_result = self._skeleton_prefill(task_prompt, ...)
        
        # Phase 2: Skeleton Decode (循环)
        for step in range(max_steps):
            if self._check_stop_conditions():
                break
            draft_tokens, ... = self.inference_engine.decode_step_single(...)
        
        # Phase 3-4: Parse & Schedule
        parse_result = self._skeleton_parse(...)
        schedule_result = self._schedule_branches(...)
        
        # Phase 5-6: Parallel Prefill & Decode
        prefill_result = self._parallel_prefill(...)
        for step in range(max_steps):
            if self._check_stop_conditions():
                break
            self.inference_engine.decode_step_parallel(...)
        
        # Phase 7: Merge
        return self._merge_results(...)


# InferenceEngine (底层)
class InferenceEngine:
    def __init__(self, base_model, eagle_layer, drafter, state_manager, ...):
        self.base_model = base_model
        self.eagle_layer = eagle_layer
        self.drafter = drafter
        self.state_manager = state_manager
    
    def decode_step_single(self, ...):
        # 1. Draft
        draft_tokens, ... = self.drafter.generate_draft_tree(...)
        
        # 2. Verify
        logits, hidden_states = self.verify_step_single(...)
        best_candidate, accept_length, sample_token = eval_utils.evaluate_single(...)
        
        # 3. Update
        self.update_state_single(...)
        
        return draft_tokens, ...
```

---

## 九、关键设计决策

### 8.1 Generate 层独立

- **决策**：将生成逻辑从 `SpecSoTModel` 抽取到 `SpecSoTOrchestrator`
- **原因**：`specsot_model.py` 文件过长（3000+ 行），需要拆分；生成流程与模型定义应解耦
- **实现**：`SpecSoTOrchestrator` 持有 Model、StateManager、InferenceEngine 的引用

### 8.2 推理引擎独立

- **决策**：创建 `InferenceEngine` 类封装 prefill/decode/verify/update 方法
- **原因**：这些是底层推理原语，应与生成流程分离
- **实现**：`InferenceEngine` 直接操作 Base Model 和 Eagle Layer

### 8.3 评估函数集中

- **决策**：将 `evaluate_single`, `evaluate_parallel`, `greedy_sampling`, `rejection_sampling` 移至 `core/eval_utils.py`
- **原因**：这些函数与推理引擎紧密相关，应该放在一起
- **使用**：`InferenceEngine` 调用这些评估函数

### 8.4 BIM 工具函数内置

- **决策**：将 `build_parallel_prefill_mask`, `build_continuous_decode_mask` 等 BIM 相关函数移入 `BranchStateManager`
- **原因**：这些函数只与 BIM 状态管理相关，不应放在通用 utils 中
- **实现**：作为 `BranchStateManager` 的方法或内部函数

### 8.5 GPUMemoryMonitor 移至 utils

- **决策**：将 `GPUMemoryMonitor` 从 `engine/worker.py` 移至 `utils/gpu_monitor.py`
- **原因**：这是通用工具，不应该绑定在 Worker 中

### 8.6 KV Cache 不封装

- **决策**：保持 `KVCache` 类和 `initialize_past_key_values` 函数，不再额外封装 Manager
- **原因**：当前实现已足够清晰，额外封装会增加复杂度
- **注意**：KV Cache 使用自定义类，不是普通 torch.Tensor，操作需使用 `.copy()`, `.cat()` 等方法

---

## 十、工作量评估

| 模块 | 任务 | 复杂度 | 优先级 |
|------|------|--------|--------|
| config/ | 设备配置模块 | 低 | P0 |
| generate/ | SpecSoTOrchestrator 抽取 | 高 | P0 |
| core/state_manager.py | BranchStateManager 完善 | 高 | P0 |
| core/inference_engine.py | 推理引擎抽取 | 高 | P0 |
| core/eval_utils.py | 评估函数迁移 | 低 | P1 |
| utils/gpu_monitor.py | GPUMemoryMonitor 迁移 | 低 | P2 |
| utils/tensor_utils.py | 张量工具迁移 | 低 | P2 |

---

## 十一、验证方法

### 功能验证

```bash
# 单机单卡
python run_specsot.py --distributed False --devices "127.0.0.1#0"

# 单机多卡
python run_specsot.py --distributed True --devices "127.0.0.1#0,127.0.0.1#1"

# BIM 模式
python run_specsot.py --use_bim_mode True

# Batching 模式
python run_specsot.py --use_bim_mode False
```

### 一致性验证

```bash
# 对比 Batching 和 BIM 模式输出
python test_mode_consistency.py
```
