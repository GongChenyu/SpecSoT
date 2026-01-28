SpecSoT 下一代系统架构设计文档 (v2.0)
1. 核心设计理念
为了解决当前系统在“大规模 Draft Token”、“变长分支”和“分布式调度”上的痛点，新架构将遵循以下三个原则：

计算与调度解耦 (Decoupling)：Model 类只负责数学运算（Forward），不再负责循环控制。Engine 类负责全局调度。

显存页式管理 (Paged & Radix Attention)：引入逻辑块（Logical Block）与物理块（Physical Block）的映射，实现 Skeleton 前缀的零拷贝复用。

动态分布式编排 (Elastic Orchestration)：基于 Ray 或类似框架，实现 Worker 的动态分组。Skeleton 阶段组团计算（TP/SP），Parallel 阶段打散计算（Data Parallel）。

2. 推荐文件结构 (File Structure)
新的目录结构清晰地划分了控制平面（Engine/Scheduler）、数据平面（Memory/Cache）和计算平面（Workers/Models）。

Plaintext

SpecSoT-v2/
├── api/                        # [接口层] 对外暴露服务
│   ├── server.py               # RESTful API Server (FastAPI)
│   └── protocol.py             # 请求/响应数据结构定义
│
├── core/                       # [控制平面] 大脑，负责逻辑与调度
│   ├── __init__.py
│   ├── engine.py               # 【核心】SpecSoTEngine，整个系统的指挥官
│   ├── scheduler.py            # 【核心】调度器。负责 Continuous Batching，决定下一帧跑哪些 Token
│   └── policy.py               # 调度策略 (FCFS, Max-Throughput, SoT-Priority)
│
├── memory/                     # [数据平面] 显存管理 (参考 vLLM PagedAttention)
│   ├── __init__.py
│   ├── block_manager.py        # 逻辑块管理器。负责维护 BlockTable (逻辑 -> 物理映射)
│   ├── cache_engine.py         # 物理显存操作。负责 GPU 上的 malloc/free 和 Tensor 移动
│   └── radix_tree.py           # 前缀树管理。用于实现 SoT 的 Skeleton 前缀共享 (RadixAttention)
│
├── workers/                    # [计算平面] 分布式执行单元
│   ├── __init__.py
│   ├── worker.py               # Worker 主类。驻留在 GPU 上，接收 Engine 指令
│   ├── model_runner.py         # 模型执行器。负责准备 Input Tensors，调用 Model Forward
│   └── dist_coordinator.py     # 分布式协调器 (NCCL/Ray)。管理通信组 (Process Group)
│
├── models/                     # [模型定义] 纯粹的计算图定义 (无业务逻辑)
│   ├── __init__.py
│   ├── base_model.py           # 基础大模型 (Base Model) 包装器
│   ├── draft_model.py          # 投机模型 (Eagle/Draft) 包装器
│   └── layers/
│       ├── attention.py        # 支持 PagedAttention 的算子实现
│       ├── eagle_layers.py     # Eagle 专用层
│       └── sampler.py          # 采样算子
│
├── strategies/                 # [策略层] 针对不同任务的解码逻辑
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── speculative.py          # 标准投机解码逻辑
│   └── skeleton_of_thought.py  # SoT 逻辑：负责解析骨架、Fork 分支请求
│
└── utils/
    ├── tokenizer.py
    └── configs.py
3. 关键模块详解
3.1 core/scheduler.py (调度器)
这是系统的“心脏”。它不再是一个简单的 for 循环，而是一个基于状态机的调度器。

功能：在每个 Step，检查 Waiting 队列。

SoT 特性支持：

当 Skeleton 生成完成，Scheduler 会收到信号。

它将 Skeleton 的 KV Block 标记为“共享”。

它瞬间创建 N 个新的 SequenceGroup（对应 N 个分支），这些 Group 的 Block Table 初始指向共享的 Skeleton Block。

Continuous Batching：如果 Branch A 结束了，Branch B 还在跑，Scheduler 会立即从队列中抓取下一个请求填补空缺，保持 GPU 满载。

3.2 memory/radix_tree.py (Radix Attention)
解决“Skeleton 阶段 Pre-fill 并行”后如何复用的问题。

功能：维护一个前缀树，树的节点是 KV Cache 的物理块索引。

机制：Skeleton 生成的 KV Cache 被注册到树中。当生成 N 个分支时，系统查询这棵树，发现前缀匹配，直接返回物理块索引引用，无需显存拷贝。

3.3 workers/dist_coordinator.py (动态分布式)
解决“Skeleton 需要 TP/SP，Parallel 需要 DP”的冲突。

功能：基于 Ray 或 Torch Distributed 管理 GPU 进程。

动态分组：

Phase 1 (Skeleton): 将 GPU 0,1,2,3 组成一个 PlacementGroup，使用 Tensor Parallelism 加速长骨架生成。

Phase 2 (Parallel): 解散组。GPU 0 跑 Branch 1, GPU 1 跑 Branch 2...

或者：如果模型很大，保持 GPU 0+1 一组跑 Branch A，GPU 2+3 一组跑 Branch B。这由 Scheduler 动态下发指令控制。

4. 重构演进路线图 (Roadmap)
不要试图一次性重写所有代码。建议分三个阶段进行：

第一阶段：解耦循环 (Loop Decoupling)
目标：将 generate 死循环打破，变为 Step-by-Step 的执行。

创建 core/engine.py 和 workers/model_runner.py。

将 SpecSoTModel.generate 中的 while 循环提取到 Engine 中。

Model 只保留 forward 函数。

成果：代码逻辑清晰，可以手动控制单步执行。

第二阶段：显存池化 (Memory Pooling)
目标：消除分支生成时的 KV Cache 物理拷贝，实现零拷贝 Fork。

引入 BlockTable 概念（简单的 List[int] 即可）。

修改 Attention 层，使其支持从 Block Table 读取 KV（哪怕先用 Python 模拟 PagedAttention）。

实现 fork_request 逻辑，只复制 Block Table，不复制 Tensor。

成果：分支生成的启动延迟（Latency）几乎降为零，显存占用大幅降低。

第三阶段：全异步分布式 (Fully Async Distributed)
目标：实现异构流水线，Skeleton 和 Parallel 阶段动态调度。

引入 Ray 框架接管 Worker 启动。

实现 Scheduler 的“迭代级调度” (Iteration-level Scheduling)。

支持 Worker 的动态分组（Detach/Attach）。

成果：系统吞吐量（Throughput）最大化，能够处理极高并发。

5. 针对你当前项目的特别提示
不要过早优化 CUDA Kernel：在架构未定型前，先用 PyTorch 高级 API 实现 PagedAttention 的逻辑（通过 index_select 和 reshape）。逻辑通了再去写 Triton/CUDA Kernel。

保留 Eagle 的独立性：Eagle 本质上是一个特殊的 Draft Model。在 workers/model_runner.py 中，应该有一个专门的 run_speculative_step 方法，在这个方法里调用 Draft Model。

利用 Python 的灵活性：SoT 的 Prompt 解析（Parsing）逻辑（strategies/skeleton_of_thought.py）应该尽量独立，不要耦合在模型代码里，这样方便你换不同的 Prompt 模板。