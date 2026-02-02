# SpecSoT v2

**Speculative Decoding + Skeleton-of-Thought 推理系统**

SpecSoT 是一个高效的 LLM 推理系统，结合投机解码 (Speculative Decoding) 和骨架思维 (Skeleton-of-Thought) 技术，实现并行分支解码加速。

## 系统概述

### 核心特性

- **投机解码 (EAGLE)**: 使用轻量级 Draft Model 预测多个候选 token，Base Model 并行验证
- **骨架并行 (SpecSoT)**: 先生成响应骨架，再并行解码各分支内容
- **双模式推理**: 支持 BIM (In-One-Sequence) 和 Batching 两种并行模式
- **分布式支持**: 支持多 GPU 分布式推理
- **语义约束**: FSM 状态机确保骨架格式正确

### 执行模式

| 模式 | enable_parallel | distributed | 说明 |
|------|-----------------|-------------|------|
| 纯 EAGLE | False | False | 标准投机解码 |
| SpecSoT 单机 | True | False | 骨架并行 + 单 GPU |
| SpecSoT 分布式 | True | True | 骨架并行 + 多 GPU |

---

## 架构介绍

### 10 层架构

```
┌─────────────────────────────────────────────────────────┐
│                      API 层                              │
│              (server.py, protocol.py)                   │
├─────────────────────────────────────────────────────────┤
│                    Engine 层                             │
│              (MasterEngine, WorkerEngine)               │
├─────────────────────────────────────────────────────────┤
│                   Generate 层                            │
│              (SpecSoTOrchestrator)                      │
├─────────────────────────────────────────────────────────┤
│                    Core 层                               │
│     (Drafter, StateManager, InferenceEngine, KVCache)   │
├─────────────────────────────────────────────────────────┤
│                   Models 层                              │
│        (Base Models: LLaMA/Qwen, Draft: Eagle)          │
├─────────────────────────────────────────────────────────┤
│                 Scheduling 层                            │
│           (BranchScheduler, BranchManager)              │
├─────────────────────────────────────────────────────────┤
│                 Processing 层                            │
│         (SemanticLogitsProcessor, Prompts)              │
├─────────────────────────────────────────────────────────┤
│                Distributed 层                            │
│           (CommManager, DistributedPrefill)             │
├─────────────────────────────────────────────────────────┤
│                   Config 层                              │
│     (SystemConfig, DistributedConfig, DeviceConfig)     │
├─────────────────────────────────────────────────────────┤
│                   Utils 层                               │
│          (GPUMonitor, Logging, TensorUtils)             │
└─────────────────────────────────────────────────────────┘
```

### 7 阶段流水线 (SpecSoT 模式)

```
1. Skeleton Prefill  → 骨架 Prefill
2. Skeleton Decode   → 骨架解码 (投机解码)
3. Skeleton Parse    → 骨架解析 (FSM)
4. Schedule Branches → 分支调度
5. Parallel Prefill  → 并行 Prefill
6. Parallel Decode   → 并行解码 (投机解码)
7. Merge Results     → 结果合并
```

---

## 目录结构

```
SpecSoT_v2/
├── api/                    # API 层
│   ├── protocol.py         # 请求/响应协议
│   └── server.py           # FastAPI HTTP 服务
├── config/                 # 配置层
│   ├── device_config.py    # 设备配置
│   ├── distributed_config.py # 分布式配置
│   └── system_config.py    # 系统配置
├── core/                   # 核心组件层
│   ├── drafter.py          # Draft Tree 生成器
│   ├── eval_utils.py       # 评估/采样工具
│   ├── inference_engine.py # 推理引擎
│   ├── kv_cache.py         # KV Cache 管理
│   └── state_manager.py    # 分支状态管理
├── distributed/            # 分布式层
│   ├── communication/      # 通信模块
│   └── distributed_prefill.py
├── engine/                 # 引擎层
│   ├── master.py           # Master 主控
│   ├── worker.py           # Worker 执行器
│   └── utils.py            # 引擎工具
├── generate/               # 生成层
│   ├── datatypes.py        # 数据类型定义
│   └── orchestrator.py     # 生成编排器
├── models/                 # 模型层
│   ├── modeling/           # Base Model 实现
│   └── modeling_draft/     # Draft Model (Eagle)
├── processing/             # 处理层
│   ├── logits_processor.py # Logits 处理器
│   └── prompts.py          # 提示模板
├── scheduling/             # 调度层
│   ├── branch_manager.py   # 分支管理器
│   └── branch_scheduler.py # 分支调度器
├── utils/                  # 工具层
│   ├── gpu_monitor.py      # GPU 监控
│   ├── logging_utils.py    # 日志工具
│   └── tensor_utils.py     # 张量工具
├── main.py                 # 统一命令行入口
├── specsot_model.py        # 主模型类
└── README.md               # 本文档
```

---

## 快速开始

### 安装依赖

```bash
pip install torch transformers accelerate pynvml
pip install fastapi uvicorn  # API 服务 (可选)
```

### 启动方式

#### 1. 单次推理

```bash
python -m SpecSoT_v2 infer \
    --prompt "请解释量子计算的基本原理" \
    --base_model_path /path/to/Llama-3.1-8B-Instruct \
    --eagle_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --max_new_tokens 512
```

#### 2. 启动 API 服务

```bash
python -m SpecSoT_v2 serve \
    --base_model_path /path/to/Llama-3.1-8B-Instruct \
    --eagle_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --host 0.0.0.0 \
    --port 8000
```

#### 3. 批量评估

```bash
python run_specsot.py \
    --distributed False \
    --enable_parallel True \
    --task planning \
    --num_samples 10
```

#### 4. 分布式推理

```bash
python run_specsot.py \
    --distributed True \
    --devices "127.0.0.1#0,127.0.0.1#1,127.0.0.1#2" \
    --layer_splits "14,28"
```

---

## 参数说明

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--base_model_path` | str | - | Base Model 路径 |
| `--eagle_model_path` | str | - | Eagle Model 路径 |
| `--use_eagle3` | bool | True | 是否使用 Eagle3 |

### 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_parallel` | bool | True | 启用骨架并行 |
| `--use_semantic_constraint` | bool | True | 使用 FSM 语义约束 |
| `--max_new_tokens` | int | 3000 | 最大生成 token 数 |
| `--temperature` | float | 0.0 | 采样温度 (0=greedy) |

### 分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--distributed` | bool | False | 启用分布式模式 |
| `--devices` | str | "127.0.0.1#0" | 设备列表 (ip#gpu_id) |
| `--layer_splits` | str | "14,28" | 层分割点 |
| `--comm_mode` | str | "p2p" | 通信模式 (p2p/ring) |

### 调度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_scheduling` | bool | True | 启用分支调度 |
| `--max_parallel` | int | 2 | 最大并行分支数 |

---

## API 接口

### POST /generate

生成文本。

**请求体:**
```json
{
    "prompt": "你的问题",
    "max_new_tokens": 512,
    "temperature": 0.0,
    "enable_parallel": true,
    "use_semantic_constraint": true
}
```

**响应:**
```json
{
    "text": "生成的文本",
    "tokens": 256,
    "time_ms": 1234.5,
    "mode": "specsot",
    "stats": {...}
}
```

### GET /health

健康检查。

### GET /info

模型信息。

---

## 注意事项

### 显存要求

| 模型 | 最小显存 | 推荐显存 |
|------|----------|----------|
| LLaMA-3.1-8B | 16GB | 24GB |
| Qwen3-4B | 8GB | 16GB |
| Vicuna-7B | 14GB | 24GB |

### 模型兼容性

| Base Model | Eagle Model | use_eagle3 |
|------------|-------------|------------|
| LLaMA-3.1-8B-Instruct | EAGLE3-LLaMA3.1-Instruct-8B | True |
| Qwen3-4B | Qwen3-4B_eagle3 | True |
| Vicuna-7B-v1.3 | EAGLE-Vicuna-7B-v1.3 | False |

### 性能优化建议

1. **显存不足**: 减小 `max_new_tokens` 或使用分布式模式
2. **速度优先**: 设置 `enable_parallel=True` 启用骨架并行
3. **质量优先**: 设置 `use_semantic_constraint=True` 确保格式正确

---

## 代码示例

### Python API

```python
from SpecSoT_v2 import SpecSoTModel

# 加载模型
model = SpecSoTModel.from_pretrained(
    base_model_path="/path/to/base_model",
    ea_model_path="/path/to/eagle_model",
    use_eagle3=True,
    dtype=torch.float16,
    device_map="cuda:0",
)

# 生成
output_ids, stats = model.generate(
    task_prompt="请解释量子计算的基本原理",
    max_new_tokens=512,
    temperature=0.0,
    enable_parallel=True,
)

# 解码
text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
```

### HTTP API

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_new_tokens": 100}'
```

---

## 许可证

MIT License
