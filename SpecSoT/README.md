# SpecSoT 模块结构说明

## 目录结构

```
SpecSoT/
├── __init__.py                    # 模块导出
├── specsot_model.py              # SpecSoT 主模型类
├── eagle_layer.py                # Eagle Layer (草稿模型)
├── logits_processor.py           # Logits 处理器
├── inference_utils.py            # 推理工具函数
├── prompts.py                    # 提示模板
├── configs.py                    # 配置类
├── kv_cache.py                   # KV Cache 实现
├── utils_c.py                    # C 工具函数
├── global_recorder.py            # 全局记录器
├── modeling_qwen3_kv.py          # Qwen3 + KV Cache
├── modeling_qwen2_kv.py          # Qwen2 + KV Cache
├── modeling_llama_kv.py          # LLaMA + KV Cache
└── modeling_mixtral_kv.py        # Mixtral + KV Cache
```

## 核心模块

### 1. specsot_model.py
主模型类 `SpecSoTModel`，包含：
- `generate()` - 主入口
- `_generate_standard()` - 标准 SD 模式
- `_generate_with_skeleton()` - 骨架并行模式
- `_decode_loop_single()` - 单序列解码循环
- `_decode_loop_parallel()` - 并行分支解码循环
- `_init_parallel_state()` - 初始化并行状态
- `_verify_step_parallel()` - 并行验证
- `_update_parallel_state()` - 更新并行状态

### 2. eagle_layer.py
轻量级草稿模型 `EagleLayer`，包含：
- `generate_draft_tree()` - Draft Tree 生成主入口
- `_expand_root()` - 根节点扩展
- `_grow_tree()` - 树生长
- `_post_process_tree()` - 后处理

### 3. logits_processor.py
语义约束处理器：
- `SemanticLogitsProcessor` - 骨架格式约束

### 4. inference_utils.py
推理工具函数：
- `prepare_logits_processor()` - 准备 logits 处理器
- `initialize_tree_single()` - 单序列树初始化
- `initialize_tree_parallel()` - 并行树初始化
- `tree_decoding_single()` - 单序列验证
- `evaluate_posterior()` - 后验评估
- `update_inference_inputs()` - 更新推理输入

### 5. prompts.py
提示模板：
- `base_prompt` - 基础系统提示
- `skeleton_trigger_zh` - 骨架生成触发提示
- `parallel_trigger_zh` - 并行生成触发提示

## 使用示例

```python
from SpecSoT import SpecSoTModel

# 加载模型
model = SpecSoTModel.from_pretrained(
    base_model_path="path/to/base_model",
    ea_model_path="path/to/eagle_model",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 标准模式生成
output_ids, stats = model.generate(
    task_prompt="你的问题",
    max_new_tokens=2048,
    temperature=0.0,
    enable_parallel=False,
)

# 骨架并行模式生成
output_ids, stats = model.generate(
    task_prompt="你的问题",
    max_new_tokens=3000,
    temperature=0.0,
    enable_parallel=True,
    para_token_ids={
        "para_begin_token_id": ...,
        "para_end_token_id": ...,
        ...
    },
)
```

## 运行脚本

使用 `run_specsot.py` 运行推理：

```bash
python run_specsot.py \
    --base_model_path /path/to/base_model \
    --eagle_model_path /path/to/eagle_model \
    --task planning \
    --enable_parallel True \
    --max_new_tokens 3000 \
    --num_samples 10
```

## 依赖关系

```
SpecSoTModel
├── EagleLayer
│   ├── configs.py (EConfig)
│   └── utils_c.py
├── modeling_*_kv.py (Base Models)
│   ├── kv_cache.py (KVCache)
│   └── global_recorder.py
├── inference_utils.py
├── logits_processor.py
└── prompts.py
```

## 导出接口

从 `SpecSoT` 模块可以导入：

```python
from SpecSoT import (
    # 主模型
    SpecSoTModel,
    
    # Eagle Layer
    EagleLayer,
    
    # Processors
    SemanticLogitsProcessor,
    
    # Utils
    prepare_logits_processor,
    initialize_tree_single,
    initialize_tree_parallel,
    tree_decoding_single,
    evaluate_posterior,
    update_inference_inputs,
    reset_tree_mode,
    stack_with_left_padding,
    initialize_past_key_values,
    
    # Config
    EConfig,
    
    # Prompts
    base_prompt,
    skeleton_trigger_zh,
    parallel_trigger_zh,
)
```
