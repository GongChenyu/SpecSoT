# modeling_qwen3_kv.py 修改说明

## 重要修改 (2024)

### Per-Layer Cache Synchronization 实现

为了解决batch同步导致的死锁问题，我们实现了**逐层计算、逐层同步**的机制。

#### 问题背景
原始实现在prefill阶段会先计算当前rank的所有12层，然后一次性同步所有层的cache。这导致：
- 所有rank同时等待all_gather或ring通信完成
- 多层cache同时同步造成通信死锁
- 程序hang在"开始 Cache 同步"处

#### 解决方案
按照需求"每进行一层的计算就需要进行一次cache同步"，改为：
1. 逐层forward计算
2. 每计算完一层立即同步该层cache
3. 继续下一层

#### 修改文件

**1. modeling_qwen3_kv_distributed.py**

新增两个方法：

```python
def forward_single_layer(
    self,
    layer_idx: int,
    hidden_states: torch.FloatTensor,
    ...
) -> dict:
    """单层forward计算，返回该层的hidden_states和past_key_value"""
    
def sync_single_layer_cache(
    self,
    layer_idx: int,
    kv_cache: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """同步单层的KV cache"""
```

**2. distributed_inference.py**

重构 `prefill_phase()` 方法：

原逻辑：
```python
for chunk in chunks:
    outputs = forward_pipeline_stage(...)  # 计算所有12层
all_kv_caches = sync_kv_caches(outputs['past_key_values'])  # 一次性同步
```

新逻辑：
```python
for chunk in chunks:
    for layer_idx in range(start_layer, end_layer):  # 逐层
        layer_output = forward_single_layer(layer_idx, ...)
        synced_cache = sync_single_layer_cache(layer_idx, layer_output['past_key_value'])
        # 立即更新到all_kv_caches
```

**3. debug_launcher.py**

增强功能：
- `cleanup_port()`: 自动清理占用的端口
- `gpu_ids`: 指定使用的GPU设备 (例如 [5, 6, 7])
- 自动设置 `CUDA_VISIBLE_DEVICES` 环境变量

#### 测试

使用 `test_per_layer_sync.py` 或直接运行：

```bash
cd Communication-Optimize
python debug_launcher.py
```

## 原始说明

原始的 `modeling_qwen3_kv.py` 文件已经可以直接使用，**不需要修改**。

分布式功能已经在以下新文件中实现：

## 新增文件

1. **modeling_qwen3_kv_distributed.py**
   - 继承自 `modeling_qwen3_kv.py` 中的类
   - 添加了 Pipeline Parallel 支持
   - 添加了 KV Cache 同步功能
   - 提供了 `forward_pipeline_stage()` 方法用于pipeline推理

2. **cache_sync_manager.py**
   - 独立的 KV Cache 同步管理器
   - 支持两种同步策略（pairwise 和 ring）
   - 使用独立线程避免阻塞推理
   
3. **distributed_inference.py**
   - 主推理脚本
   - 实现了 SP+PP 的完整流程
   - 包含 prefill 和 decode 两个阶段

## 使用说明

直接使用新文件即可，无需修改原始的 `modeling_qwen3_kv.py`。

## 架构说明

```
modeling_qwen3_kv.py (原始文件，不修改)
    ↓ 继承
modeling_qwen3_kv_distributed.py (分布式版本)
    ↓ 使用
distributed_inference.py (主推理脚本)
    ↓ 依赖
cache_sync_manager.py (Cache同步)
```

这种设计的好处：
- 保持原始代码不变，便于维护
- 分布式功能作为扩展，模块化设计
- 易于切换和测试不同版本

## 关键类和方法

### Qwen3ModelDistributed (继承 Qwen3Model)

新增方法：
- `set_pipeline_range(start_layer, end_layer)`: 设置PP范围
- `set_full_model_mode()`: 切换到全量模型模式
- `forward_pipeline_stage(...)`: Pipeline stage的forward
- `sync_kv_caches(kv_caches)`: 同步KV cache

### Qwen3ForCausalLMDistributed (继承 Qwen3ForCausalLM)

新增方法：
- `set_pipeline_range(...)`: 设置PP范围
- `set_full_model_mode()`: 切换模式
- `forward_pipeline_stage(...)`: Pipeline forward
- `sync_kv_caches(...)`: 同步cache

## 实现细节

### Prefill阶段 (SP+PP)

1. 每个设备只加载部分层（PP）
2. Prompt被切分为chunks（SP）
3. 逐chunk在pipeline上流动
4. 每层计算完成后同步cache

### Decode阶段 (全量冗余)

1. 所有设备使用完整的KV cache
2. 每个设备独立计算（无通信）
3. 所有设备生成相同的token

### Cache同步

使用 `CacheSyncManager` 在独立线程中同步：
- Pairwise: 使用 `all_gather` 快速同步
- Ring: 环形传递，减少通信量

## 注意事项

1. 原始文件 `modeling_qwen3_kv.py` 保持不变
2. 所有分布式功能在新文件中实现
3. 通过继承和组合实现功能扩展
4. 便于后续维护和升级
