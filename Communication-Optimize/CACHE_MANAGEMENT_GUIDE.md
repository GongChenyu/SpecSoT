# KV Cache管理系统说明

## 两种Cache操作模式

### 1. Sequence维度追加 (Sequence Level)
**场景**: 同一个rank处理不同chunk时
- **含义**: 同一层的不同tokens的cache拼接
- **示例**: Rank 0处理chunk 0的layer 0 → 处理chunk 1的layer 0
- **操作**: 在sequence维度（dim=2）追加
- **API**: `KVCache.append_sequence(tensor)`

```python
# 使用示例
for chunk in chunks:
    # 计算新的cache
    new_cache = model.forward_single_layer(...)
    
    # Sequence维度追加（同一层，不同tokens）
    layer_cache.append_sequence(new_cache)
```

### 2. Layer维度合并 (Layer Level)
**场景**: 不同rank处理同一chunk时
- **含义**: 同一tokens的不同layers的cache合并
- **示例**: Rank 0, 1, 2 分别处理chunk 0的layer 0-11, 12-23, 24-35
- **操作**: 将不同PP stage的cache片段在sequence维度拼接
- **API**: `KVCache.merge_from_ranks(tensors)`

```python
# 使用示例
# 通过P2P通信收集所有rank的cache
all_rank_caches = [rank0_cache, rank1_cache, rank2_cache]

# Layer维度合并（不同rank的sequence片段拼接）
merged_cache = layer_cache.merge_from_ranks(all_rank_caches)
```

## Cache接收状态追踪矩阵

### 矩阵结构
- **维度**: `[num_chunks, num_pp_stages]`
- **数值**: 0 = 未接收, 1 = 已接收
- **位置**: `distributed_inference.cache_received_indicator`

### 示例输出
```
============================================================
Cache接收状态矩阵 (Chunk × PP Stage):
Chunk    |  PP0  PP1  PP2
---------------------------
Chunk0   |   1    1    1
Chunk1   |   1    1    1
接收完成度: 6/6 (100.0%)
============================================================
```

### 查看方法
```python
# 在distributed_inference.py中自动调用
engine._log_cache_received_status()
```

## 完整流程示例

### Prefill阶段的Cache管理

```
输入序列: 21 tokens (7 tokens × 3 chunks)
模型: 36 layers (12 layers × 3 PP stages)

Timeline:
---------
Step 1: Rank 0 处理 Chunk 0, Layers 0-11
  → 生成 7 tokens 的 cache (sequence维度: 7)
  → 同步到所有rank (layer level合并: 7 → 21)
  → 状态矩阵[0, 0-2] = 1

Step 2: Rank 0 处理 Chunk 1, Layers 0-11
  → 生成新的 7 tokens 的 cache
  → Sequence维度追加: 7 → 14 (本地cache)
  → 同步到所有rank (layer level合并: 14 → 42)
  → 状态矩阵[1, 0-2] = 1

Step 3: Rank 0 处理 Chunk 2, Layers 0-11
  → 生成新的 7 tokens 的 cache
  → Sequence维度追加: 14 → 21 (本地cache)
  → 同步到所有rank (layer level合并: 21 → 63)
  → 状态矩阵[2, 0-2] = 1

最终结果:
- 每个rank拥有所有36层的完整cache
- 每层cache包含完整的21个tokens
- 状态矩阵全部为1 (100%接收完成)
```

## API参考

### KVCache类

#### `append_sequence(tensor: torch.Tensor) -> torch.Tensor`
在sequence维度追加新数据
- **参数**: `tensor` - 要追加的cache，shape: [B, H, L, D]
- **返回**: 追加后的完整cache
- **用途**: 同一层处理不同chunk时使用

#### `merge_from_ranks(tensors: list, dim: int = 2) -> torch.Tensor`
合并来自不同rank的cache
- **参数**: 
  - `tensors` - cache列表（来自不同rank）
  - `dim` - 拼接维度（默认2=sequence）
- **返回**: 合并后的cache
- **用途**: 不同PP stage的cache同步时使用

#### `get_data() -> torch.Tensor`
获取当前有效的cache数据
- **返回**: 连续内存的cache（up to current_length）

#### `reset()`
重置cache到空状态
- **用途**: 新推理开始前调用

### DistributedInference类

#### `cache_received_indicator: torch.Tensor`
Cache接收状态矩阵
- **形状**: `[num_chunks, num_pp_stages]`
- **类型**: `torch.int8` (CPU)

#### `_log_cache_received_status()`
打印cache接收状态（调试用）
- **输出**: 格式化的状态矩阵表格

## 调试技巧

### 1. 检查cache连续性
```python
cache = kv_cache.get_data()
assert cache.is_contiguous(), "Cache must be contiguous!"
```

### 2. 验证cache形状
```python
# Sequence维度追加后
expected_len = chunk0_len + chunk1_len
assert cache.shape[2] == expected_len

# Layer维度合并后
expected_len = rank0_len + rank1_len + rank2_len
assert cache.shape[2] == expected_len
```

### 3. 监控接收状态
```python
# 检查是否所有cache都已接收
all_received = (engine.cache_received_indicator == 1).all()
print(f"All cache received: {all_received}")
```

## 注意事项

1. **内存连续性**: 所有cache操作都保证返回连续内存
2. **预分配缓冲区**: KVCache使用固定大小的预分配内存，避免动态分配
3. **线程安全**: 当前实现不支持并发修改同一个KVCache
4. **容量检查**: merge_from_ranks会检查是否超出预分配容量

## 性能优化建议

1. **批量操作**: 尽量批量追加而不是逐个token追加
2. **避免拷贝**: 使用`get_data()`获取数据，避免不必要的`.clone()`
3. **状态复用**: Decode阶段直接使用prefill的cache，无需重新分配
