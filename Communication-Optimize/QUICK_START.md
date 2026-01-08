# Quick Start Guide - Per-Layer Synchronization

## 最新改动 (解决死锁问题)

### 问题
程序hang在cache同步阶段，因为一次性同步所有12层的cache导致通信死锁。

### 解决方案
实现**逐层计算、逐层同步**机制：
- 每计算完一层，立即同步该层的cache
- 然后继续下一层
- 避免了批量同步的死锁问题

## 如何运行

### 方法1：使用VSCode Debug (推荐)

1. 打开VSCode
2. 按F5或点击"Run and Debug"
3. 选择 "Debug Distributed Inference (3 Ranks)"
4. 开始调试

### 方法2：使用debug_launcher.py

```bash
cd Communication-Optimize
python debug_launcher.py
```

配置在 `debug_launcher.py` 的 `CONFIG` 字典中：
```python
CONFIG = {
    'model_path': '/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B',
    'chunk_size': 128,
    'sync_strategy': 'pairwise',  # or 'ring'
    'device_mode': 'single_node',
    'world_size': 3,
    'gpu_ids': [5, 6, 7],  # 使用GPU 5, 6, 7
}
```

### 方法3：使用shell脚本

```bash
cd Communication-Optimize
bash launch_distributed.sh
```

## 关键参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `sync_strategy` | Cache同步策略 | `pairwise`, `ring` |
| `device_mode` | 设备模式 | `single_node` (单机多卡), `multi_node` (多机单卡) |
| `chunk_size` | SP chunk大小 | 128, 256等 |
| `gpu_ids` | 指定GPU ID | 例如 [5, 6, 7] |
| `world_size` | 进程数量 | 必须等于gpu_ids长度 |

## 架构说明

### Prefill阶段 (SP+PP)

```
Rank 0 (GPU 5): Layer 0-11
Rank 1 (GPU 6): Layer 12-23  
Rank 2 (GPU 7): Layer 24-35

每个chunk:
  Rank 0: embedding → Layer 0 → sync → Layer 1 → sync → ... → Layer 11 → sync → 发送到Rank 1
  Rank 1: 接收 → Layer 12 → sync → Layer 13 → sync → ... → Layer 23 → sync → 发送到Rank 2
  Rank 2: 接收 → Layer 24 → sync → Layer 25 → sync → ... → Layer 35 → sync → norm
```

### Decode阶段 (全量冗余计算)

所有3个rank独立计算所有36层，使用prefill阶段同步好的cache。

## 文件结构

```
Communication-Optimize/
├── distributed_inference.py          # 主推理引擎
├── modeling_qwen3_kv_distributed.py  # 分布式模型包装
├── cache_sync_manager.py             # Cache同步管理器
├── debug_launcher.py                 # 调试启动器
├── launch_distributed.sh             # Shell启动脚本
├── test_per_layer_sync.py           # 测试脚本
└── MODIFICATIONS.md                  # 详细修改说明
```

## 新增的关键方法

### modeling_qwen3_kv_distributed.py

```python
# 单层forward
forward_single_layer(layer_idx, hidden_states, ...) -> dict

# 单层cache同步  
sync_single_layer_cache(layer_idx, kv_cache) -> Tuple
```

### distributed_inference.py

```python
# 重构后的prefill_phase
# 现在逐层计算和同步，而不是批量操作
prefill_phase(prompt) -> (hidden_states, kv_caches)
```

## 监控和调试

### 日志级别
修改 `debug_launcher.py` 中的:
```python
# 详细调试信息
'prompt': '请详细介绍一下人工智能的发展历史。',
'max_new_tokens': 50,  # 调试时用较小的值
```

在distributed_inference.py中设置日志级别：
```python
logging.basicConfig(level=logging.DEBUG)  # 更详细的输出
```

### 查看日志
每个rank的日志在终端独立输出，可以看到：
- 层级计算进度
- Cache同步状态
- 通信时间统计

## 故障排查

### 端口被占用
```bash
# 手动清理
lsof -ti:29500 | xargs kill -9

# 或修改端口
CONFIG['master_port'] = '29501'
```

### GPU显存不足
减小batch size或chunk size：
```python
CONFIG['chunk_size'] = 64  # 降低到64
```

### 通信超时
检查网络连接（多机模式）或增加超时时间。

## 性能优化建议

1. **单机多卡**: 使用NCCL后端（已自动选择）
2. **多机环境**: 确保节点间网络质量良好
3. **Chunk大小**: 根据序列长度和显存调整
4. **同步策略**: 
   - `pairwise`: 适合小规模（2-4设备）
   - `ring`: 适合大规模（4+设备）

## 下一步

测试通过后，可以：
1. 调整prompt和max_new_tokens测试不同场景
2. 比较pairwise vs ring性能
3. 在多机环境测试（Jetson Orin节点）
4. 集成到实际应用
