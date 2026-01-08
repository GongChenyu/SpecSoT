# VSCode 调试指南

## 快速开始

### 方式1: 使用调试启动器（推荐）

1. 打开 `debug_launcher.py` 文件
2. 修改 `CONFIG` 字典中的参数（如果需要）
3. 按 `F5` 或点击"运行和调试"
4. 选择配置："调试分布式推理 (单机多卡)"

### 方式2: 直接运行调试启动器

```bash
cd Communication-Optimize
python debug_launcher.py
```

### 方式3: 调试单个Rank（高级）

如果你只想调试特定的rank（例如调试某个rank的逻辑），可以：

1. 在一个终端启动其他rank：
```bash
# 终端1 - Rank 1
CUDA_VISIBLE_DEVICES=1 python distributed_inference.py --rank 1 --world_size 3 ...

# 终端2 - Rank 2  
CUDA_VISIBLE_DEVICES=2 python distributed_inference.py --rank 2 --world_size 3 ...
```

2. 在VSCode中调试Rank 0：
   - 选择配置："调试单个Rank (Rank 0)"
   - 按 `F5` 开始调试

## 可用的调试配置

### 1. 调试分布式推理 (单机多卡)
- 启动完整的3个进程
- 使用Pairwise同步策略
- 适合调试整体流程

### 2. 调试单个Rank (Rank 0)
- 只启动Rank 0进程
- 需要手动启动其他Rank
- 适合深度调试特定rank的逻辑

### 3. 调试Pairwise同步
- 使用Pairwise同步策略
- 调试两两通信的cache同步

### 4. 调试Ring同步
- 使用Ring同步策略
- 调试环形通信的cache同步

## 调试技巧

### 1. 设置断点
- 在 `distributed_inference.py` 中的任何位置设置断点
- 在 `modeling_qwen3_kv_distributed.py` 中设置断点
- 在 `cache_sync_manager.py` 中设置断点

### 2. 查看变量
- 使用VSCode的"变量"面板查看所有局部变量
- 使用"监视"面板添加表达式监视
- 鼠标悬停在变量上查看值

### 3. 调试多进程
- 每个rank在单独的进程中运行
- 输出会显示在集成终端中
- 可以通过进程ID识别不同的rank

### 4. 修改参数
在 `debug_launcher.py` 中修改 `CONFIG` 字典：

```python
CONFIG = {
    'model_path': '...',
    'chunk_size': 128,        # 修改chunk大小
    'sync_strategy': 'ring',  # 切换同步策略
    'max_new_tokens': 50,     # 调试时用小值
    'prompt': '你的测试prompt',
}
```

### 5. 查看日志
如果使用shell脚本启动，日志在：
```bash
tail -f logs/rank0.log
tail -f logs/rank1.log
tail -f logs/rank2.log
```

## 常见问题

### Q: 如何只调试特定的函数？
A: 在该函数开始处设置断点，然后按F5启动调试。

### Q: 如何跳过某些代码？
A: 使用"Step Over" (F10) 而不是"Step Into" (F11)。

### Q: 进程挂起怎么办？
A: 按 Ctrl+C 终止，或者在VSCode中点击"停止"按钮。

### Q: 如何查看tensor的值？
A: 在调试控制台中输入：
```python
print(tensor.shape)
print(tensor)
```

### Q: 端口被占用怎么办？
A: 修改 `CONFIG` 中的 `master_port` 为其他值。

## 文件说明

- `debug_launcher.py` - Pairwise策略调试启动器
- `debug_launcher_ring.py` - Ring策略调试启动器
- `.vscode/launch.json` - VSCode调试配置
- `distributed_inference.py` - 主推理脚本
- `modeling_qwen3_kv_distributed.py` - 分布式模型实现
- `cache_sync_manager.py` - Cache同步管理器

## 性能调试

如果要调试性能问题：

1. 使用较小的模型或较短的prompt
2. 减小 `max_new_tokens` 的值
3. 添加时间测量：
```python
import time
start = time.time()
# 你的代码
print(f"耗时: {time.time() - start:.3f}s")
```

## 下一步

- 尝试修改同步策略看性能差异
- 调试cache同步的正确性
- 优化通信开销
