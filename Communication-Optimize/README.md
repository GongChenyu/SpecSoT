# SP+PP分布式推理项目

这个项目实现了基于Sequence Parallel (SP) + Pipeline Parallel (PP) 的三设备分布式大模型推理系统。

## 项目结构

```
Communication-Optimize/
├── modeling_qwen3_kv.py              # 原始Qwen3模型
├── modeling_qwen3_kv_distributed.py  # 分布式版本的Qwen3模型
├── cache_sync_manager.py             # KV Cache同步管理器
├── distributed_inference.py          # 分布式推理主脚本
├── launch_distributed.sh             # 启动脚本
├── test_both_strategies.sh           # 测试两种同步策略
└── README.md                         # 本文件
```

## 系统设计

### Prefill阶段：SP + PP

- **Sequence Parallel (SP)**：将长prompt切分为多个chunk（默认128 tokens）
- **Pipeline Parallel (PP)**：模型的transformer层均分到3台设备
  - Device 0: Layer 0 ~ N/3
  - Device 1: Layer N/3 ~ 2N/3
  - Device 2: Layer 2N/3 ~ N

### Decode阶段：全量冗余计算

所有设备使用完整的KV cache进行相同的全量推理，无需通信。

### KV Cache同步策略

支持两种同步策略：

#### 1. Pairwise（两两通信）
- 使用`all_gather`操作
- 所有设备同时获得完整的cache
- 通信量：O(N²)，但并行度高

#### 2. Ring（环形通信）
- 设备按环形拓扑传递cache
- 通信量：O(N)，但需要多轮
- 适合带宽受限的场景

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers accelerate

# 创建日志目录
mkdir -p logs
```

### 2. 单机多卡模式（推荐用于测试）

```bash
# 赋予脚本执行权限
chmod +x launch_distributed.sh
chmod +x test_both_strategies.sh

# 使用pairwise策略
./launch_distributed.sh \
    /path/to/Qwen3-4B \
    localhost \
    29500 \
    128 \
    pairwise

# 使用ring策略
./launch_distributed.sh \
    /path/to/Qwen3-4B \
    localhost \
    29500 \
    128 \
    ring
```

### 3. 多机模式

在每台机器上分别运行：

```bash
# 机器1 (Rank 0)
python distributed_inference.py \
    --model_path /path/to/Qwen3-4B \
    --rank 0 \
    --world_size 3 \
    --master_addr <MASTER_IP> \
    --master_port 29500 \
    --chunk_size 128 \
    --sync_strategy pairwise \
    --prompt "你的prompt" \
    --max_new_tokens 100

# 机器2 (Rank 1) - 修改--rank 1
# 机器3 (Rank 2) - 修改--rank 2
```

### 4. 测试两种策略

```bash
./test_both_strategies.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | Qwen3模型路径 | 必需 |
| `--rank` | 当前设备rank (0, 1, 2) | 必需 |
| `--world_size` | 总设备数 | 3 |
| `--master_addr` | 主节点地址 | localhost |
| `--master_port` | 主节点端口 | 29500 |
| `--chunk_size` | SP的chunk大小 | 128 |
| `--sync_strategy` | cache同步策略 (pairwise/ring) | pairwise |
| `--prompt` | 输入prompt | 默认prompt |
| `--max_new_tokens` | 最大生成token数 | 100 |

## 性能测量

系统会自动记录以下时间点：

1. **Prefill开始/结束时间**：SP+PP计算时间
2. **Cache同步完成时间**：KV cache同步耗时
3. **Decode开始/结束时间**：全量推理时间
4. **总时间**：端到端延迟

查看日志：

```bash
# 查看Rank 0的日志
tail -f logs/rank0.log

# 或者实时查看所有设备
tail -f logs/rank*.log
```

## 架构特点

### 计算与通信重叠
- Cache同步在独立线程中进行
- 避免阻塞主推理流程

### 灵活的同步策略
- 支持两种同步策略，可根据网络拓扑选择
- 易于扩展新的同步策略

### 两阶段设计
- Prefill：SP+PP，最大化并行度
- Decode：全量计算，避免通信开销

## 注意事项

1. **设备数量**：当前实现固定为3台设备，可修改`world_size`参数扩展
2. **模型加载**：每台设备都加载完整模型，确保有足够的显存
3. **网络配置**：多机模式需要确保网络互通，防火墙开放相应端口
4. **NCCL后端**：需要支持NCCL的CUDA环境

## 故障排查

### 问题1：进程卡住不动
- 检查所有设备是否都已启动
- 检查网络连接和防火墙设置
- 查看日志中的错误信息

### 问题2：CUDA out of memory
- 减小`chunk_size`
- 减小`max_new_tokens`
- 使用更大显存的GPU

### 问题3：通信超时
- 增大超时时间（在代码中修改）
- 检查网络带宽
- 尝试使用ring策略减少通信量

## 扩展性

### 添加新的同步策略

在`cache_sync_manager.py`中添加新方法：

```python
def _my_custom_sync(self, layer_idx, kv_cache):
    # 实现自定义同步逻辑
    pass
```

### 支持更多设备

修改`world_size`参数和相应的layer分配逻辑。

## 性能优化建议

1. **使用高速网络**：InfiniBand > 10GbE > 1GbE
2. **优化chunk大小**：根据模型和硬件调整
3. **使用混合精度**：FP16可减少通信量
4. **启用NCCL优化**：设置环境变量如`NCCL_ALGO`

## License

基于原始Qwen3模型的Apache 2.0许可证。
