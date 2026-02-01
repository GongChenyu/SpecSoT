# 统一 BIM 状态管理与双模式推理系统设计文档

## 一、背景与目标

### 当前问题
- **Base Model** 使用 BIM (Branch Index Map) 实现单序列多分支推理
- **Eagle Layer** 使用 Batching 方式，通过 padding 对齐不同长度的分支
- 两种方式不统一，状态变量分散在多个模块中，增加了系统复杂度
- KV Cache 使用自定义 `KVCache` 类（非 torch.Tensor），需要特殊处理

### 目标
1. **创建统一的 BIM 状态管理类** - 集中管理 BIM、Position IDs、分支状态
2. **Base Model 和 Eagle Layer 共享状态** - 一套变量维护整个系统
3. **支持双模式切换** - Batching 模式和 In-One-Sequence (BIM) 模式
4. **整个 Draft 流程统一范式** - expand_root 和 grow_tree 都支持 BIM 模式
5. **兼容自定义 KV Cache** - 使用 `.copy()`, `.cat()` 等方法操作

---

## 二、内存管理核心理念

### 2.1 核心原则

**程序初始化后，在固定内存地址上操作，避免各种内存复制（除非必要）。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         内存管理设计原则                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 预分配固定大小的 KV Cache 缓冲区                                         │
│     - 初始化时分配 [batch_size, num_heads, max_seq_len, head_dim]           │
│     - 整个推理过程在这块内存上原地操作                                       │
│                                                                             │
│  2. 避免内存复制的场景                                                       │
│     ❌ 每次 forward 后 clone() KV Cache                                      │
│     ❌ 分支扩展时 expand() + clone() prefix cache                            │
│     ❌ verify 后重新分配 cache 空间                                          │
│                                                                             │
│  3. 允许内存复制的场景（必要情况）                                            │
│     ✅ Batching 模式下复制 prefix cache 到各分支（初始化时一次性）             │
│     ✅ 分布式通信时的 cache 传输                                             │
│                                                                             │
│  4. 原地操作方式                                                             │
│     - 使用索引切片写入: cache[:, :, start:end, :] = new_data                │
│     - 使用 KVCache.copy(indices, prev_length) 进行索引搬运                   │
│     - 使用 current_length 追踪有效长度，而非重新分配                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 KVCache 类设计目标

```python
class KVCache:
    """
    支持原地操作的 KV Cache 封装类
    
    设计目标：
    1. 预分配固定内存，避免动态扩展
    2. 提供原地操作接口，避免复制
    3. 追踪有效长度，支持裁剪和扩展
    """
    
    def __init__(self, shape, device, dtype):
        # 一次性分配最大容量
        self.data = torch.zeros(shape, device=device, dtype=dtype)
        self.current_length = 0
        self.capacity = shape[2]  # max_seq_len
    
    def append(self, new_kv: torch.Tensor):
        """原地追加新的 KV（无复制）"""
        new_len = new_kv.shape[2]
        end = self.current_length + new_len
        self.data[:, :, self.current_length:end, :] = new_kv
        self.current_length = end
    
    def copy(self, src_indices: torch.Tensor, dst_start: int):
        """原地索引搬运（用于 verify 后的 cache 更新）"""
        # 从 src_indices 指定的位置搬运到 dst_start 开始的连续位置
        self.data[:, :, dst_start:dst_start+len(src_indices), :] = \
            self.data[:, :, src_indices, :]
    
    def truncate(self, length: int):
        """裁剪到指定长度（只更新 current_length，不释放内存）"""
        self.current_length = length
    
    def get_valid(self) -> torch.Tensor:
        """获取有效部分的视图（无复制）"""
        return self.data[:, :, :self.current_length, :]
```

---

## 三、双模式对比

### 3.1 模式特性对比

| 特性 | In-One-Sequence (BIM) 模式 | Batching 模式 |
|------|---------------------------|--------------|
| **输入形状** | `[1, total_len]` | `[num_branches, max_len]` |
| **KV Cache batch_size** | 1 | num_branches |
| **Cache 管理** | 共享 prefix cache，原地操作 | 复制 prefix cache N 份 |
| **Padding** | 无需 padding | 需要左填充对齐 |
| **Attention Mask** | BIM-based mask | 标准 causal + padding mask |
| **内存效率** | 高（无复制） | 较低（prefix 复制 + padding） |
| **实现复杂度** | 较高（BIM 管理） | 较低 |
| **分支独立性** | 通过 BIM 索引隔离 | 完全独立 |

### 3.2 各阶段对双模式的影响

| 阶段 | BIM 模式 | Batching 模式 | 影响程度 |
|------|----------|--------------|----------|
| **Skeleton 生成** | 单序列，无影响 | 单序列，无影响 | 无 |
| **Skeleton 解析** | 无影响 | 无影响 | 无 |
| **Parallel Prefill** | 拉平序列，BIM mask | 复制 cache，padding | **高** |
| **Parallel Decode** | 单序列计算，BIM 索引 | 批量验证 | **高** |
| **Draft Tree 生成** | **关键**：合并 token 计算 | 批量 expand_root | **高** |
| **状态更新** | 原地更新，BIM 索引 | 各分支独立，对齐 | **高** |
| **Continuous Decode** | 新旧分支拉平 | prefix 复制 + 长度对齐 | **高** |

---

## 四、In-One-Sequence (BIM) 模式详解

### 4.1 核心概念

**整个过程 batch_size=1，所有任务/分支通过 BIM 和索引管理。Base Model 和 Draft Model 使用相同的管理方式。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIM 模式内存布局                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KV Cache: [1, num_heads, capacity, head_dim]                               │
│                                                                             │
│  序列布局:                                                                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┬───────┐         │
│  │   Prefix    │  Branch 0   │  Branch 1   │  Branch 2   │ ...   │         │
│  │  (共享)     │  (任务0)    │  (任务1)    │  (任务2)    │       │         │
│  └─────────────┴─────────────┴─────────────┴─────────────┴───────┘         │
│  BIM: [-1,-1,...,-1, 0,0,...,0, 1,1,...,1, 2,2,...,2, ...]                  │
│                                                                             │
│  Position IDs:                                                              │
│  [0,1,2,...,prefix_len-1, prefix_len,..., prefix_len,..., prefix_len,...]  │
│                      ↑ 各分支从 prefix_len 开始独立计数                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Draft Tree 生成的关键处理

**这是 BIM 模式的核心难点，必须正确处理！**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                Draft Tree 生成流程 (BIM 模式)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  假设：5 个分支，每个分支 expand_root 后生成 top_k=10 个 token               │
│                                                                             │
│  Step 1: expand_root                                                        │
│  ────────────────────                                                       │
│  输入：5 个分支的 tip token，构成单序列                                      │
│        input_ids: [1, 5] (5个分支的tip token拉平成单序列)                   │
│                                                                             │
│  Draft Model Forward (保持单序列！):                                        │
│        输入: [1, 5] (单序列，5个token)                                      │
│        BIM: [0, 1, 2, 3, 4] (各token对应的分支ID)                           │
│        Position: [P0, P1, P2, P3, P4] (各分支当前的position)                │
│        Attention Mask: BIM mask (各token只能看自己分支的history)            │
│        输出: [1, 5, vocab] → 每个位置取 top_k=10                            │
│                                                                             │
│  此时 Draft KV Cache 追加了 5 个位置（单序列方式）                           │
│                                                                             │
│  Step 2: grow_tree (关键！)                                                 │
│  ────────────────────────                                                   │
│  现在有 5 × 10 = 50 个 token 需要计算                                       │
│                                                                             │
│  ⚠️ 关键点：这 50 个 token 构成一个 sequence 输入 Draft Model               │
│                                                                             │
│  输入构造：                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Branch 0 的 10 个 token | Branch 1 的 10 个 | ... | Branch 4 的 10 个 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│  input_ids: [1, 50] 或 [50] 拉平                                            │
│                                                                             │
│  临时状态管理：                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 临时 BIM (draft_bim):                                             │  │
│  │    [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3, 4,4,...,4]           │  │
│  │     ↑ 10个    ↑ 10个    ↑ 10个    ↑ 10个    ↑ 10个                   │  │
│  │                                                                      │  │
│  │ 2. 临时 Position IDs:                                                │  │
│  │    各分支从 expand_root 后的位置继续递增                              │  │
│  │    如果 expand_root 后位置是 P，则:                                   │  │
│  │    [P+1, P+1, ..., P+1, P+1, ..., ...]                               │  │
│  │    注意：同一层的所有 token position 相同！                           │  │
│  │                                                                      │  │
│  │ 3. 临时 Attention Mask:                                              │  │
│  │    - 每个 token 只能看到自己分支的 history                            │  │
│  │    - 树结构内部的父子关系需要通过 tree_mask 体现                       │  │
│  │    - 构造: cross_branch_mask (BIM) + intra_tree_mask (树结构)         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Draft Model Forward:                                                       │
│  输入: [1, 50, hidden_dim]                                                  │
│  KV Cache: 追加 50 个位置到临时 cache                                        │
│  输出: [1, 50, hidden_dim] → 计算 logits → top_k 选择                       │
│                                                                             │
│  Step 3: 临时 Cache 处理                                                    │
│  ────────────────────────                                                   │
│  ⚠️ grow_tree 阶段的 Cache 是临时的！                                       │
│  - expand_root 的 cache 需要保留（用于 verify 后恢复）                       │
│  - grow_tree 的 cache 在每轮结束后丢弃                                       │
│  - 实现方式：记录 expand_root 后的 cache_length，grow_tree 结束后 truncate   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 BIM 模式代码示例

```python
class DrafterBIM:
    """BIM 模式下的 Drafter - 始终保持单序列！"""
    
    def generate_draft_tree(
        self,
        hidden_states: torch.Tensor,  # [1, total_len, hidden]
        state_manager: BranchStateManager,
    ):
        """
        生成 draft tree (BIM 模式)
        
        核心原则：始终是单序列，通过 BIM 索引区分不同分支
        
        Args:
            hidden_states: Base Model 输出，包含所有分支
            state_manager: 统一状态管理器
        """
        # =========================================================
        # Step 1: expand_root (单序列！)
        # =========================================================
        
        # 从 BIM 序列中获取各分支 tip 的索引
        tip_indices = state_manager.get_branch_tip_indices()  # [num_branches]
        num_branches = len(tip_indices)
        
        # 提取各分支 tip 的 input_ids（构成单序列）
        tip_token_ids = state_manager.get_tip_token_ids()  # [num_branches]
        
        # 构造单序列输入: [1, num_branches]
        expand_input_ids = tip_token_ids.unsqueeze(0)  # [1, 5]
        
        # 构造 BIM: 各 token 对应的分支 ID
        expand_bim = torch.arange(num_branches, device=self.device)  # [0, 1, 2, 3, 4]
        
        # 构造 Position IDs: 各分支当前的 position
        expand_positions = state_manager.get_tip_positions().unsqueeze(0)  # [1, 5]
        
        # 构造 Attention Mask: 各 token 只能看自己分支的 history
        expand_mask = self._build_expand_root_mask(
            query_bim=expand_bim,
            history_bim=state_manager.bim[:state_manager.current_length],
        )
        
        # Draft Model forward (单序列！)
        draft_output = self.draft_model(
            input_ids=expand_input_ids,           # [1, 5]
            position_ids=expand_positions,        # [1, 5]
            attention_mask=expand_mask,
            past_key_values=self.draft_kv_cache,
            use_cache=True,
        )
        
        # 记录 expand_root 后的 cache 长度（用于后续 truncate）
        expand_root_cache_len = self.draft_kv_cache.current_length
        
        # Top-k 选择: [1, 5, vocab] -> 每个位置取 top_k=10
        logits = draft_output.logits  # [1, 5, vocab]
        top_k_scores, top_k_tokens = logits.topk(self.top_k, dim=-1)
        # top_k_tokens: [1, 5, 10]
        
        # =========================================================
        # Step 2: grow_tree (单序列！50 个 token 拉平)
        # =========================================================
        
        for depth in range(self.tree_depth - 1):
            # 将 5 × 10 = 50 个 token 拉平成单序列
            # [1, 5, 10] → [1, 50]
            flat_tokens = top_k_tokens.view(1, -1)  # [1, 50]
            
            # 构造临时 BIM: 每 10 个 token 对应一个分支
            # [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3, 4,4,...,4]
            draft_bim = torch.repeat_interleave(
                torch.arange(num_branches, device=self.device),
                self.top_k
            )  # [50]
            
            # 构造临时 Position IDs
            # 同一深度、同一分支的所有候选 token 共享相同 position
            base_positions = state_manager.get_tip_positions()  # [5]
            current_position = base_positions + depth + 1
            draft_positions = torch.repeat_interleave(
                current_position, self.top_k
            ).unsqueeze(0)  # [1, 50]
            
            # 构造临时 Attention Mask
            # - 每个 token 只能看到自己分支的 history (通过 BIM)
            # - 树结构内部的父子关系 (通过 tree_mask)
            draft_attention_mask = self._build_grow_tree_mask(
                query_bim=draft_bim,
                history_bim=state_manager.bim[:state_manager.current_length],
                tree_structure=self.tree_structure,
            )
            
            # Draft Model forward (单序列！)
            draft_output = self.draft_model(
                input_ids=flat_tokens,               # [1, 50]
                position_ids=draft_positions,        # [1, 50]
                attention_mask=draft_attention_mask,
                past_key_values=self.draft_kv_cache,
                use_cache=True,
            )
            
            # Top-k 选择，更新 tree_structure...
            # ...
        
        # =========================================================
        # Step 3: 清理临时 Cache
        # =========================================================
        
        # 将 draft cache 恢复到 expand_root 后的长度
        # grow_tree 的 cache 是临时的，不保留
        self.draft_kv_cache.truncate(expand_root_cache_len)
        
        return draft_tokens, tree_structure, tree_mask
```

---

## 五、Batching 模式详解

### 5.1 核心概念

**正常 batch 推理，Base Model 和 Draft Layer 都按 batch 方式。关键点在于对齐。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Batching 模式内存布局                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KV Cache: [num_branches, num_heads, max_seq_len, head_dim]                 │
│                                                                             │
│  每个分支独立的 cache:                                                       │
│  Branch 0: [prefix_cache | branch_0_cache | ...]                            │
│  Branch 1: [prefix_cache | branch_1_cache | ...]                            │
│  Branch 2: [prefix_cache | branch_2_cache | ...]                            │
│  ...                                                                        │
│                                                                             │
│  ⚠️ prefix_cache 需要复制 num_branches 份（初始化时一次性复制）              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 对齐机制 (关键！)

**不同分支的接收长度不同时，需要对齐。对齐时必须正确管理 mask 和 position。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Batching 模式对齐机制                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  场景：Verify 后各分支接收长度不同                                           │
│  ───────────────────────────────────                                        │
│  Branch 0: 接收 3 个 token                                                  │
│  Branch 1: 接收 1 个 token                                                  │
│  Branch 2: 接收 2 个 token                                                  │
│                                                                             │
│  对齐前：                                                                    │
│  Branch 0: [t0, t1, t2]                                                     │
│  Branch 1: [t0]                                                             │
│  Branch 2: [t0, t1]                                                         │
│                                                                             │
│  对齐后（左填充）：                                                          │
│  max_len = 3                                                                │
│  Branch 0: [t0, t1, t2]       padding_mask: [1, 1, 1]                       │
│  Branch 1: [PAD, PAD, t0]     padding_mask: [0, 0, 1]                       │
│  Branch 2: [PAD, t0, t1]      padding_mask: [0, 1, 1]                       │
│                                                                             │
│  状态管理：                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ AlignmentState:                                                      │  │
│  │   - padding_lengths: [0, 2, 1]  # 各分支的 padding 长度               │  │
│  │   - valid_lengths: [3, 1, 2]    # 各分支的有效长度                    │  │
│  │   - padding_mask: [3, 3]        # 批量 padding mask                   │  │
│  │   - position_offsets: [0, 2, 1] # 位置偏移（用于计算 position_ids）   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Position IDs 计算：                                                        │
│  假设各分支当前 cache 长度: [100, 98, 99]                                   │
│  Branch 0: [100, 101, 102]                                                  │
│  Branch 1: [PAD, PAD, 98]   # PAD 位置的 position 不影响计算                │
│  Branch 2: [PAD, 99, 100]                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Continuous Decoding 场景 (复杂！)

**Prefill 新分支和 Decode 老分支同时进行时，需要特殊处理。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                Continuous Decoding 场景 (Batching 模式)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  场景：新分支 prefill (200 tokens) + 老分支 decode (2 tokens) 同时进行       │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│  初始状态：                                                                  │
│  - 新分支 (Branch 2): 需要 prefill 200 个 token                             │
│  - 老分支 (Branch 0, 1): 各自需要 decode 2 个 token                         │
│  - 老分支当前 cache 长度: 150 (Batching 下所有分支 cache 长度一致!)          │
│                                                                             │
│  ⚠️ 关键理解：Batching 模式下所有分支的 cache 长度始终一致                   │
│  （即使有 padding，长度也是统一的）                                          │
│                                                                             │
│  Step 1: 为新分支创建等长 Cache                                             │
│  ─────────────────────────────                                              │
│  新分支 cache 长度 = 老分支 cache 长度 = 150                                │
│                                                                             │
│  新分支 cache 布局:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ [0:50]   = prefix_cache (复制共享 prefix)                       │       │
│  │ [50:150] = padding (100 个位置的 padding)                       │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Step 2: 输入对齐（左填充）                                                 │
│  ─────────────────────────                                                  │
│  新分支输入: 200 个 token (prompt)                                          │
│  老分支输入: 2 个 token (decode)                                            │
│  max_input_len = 200                                                        │
│                                                                             │
│  对齐后输入:                                                                 │
│  Branch 0: [PAD × 198, t0, t1]      # 左填充到 200                          │
│  Branch 1: [PAD × 198, t0, t1]      # 左填充到 200                          │
│  Branch 2: [prompt tokens × 200]    # 新分支 prefill                        │
│                                                                             │
│  input_ids: [3, 200]                                                        │
│                                                                             │
│  Step 3: Position IDs 和 Attention Mask                                    │
│  ─────────────────────────────────────                                      │
│  老分支 Position (只有最后 2 个有效):                                        │
│  Branch 0: [PAD, PAD, ..., 150, 151]                                        │
│  Branch 1: [PAD, PAD, ..., 150, 151]                                        │
│                                                                             │
│  新分支 Position (从 prefix_len=50 开始):                                   │
│  Branch 2: [50, 51, 52, ..., 249]                                           │
│                                                                             │
│  Attention Mask:                                                            │
│  - 老分支: padding 位置 mask=0，有效位置 mask=1                             │
│  - 新分支: 全部 mask=1 (causal mask)                                        │
│                                                                             │
│  Step 4: Forward 后的 Cache 长度                                            │
│  ─────────────────────────────                                              │
│  所有分支 cache 长度统一更新: 150 + 200 = 350                               │
│                                                                             │
│  最终 cache 布局:                                                           │
│  Branch 0: [prefix | old_content | PAD×198 | t0, t1]  有效长度=152         │
│  Branch 1: [prefix | old_content | PAD×198 | t0, t1]  有效长度=152         │
│  Branch 2: [prefix | PAD×100 | prompt×200]            有效长度=250         │
│                                                                             │
│  ⚠️ 注意：cache 物理长度 = 350，但各分支的「有效」长度不同                   │
│  需要用 valid_lengths 数组追踪: [152, 152, 250]                             │
│                                                                             │
│  状态管理：                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ ContinuousDecodingState:                                             │  │
│  │   - cache_length: 350              # 统一的 cache 物理长度            │  │
│  │   - valid_lengths: [152, 152, 250] # 各分支的有效长度                 │  │
│  │   - input_lengths: [2, 2, 200]     # 本次输入长度                     │  │
│  │   - is_prefill: [False, False, True]                                 │  │
│  │   - padding_mask: [3, 200]         # 输入的 padding mask             │  │
│  │   - position_ids: [3, 200]         # 各分支独立的 position            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Batching 模式代码示例

```python
class AlignmentManager:
    """Batching 模式对齐管理器"""
    
    def __init__(self, num_branches: int, device: torch.device):
        self.num_branches = num_branches
        self.device = device
        
        # 各分支状态
        self.cache_lengths = torch.zeros(num_branches, dtype=torch.long, device=device)
        self.padding_lengths = torch.zeros(num_branches, dtype=torch.long, device=device)
    
    def align_for_decode(
        self,
        accept_lengths: torch.Tensor,  # [num_branches] 各分支接收长度
        accepted_tokens: List[torch.Tensor],  # 各分支接收的 token
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对齐各分支进行下一步 decode
        
        Returns:
            input_ids: [num_branches, max_len] 对齐后的输入
            attention_mask: [num_branches, max_len] padding mask
            position_ids: [num_branches, max_len] 位置编码
        """
        max_len = accept_lengths.max().item()
        
        # 左填充对齐
        input_ids = torch.full(
            (self.num_branches, max_len),
            self.pad_token_id,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (self.num_branches, max_len),
            device=self.device,
        )
        position_ids = torch.zeros(
            (self.num_branches, max_len),
            dtype=torch.long,
            device=self.device,
        )
        
        for i in range(self.num_branches):
            valid_len = accept_lengths[i].item()
            padding_len = max_len - valid_len
            
            # 填充有效 token
            input_ids[i, padding_len:] = accepted_tokens[i]
            attention_mask[i, padding_len:] = 1
            
            # 计算 position
            start_pos = self.cache_lengths[i].item()
            position_ids[i, padding_len:] = torch.arange(
                start_pos, start_pos + valid_len, device=self.device
            )
            
            # 更新状态
            self.padding_lengths[i] = padding_len
            self.cache_lengths[i] += valid_len
        
        return input_ids, attention_mask, position_ids
    
    def align_continuous_decode(
        self,
        new_branch_prompts: List[torch.Tensor],  # 新分支 prefill 内容
        old_branch_tokens: List[torch.Tensor],   # 老分支 decode 内容
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[bool]]:
        """
        对齐 continuous decoding（prefill + decode 混合）
        
        Returns:
            input_ids: 对齐后的输入
            attention_mask: 复杂的 attention mask
            position_ids: 各分支独立的 position
            is_prefill: 各分支是否是 prefill
        """
        # ... 实现见上文描述
```

---

## 六、BranchStateManager 完整设计

```python
class BranchStateManager:
    """
    统一的分支状态管理器

    支持双模式：
    - use_bim_mode=True: In-One-Sequence 模式
    - use_bim_mode=False: Batching 模式
    
    核心原则：
    - 预分配内存，避免动态复制
    - 统一管理 Base Model 和 Draft Model 的状态
    - 提供模式无关的上层接口
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        use_bim_mode: bool = False,
        num_branches: int = 1,
    ):
        # =====================================================
        # 配置
        # =====================================================
        self.capacity = capacity
        self.device = device
        self.use_bim_mode = use_bim_mode
        self.num_branches = num_branches
        
        # =====================================================
        # BIM 模式状态
        # =====================================================
        if use_bim_mode:
            self.bim = torch.full((capacity,), -2, dtype=torch.long, device=device)
            self.position_ids = torch.zeros(capacity, dtype=torch.long, device=device)
            self.current_length = 0
        
        # =====================================================
        # Batching 模式状态
        # =====================================================
        else:
            self.alignment_manager = AlignmentManager(num_branches, device)
        
        # =====================================================
        # 共享状态
        # =====================================================
        self.prefix_len = 0
        self.active_branches: List[int] = []
        self.branch_tips: Dict[int, int] = {}
        self.branch_outputs: Dict[int, List[int]] = {}
        self.branch_cache_lengths: Dict[int, int] = {}  # 各分支的 cache 长度

    # =========================================================
    # 模式无关的上层接口
    # =========================================================
    
    def init_prefix(self, prefix_len: int, prefix_cache: Optional[KVCache] = None):
        """初始化共享前缀"""
        self.prefix_len = prefix_len
        
        if self.use_bim_mode:
            self.bim[:prefix_len] = -1
            self.position_ids[:prefix_len] = torch.arange(prefix_len, device=self.device)
            self.current_length = prefix_len
        else:
            # Batching 模式：记录 prefix_cache，后续复制给各分支
            self._prefix_cache = prefix_cache
    
    def add_branch(self, branch_id: int, tokens: List[int]) -> int:
        """添加分支"""
        if self.use_bim_mode:
            return self._add_branch_bim(branch_id, tokens)
        else:
            return self._add_branch_batching(branch_id, tokens)
    
    def prepare_decode_inputs(
        self,
        accept_lengths: torch.Tensor,
        accepted_tokens: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备 decode 阶段的输入（处理对齐）"""
        if self.use_bim_mode:
            return self._prepare_decode_bim(accept_lengths, accepted_tokens)
        else:
            return self.alignment_manager.align_for_decode(
                accept_lengths, accepted_tokens
            )
    
    # =========================================================
    # BIM 模式内部实现
    # =========================================================
    
    def _add_branch_bim(self, branch_id: int, tokens: List[int]) -> int:
        """BIM 模式：拉平添加，原地写入"""
        start = self.current_length
        length = len(tokens)
        
        # 原地更新 BIM
        self.bim[start:start+length] = branch_id
        
        # Position: 从 prefix_len 开始
        self.position_ids[start:start+length] = torch.arange(
            self.prefix_len, self.prefix_len + length, device=self.device
        )
        
        self.current_length += length
        self.active_branches.append(branch_id)
        self.branch_tips[branch_id] = start + length - 1
        self.branch_outputs[branch_id] = tokens.copy()
        self.branch_cache_lengths[branch_id] = self.prefix_len + length
        
        return start
    
    def _prepare_decode_bim(
        self,
        accept_lengths: torch.Tensor,
        accepted_tokens: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BIM 模式：拉平所有接收的 token"""
        flat_tokens = []
        flat_bim = []
        flat_positions = []
        
        for i, branch_id in enumerate(self.active_branches):
            acc_len = accept_lengths[i].item()
            if acc_len == 0:
                continue
            
            tokens = accepted_tokens[i][:acc_len]
            flat_tokens.append(tokens)
            flat_bim.extend([branch_id] * acc_len)
            
            # Position: 从当前 tip 继续
            current_pos = self.position_ids[self.branch_tips[branch_id]].item()
            positions = torch.arange(
                current_pos + 1, current_pos + 1 + acc_len, device=self.device
            )
            flat_positions.append(positions)
            
            # 更新状态
            self._extend_branch_bim(branch_id, tokens.tolist())
        
        input_ids = torch.cat(flat_tokens).unsqueeze(0)  # [1, total_len]
        bim = torch.tensor(flat_bim, device=self.device)
        position_ids = torch.cat(flat_positions).unsqueeze(0)
        
        # 构建 attention mask
        attention_mask = self.build_decode_attention_mask(query_bim=bim)
        
        return input_ids, attention_mask, position_ids
    
    # =========================================================
    # Attention Mask 构建
    # =========================================================
    
    def build_prefill_attention_mask(self, query_len: int, dtype=torch.float32):
        """构建 prefill 阶段的 attention mask"""
        if not self.use_bim_mode:
            raise ValueError("Use standard attention mask for batching mode")
        
        # BIM mask 构建逻辑...
        # (见之前的实现)
    
    def build_decode_attention_mask(self, query_bim: torch.Tensor, dtype=torch.float32):
        """构建 decode 阶段的 attention mask"""
        if not self.use_bim_mode:
            raise ValueError("Use alignment_manager for batching mode")
        
        # BIM mask 构建逻辑...
        # (见之前的实现)
```

---

## 七、实现步骤

### Phase 1: 核心状态管理 (P0)
1. 实现 `BranchStateManager` 完整类
2. 实现 `AlignmentManager` 对齐管理器
3. 实现 KVCache 原地操作接口

### Phase 2: BIM 模式 Draft Tree (P0)
1. 实现 expand_root 的 tip 提取逻辑
2. 实现 grow_tree 的 50 token 合并计算
3. 实现临时 cache 管理（truncate）
4. 实现 draft tree mask 构建

### Phase 3: Batching 模式对齐 (P0)
1. 实现 decode 后对齐逻辑
2. 实现 continuous decoding 对齐
3. 实现 position/mask 状态追踪

### Phase 4: 集成测试 (P1)
1. 单元测试各组件
2. 对比两种模式输出一致性
3. 性能和内存对比

---

## 八、验证方法

### 功能验证
```bash
# BIM 模式
python run_specsot.py --use_bim_mode True --task planning

# Batching 模式
python run_specsot.py --use_bim_mode False --task planning

# 输出一致性检查
python test_mode_consistency.py
```

### 内存验证
```bash
# 检查内存复制次数
python test_memory_copy.py --mode bim
python test_memory_copy.py --mode batching
```

### 单元测试
- 测试 `BranchStateManager` 双模式切换
- 测试 `AlignmentManager` 对齐正确性
- 测试 Draft Tree 50 token 合并计算
- 测试 Continuous Decoding 对齐
- 测试 KVCache 原地操作

---

## 九、风险和注意事项

### 技术风险
1. **BIM Draft Tree**：50 token 合并计算的 mask 构建复杂
2. **Batching 对齐**：Continuous Decoding 的对齐逻辑复杂
3. **内存管理**：确保所有操作都是原地进行，无意外复制

### 缓解措施
1. **详细日志**：记录每次内存操作（copy, truncate 等）
2. **渐进实现**：先实现简单场景，再扩展复杂场景
3. **充分测试**：每个边界情况都要覆盖
