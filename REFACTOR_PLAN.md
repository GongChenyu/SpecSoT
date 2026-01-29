# SpecSoT Generate 函数重构规划

## 重构完成状态：✅ COMPLETED

**重构日期**: 2025-01

**重构结果**:
- 文件行数: 4500+ → 3155 (减少约30%)
- 删除了 10 个冗余方法
- 新增 7 个流水线阶段方法 + 2 个控制器 + 5 个分布式通信辅助方法

---

## 一、当前问题分析

### 1.1 现有函数列表及问题

| 函数名 | 功能 | 问题 |
|--------|------|------|
| `generate()` | 主入口 | 仅做分发，逻辑清晰 ✓ |
| `generate_eagle()` | 纯投机解码 | 基础实现，保留 ✓ |
| `generate_local()` | 单机并行 | 包含完整流程，代码冗长 |
| `generate_distributed()` | 分布式并行 | 代码最长，与local大量重复 |
| `_generate_skeleton()` | 骨架生成+解析 | 已抽象，但只用于local |
| `_prepare_parallel_batch()` | 准备并行批次 | 已抽象 ✓ |
| `_parallel_decode_naive()` | 朴素并行解码 | 已抽象，但未被充分复用 |
| `_parallel_decode_continuous()` | 连续批处理解码 | 已抽象，但未被充分复用 |
| `_execute_parallel_branches_local()` | 分布式master执行分支 | 与`_parallel_decode_naive`重复 |
| `_execute_parallel_branches_with_batching()` | 分布式master批处理执行 | 与`_parallel_decode_continuous`重复 |
| `_worker_execute_distributed_naive()` | Worker朴素模式 | 大量重复逻辑 |
| `_worker_execute_distributed_scheduling()` | Worker调度模式 | 大量重复逻辑 |

### 1.2 核心问题
1. **代码重复严重**: `generate_local`、`generate_distributed`、worker方法之间存在大量重复
2. **阶段耦合**: 各阶段代码混在一起，难以维护
3. **扩展困难**: 添加新功能需要修改多处代码

## 二、重构目标

### 2.1 设计原则
1. **单一职责**: 每个方法只做一件事
2. **开闭原则**: 对扩展开放，对修改关闭
3. **依赖倒置**: 高层模块不依赖低层模块，都依赖抽象

### 2.2 流水线架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           generate() - 主控制器                              │
│  根据 enable_parallel 分流:                                                   │
│  - False → generate_eagle() (保留原有实现)                                    │
│  - True  → _run_specsot_pipeline() (新的流水线)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     _run_specsot_pipeline() - 流水线控制器                     │
│                                                                              │
│  Phase 1: Skeleton Prefill    → _skeleton_prefill()                          │
│  Phase 2: Skeleton Decode     → _skeleton_decode()                           │
│  Phase 3: Skeleton Parse      → _skeleton_parse()                            │
│  Phase 4: Schedule (可选)     → _schedule_branches()                         │
│  Phase 5: Parallel Prefill    → _parallel_prefill()                          │
│  Phase 6: Parallel Decode     → _parallel_decode()                           │
│  Phase 7: Result Merge        → _merge_results()                             │
│                                                                              │
│  注意: 分布式Worker有独立流程 → _worker_pipeline()                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 三、阶段详细设计

### Phase 1: Skeleton Prefill (`_skeleton_prefill`)

**功能**: 初始化KV Cache，执行Skeleton的Prefill

**参数变体**:
- `distributed=False`: 本地Prefill
- `distributed=True`: 分布式Prefill (所有rank参与)

```python
def _skeleton_prefill(
    self,
    input_ids: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    max_new_tokens: int,
    distributed: bool = False,
) -> SkeletonPrefillResult:
    """
    Returns:
        SkeletonPrefillResult: 包含 draft_tokens, tree_mask 等
    """
```

### Phase 2: Skeleton Decode (`_skeleton_decode`)

**功能**: 执行骨架解码循环

**说明**: 此阶段在分布式模式下仅Master执行

```python
def _skeleton_decode(
    self,
    input_ids: torch.Tensor,
    prefill_result: SkeletonPrefillResult,
    logits_processor: Optional[LogitsProcessorList],
    max_steps: int = 200,
) -> SkeletonDecodeResult:
    """
    Returns:
        SkeletonDecodeResult: 包含 skeleton_ids, skeleton_text
    """
```

### Phase 3: Skeleton Parse (`_skeleton_parse`)

**功能**: 解析骨架输出，提取分支信息

```python
def _skeleton_parse(
    self,
    skeleton_text: str,
    task_prompt: str,
) -> SkeletonParseResult:
    """
    Returns:
        SkeletonParseResult: 包含 mode, tasks, clean_branches
    """
```

### Phase 4: Schedule (`_schedule_branches`)

**功能**: 根据分支信息生成调度计划

**参数变体**:
- `use_scheduling=False`: 简单均分 (Naive)
- `use_scheduling=True`: 启发式调度 (Heuristic)
- `distributed=True`: 多设备调度

```python
def _schedule_branches(
    self,
    branch_infos: List[BranchInfo],
    use_scheduling: bool,
    distributed: bool,
    max_parallel: int,
) -> ScheduleResult:
    """
    Returns:
        ScheduleResult: 包含 schedule_plan, my_branch_ids
    """
```

### Phase 5: Parallel Prefill (`_parallel_prefill`)

**功能**: 为并行分支执行Prefill

**说明**: 复用已有的 `_prepare_parallel_batch` + `prefill_parallel`

```python
def _parallel_prefill(
    self,
    prefix_ids: torch.Tensor,
    branch_prompts: List[List[int]],
    branch_ids: List[int],
    max_new_tokens: int,
    logits_processor: Optional[LogitsProcessorList],
) -> ParallelPrefillResult:
    """
    Returns:
        ParallelPrefillResult: 包含 draft_tokens, tree_mask 等
    """
```

### Phase 6: Parallel Decode (`_parallel_decode`)

**功能**: 执行并行分支解码

**参数变体**:
- `use_scheduling=False`: 朴素模式 - 所有分支同时解码
- `use_scheduling=True`: 连续批处理模式 - 动态加入新分支

```python
def _parallel_decode(
    self,
    prefill_result: ParallelPrefillResult,
    branch_infos: List[BranchInfo],
    schedule_result: ScheduleResult,
    max_new_tokens: int,
    max_kv_len: int,
    use_scheduling: bool,
    logits_processor: Optional[LogitsProcessorList],
) -> ParallelDecodeResult:
    """
    Returns:
        ParallelDecodeResult: 包含 branch_outputs, stats
    """
```

### Phase 7: Result Merge (`_merge_results`)

**功能**: 合并骨架和分支输出

```python
def _merge_results(
    self,
    skeleton_ids: torch.Tensor,
    branch_outputs: List[List[int]],
    tasks: List[Dict],
    instruction_len: Dict[int, int],
) -> torch.Tensor:
    """
    Returns:
        merged_ids: 合并后的token IDs
    """
```

## 四、分布式特殊处理

### 4.1 Master流程

```python
def _run_specsot_pipeline(self, ...):
    # Phase 1: Skeleton Prefill (分布式)
    prefill_result = self._skeleton_prefill(..., distributed=True)
    
    # Phase 2-3: 仅Master执行骨架解码和解析
    skeleton_result = self._skeleton_decode(...)
    parse_result = self._skeleton_parse(...)
    
    # 处理 direct/error 模式
    if parse_result.mode != "plan":
        self._notify_workers_complete()  # 通知Worker结束
        return ...
    
    # Phase 4: 调度
    schedule_result = self._schedule_branches(..., distributed=True)
    
    # 分发任务给Worker
    self._distribute_tasks_to_workers(schedule_result, parse_result)
    
    # Phase 5-6: Master执行自己的分支
    my_outputs = self._execute_my_branches(...)
    
    # 收集Worker结果
    all_outputs = self._collect_worker_results(...)
    
    # Phase 7: 合并结果
    return self._merge_results(...)
```

### 4.2 Worker流程

```python
def _worker_pipeline(self, ...):
    # Phase 1: Skeleton Prefill (参与分布式Prefill)
    self._skeleton_prefill(..., distributed=True)
    
    # 等待接收任务
    task_data = self._receive_tasks_from_master()
    if task_data is None:
        return  # Master发送了完成信号
    
    # Phase 5-6: 执行分配的分支
    my_outputs = self._execute_assigned_branches(task_data, ...)
    
    # 发送结果给Master
    self._send_results_to_master(my_outputs)
```

## 五、数据结构定义

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch

@dataclass
class SkeletonPrefillResult:
    """骨架Prefill结果"""
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    input_ids: torch.Tensor
    input_len: int
    max_kv_len: int
    saved_eagle_kv: Optional[tuple] = None  # 用于恢复

@dataclass
class SkeletonDecodeResult:
    """骨架解码结果"""
    skeleton_ids: torch.Tensor
    skeleton_text: str
    decode_time: float

@dataclass
class SkeletonParseResult:
    """骨架解析结果"""
    mode: str  # 'plan', 'direct', 'error'
    tasks: Optional[List[Dict]]  # 分支任务列表
    clean_branches: Optional[List[List[int]]]  # 分支prompt tokens
    instruction_len: Optional[Dict[int, int]]
    error_msg: Optional[str] = None

@dataclass
class ScheduleResult:
    """调度结果"""
    schedule_plan: Any  # SchedulePlan对象
    my_branch_ids: List[int]  # 当前设备分配的分支
    branch_infos: List[Any]  # BranchInfo列表

@dataclass
class ParallelPrefillResult:
    """并行Prefill结果"""
    input_ids: torch.Tensor
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor
    tips_indices: torch.Tensor
    branch_begins: List[int]
    branch_lengths: List[int]

@dataclass
class ParallelDecodeResult:
    """并行解码结果"""
    branch_outputs: Dict[int, List[int]]
    stats: Dict[str, Any]
```

## 六、删除/合并的函数

### 6.1 将被删除的函数
- `generate_local()` - 合并到 `_run_specsot_pipeline()`
- `generate_distributed()` - 合并到 `_run_specsot_pipeline()`
- `_execute_parallel_branches_local()` - 合并到 `_parallel_decode()`
- `_execute_parallel_branches_with_batching()` - 合并到 `_parallel_decode()`
- `_worker_execute_distributed_naive()` - 合并到 `_worker_pipeline()`
- `_worker_execute_distributed_scheduling()` - 合并到 `_worker_pipeline()`

### 6.2 将被保留/重构的函数
- `generate()` - 保留，简化为分流器
- `generate_eagle()` - 保留原有实现
- `_generate_skeleton()` - 拆分为 `_skeleton_prefill()` + `_skeleton_decode()`
- `_prepare_parallel_batch()` - 保留，被 `_parallel_prefill()` 调用
- `_parallel_decode_naive()` - 合并到 `_parallel_decode()`
- `_parallel_decode_continuous()` - 合并到 `_parallel_decode()`
- `_broadcast_parallel_complete_signal()` - 重命名为 `_notify_workers_complete()`

## 七、重构后的代码结构

```python
class SpecSoTModel(nn.Module):
    # =========================================================================
    # 数据结构定义
    # =========================================================================
    # (在文件顶部或单独模块中定义dataclass)
    
    # =========================================================================
    # 主入口
    # =========================================================================
    def generate(self, ...) -> Tuple[torch.Tensor, Dict]:
        """统一入口，分流到eagle或specsot流水线"""
        
    def generate_eagle(self, ...) -> Tuple[torch.Tensor, Dict]:
        """纯投机解码（保留原实现）"""
    
    # =========================================================================
    # SpecSoT 流水线
    # =========================================================================
    def _run_specsot_pipeline(self, ...) -> Tuple[torch.Tensor, Dict]:
        """SpecSoT主流水线控制器"""
        
    def _worker_pipeline(self, ...) -> Tuple[torch.Tensor, Dict]:
        """分布式Worker流水线"""
    
    # =========================================================================
    # Phase 1: Skeleton Prefill
    # =========================================================================
    def _skeleton_prefill(self, ...) -> SkeletonPrefillResult:
        """骨架Prefill阶段"""
    
    # =========================================================================
    # Phase 2: Skeleton Decode
    # =========================================================================
    def _skeleton_decode(self, ...) -> SkeletonDecodeResult:
        """骨架解码阶段"""
    
    # =========================================================================
    # Phase 3: Skeleton Parse
    # =========================================================================
    def _skeleton_parse(self, ...) -> SkeletonParseResult:
        """骨架解析阶段"""
    
    # =========================================================================
    # Phase 4: Schedule
    # =========================================================================
    def _schedule_branches(self, ...) -> ScheduleResult:
        """分支调度阶段"""
    
    # =========================================================================
    # Phase 5: Parallel Prefill
    # =========================================================================
    def _parallel_prefill(self, ...) -> ParallelPrefillResult:
        """并行Prefill阶段"""
    
    # =========================================================================
    # Phase 6: Parallel Decode
    # =========================================================================
    def _parallel_decode(self, ...) -> ParallelDecodeResult:
        """并行解码阶段（支持naive/continuous模式）"""
    
    # =========================================================================
    # Phase 7: Result Merge
    # =========================================================================
    def _merge_results(self, ...) -> torch.Tensor:
        """结果合并阶段"""
    
    # =========================================================================
    # 分布式通信辅助
    # =========================================================================
    def _notify_workers_complete(self) -> None:
        """通知Worker完成"""
        
    def _distribute_tasks_to_workers(self, ...) -> None:
        """分发任务给Worker"""
        
    def _collect_worker_results(self, ...) -> Dict[int, List[int]]:
        """收集Worker结果"""
        
    def _receive_tasks_from_master(self, ...) -> Optional[TaskData]:
        """Worker接收任务"""
        
    def _send_results_to_master(self, ...) -> None:
        """Worker发送结果"""
```

## 八、实施步骤

### Step 1: 定义数据结构
创建dataclass定义各阶段的输入输出

### Step 2: 实现各阶段方法
按Phase 1-7依次实现，确保每个方法职责单一

### Step 3: 实现流水线控制器
实现 `_run_specsot_pipeline()` 和 `_worker_pipeline()`

### Step 4: 更新主入口
更新 `generate()` 调用新的流水线

### Step 5: 删除旧代码
删除不再需要的函数

### Step 6: 测试验证
运行测试确保功能正确

## 九、预期效果

### 重构前
- 函数数量: 12+
- 代码行数: ~1500行 (generate相关)
- 重复代码: 约40%

### 重构后
- 函数数量: 10-12
- 代码行数: ~800行
- 重复代码: <5%

### 可维护性提升
- 添加新调度算法: 只需修改 `_schedule_branches()`
- 添加新解码模式: 只需扩展 `_parallel_decode()`
- 修改骨架解析: 只需修改 `_skeleton_parse()`
