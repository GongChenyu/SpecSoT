# coding=utf-8
"""
分支调度系统 - 数据结构定义

该模块定义分支调度系统所需的核心数据结构：
- DeviceProfile: 设备能力描述（算力、显存、Roofline 拐点等）
- BranchInfo: 分支信息（ID、预估长度、prompt tokens 等）
- DeviceExecutionPlan: 单个设备的执行计划
- SchedulePlan: 全局调度计划
- BranchRuntimeState: 分支运行时状态

设计原则：
- 调度模块独立于分布式，单机也可用
- 适配现有 KV Cache 管理方式（追加写入，不释放空间）
- 支持 Continuous Batching（分支完成后立即加入新分支）
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional, Any


# =============================================================================
# 设备能力描述
# =============================================================================

@dataclass
class DeviceProfile:
    """
    设备能力描述，用于异构调度

    Attributes:
        device_id: 设备ID (rank 或 GPU index)
        compute_capability: 算力指标 (TFLOPS)
        memory_capacity: 显存容量 (GB)
        memory_bandwidth: 显存带宽 (GB/s)
        roofline_inflection: Roofline 拐点 (batch size)
        power_budget: 功耗预算 (W, 可选)
        max_parallel: 该设备最大并行分支数
    """
    device_id: int
    compute_capability: float = 1.0  # 相对算力，默认为 1.0
    memory_capacity: int = 24  # GB
    memory_bandwidth: float = 900.0  # GB/s
    roofline_inflection: int = 4  # batch size
    power_budget: Optional[float] = None
    max_parallel: int = 2  # 默认最大并行 2 个分支

    # 预留扩展字段
    network_bandwidth: Optional[float] = None  # 网络带宽 (GB/s)
    current_utilization: float = 0.0  # 当前利用率 (0-1)

    @classmethod
    def default_single_device(cls, device_id: int = 0) -> "DeviceProfile":
        """创建单机默认设备配置"""
        return cls(device_id=device_id)

    @classmethod
    def from_gpu_info(cls, device_id: int, gpu_name: str) -> "DeviceProfile":
        """
        根据 GPU 名称创建设备配置（简化版）

        TODO: 可扩展为从 pynvml 获取详细信息
        """
        # 简单的 GPU 算力映射
        compute_map = {
            "A100": 19.5,
            "A10": 31.2,
            "V100": 15.7,
            "RTX 4090": 82.6,
            "RTX 3090": 35.6,
            "RTX 3080": 29.8,
        }

        compute = 1.0
        for name, cap in compute_map.items():
            if name in gpu_name:
                compute = cap
                break

        return cls(
            device_id=device_id,
            compute_capability=compute,
        )


# =============================================================================
# 分支信息
# =============================================================================

@dataclass
class BranchInfo:
    """
    分支信息，描述一个待执行的分支

    Attributes:
        branch_id: 分支全局 ID（在所有分支中唯一）
        title: 分支标题（用于日志和调试）
        predicted_length: 预估生成 token 长度
        prompt_tokens: 分支 prompt 的 token IDs
        prompt_length: prompt 长度
        dependencies: 依赖分支 ID 列表（预留，当前为空）
    """
    branch_id: int
    title: str
    predicted_length: int
    prompt_tokens: List[int]
    prompt_length: int = 0
    dependencies: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.prompt_length == 0:
            self.prompt_length = len(self.prompt_tokens)


# =============================================================================
# 设备执行计划
# =============================================================================

@dataclass
class DeviceExecutionPlan:
    """
    单个设备的执行计划

    Attributes:
        device_id: 设备 ID
        execution_batches: 执行批次列表
            - 每个批次内的分支并行执行
            - 批次间串行执行
            - 例如: [[0, 1], [2]] 表示先并行执行分支 0 和 1，完成后再执行分支 2
        assigned_branches: 该设备负责的所有分支 ID
        max_parallel: 最大并行度

    Note:
        execution_batches 是静态调度的初始计划。
        在 Continuous Batching 模式下，实际执行顺序可能动态调整：
        - 当某分支提前完成时，下一批次的分支可以提前加入
        - 但分支的设备分配不会改变
    """
    device_id: int
    execution_batches: List[List[int]] = field(default_factory=list)
    assigned_branches: List[int] = field(default_factory=list)
    max_parallel: int = 2

    @property
    def total_branches(self) -> int:
        """该设备负责的分支总数"""
        return len(self.assigned_branches)

    def get_initial_batch(self) -> List[int]:
        """获取初始执行批次"""
        if self.execution_batches:
            return self.execution_batches[0]
        return []

    def get_pending_branches(self, completed: List[int]) -> List[int]:
        """获取尚未完成的分支列表"""
        return [b for b in self.assigned_branches if b not in completed]


# =============================================================================
# 全局调度计划
# =============================================================================

@dataclass
class SchedulePlan:
    """
    全局调度计划

    Attributes:
        num_branches: 分支总数
        num_devices: 设备总数
        branch_to_device: 分支 ID -> 设备 ID 映射
        device_plans: 设备 ID -> 执行计划映射
        global_branch_info: 所有分支信息列表
        scheduler_type: 调度器类型（用于日志）
    """
    num_branches: int
    num_devices: int
    branch_to_device: Dict[int, int] = field(default_factory=dict)
    device_plans: Dict[int, DeviceExecutionPlan] = field(default_factory=dict)
    global_branch_info: List[BranchInfo] = field(default_factory=list)
    scheduler_type: str = "heuristic"

    def get_device_for_branch(self, branch_id: int) -> int:
        """获取分支所在的设备 ID"""
        return self.branch_to_device.get(branch_id, 0)

    def get_plan_for_device(self, device_id: int) -> Optional[DeviceExecutionPlan]:
        """获取设备的执行计划"""
        return self.device_plans.get(device_id)

    def get_branch_info(self, branch_id: int) -> Optional[BranchInfo]:
        """获取分支信息"""
        for info in self.global_branch_info:
            if info.branch_id == branch_id:
                return info
        return None

    def summary(self) -> str:
        """生成调度计划摘要"""
        lines = [
            f"SchedulePlan (type={self.scheduler_type}):",
            f"  Branches: {self.num_branches}, Devices: {self.num_devices}",
        ]
        for device_id, plan in self.device_plans.items():
            branch_ids = plan.assigned_branches
            lines.append(f"  Device {device_id}: {branch_ids} (batches={plan.execution_batches})")
        return "\n".join(lines)


# =============================================================================
# 分支运行时状态
# =============================================================================

class BranchStatus(IntEnum):
    """分支状态枚举"""
    PENDING = 0      # 等待执行
    PREFILLING = 1   # 正在 Prefill
    DECODING = 2     # 正在 Decode
    COMPLETED = 3    # 已完成


@dataclass
class BranchRuntimeState:
    """
    单个分支的运行时状态

    Attributes:
        branch_id: 分支 ID
        status: 当前状态
        bim_start: 在 BIM 中的起始位置
        bim_end: 在 BIM 中的结束位置（只增不减，完成后不释放）
        generated_tokens: 已生成的 token 列表
        current_position: 当前 position_id
        is_finished: 是否已完成（检测到 EOS）

    Note:
        bim_start/bim_end 标记该分支在 Base Model KV Cache 中的位置范围。
        分支完成后，KV 空间不释放（适配现有 KV Cache 管理方式）。
    """
    branch_id: int
    status: BranchStatus = BranchStatus.PENDING

    # BIM 位置范围
    bim_start: int = 0
    bim_end: int = 0

    # 生成状态
    generated_tokens: List[int] = field(default_factory=list)
    current_position: int = 0

    # 完成标记
    is_finished: bool = False

    @property
    def kv_length(self) -> int:
        """该分支占用的 KV 长度"""
        return self.bim_end - self.bim_start

    @property
    def generated_length(self) -> int:
        """已生成的 token 数量"""
        return len(self.generated_tokens)

    def mark_prefilling(self, bim_start: int):
        """标记开始 Prefill"""
        self.status = BranchStatus.PREFILLING
        self.bim_start = bim_start

    def mark_decoding(self, bim_end: int):
        """标记开始 Decode"""
        self.status = BranchStatus.DECODING
        self.bim_end = bim_end

    def mark_completed(self):
        """标记完成"""
        self.status = BranchStatus.COMPLETED
        self.is_finished = True

    def add_tokens(self, tokens: List[int], new_bim_end: int):
        """添加生成的 tokens 并更新 BIM 位置"""
        self.generated_tokens.extend(tokens)
        self.bim_end = new_bim_end
        self.current_position += len(tokens)
