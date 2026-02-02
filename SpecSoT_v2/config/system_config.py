# coding=utf-8
"""
系统配置模块

提供 SpecSoT 系统的全局配置管理：
- SystemConfig: 系统配置类
- 配置验证和默认值

使用示例：
    >>> config = SystemConfig(
    ...     enable_parallel=True,
    ...     use_bim_mode=True,
    ...     max_parallel=4,
    ... )
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SystemConfig:
    """
    SpecSoT 系统配置
    
    控制系统的全局行为，包括推理模式、并行设置等。
    
    Attributes:
        # 推理模式
        enable_parallel: 是否启用并行分支解码
        use_scheduling: 是否使用分支调度
        use_bim_mode: 是否使用 BIM (In-One-Sequence) 模式
                      True: 单序列多分支（通过 BIM 索引管理）
                      False: Batching 模式（批量并行）
        
        # 分支设置
        max_parallel: 最大并行分支数
        max_branches_per_batch: 单批次最大分支数（Batching 模式）
        
        # 生成设置
        max_new_tokens: 每个分支的最大生成 token 数
        max_skeleton_tokens: 骨架阶段最大 token 数
        
        # 投机解码设置
        total_tokens: Draft Tree 的总 token 数
        depth: Draft Tree 深度
        top_k: 每层 top-k 选择数
        threshold: 接受阈值
        
        # 采样设置
        temperature: 采样温度 (0 表示 greedy)
        top_p: nucleus sampling 阈值
        repetition_penalty: 重复惩罚
        
        # 语义约束
        use_semantic_constraint: 是否使用 FSM 语义约束
        
        # 调试设置
        verbose: 是否输出详细日志
        seed: 随机种子
    """
    # 推理模式
    enable_parallel: bool = True
    use_scheduling: bool = False
    use_bim_mode: bool = True
    
    # 分支设置
    max_parallel: int = 4
    max_branches_per_batch: int = 8
    
    # 生成设置
    max_new_tokens: int = 512
    max_skeleton_tokens: int = 200
    
    # 投机解码设置
    total_tokens: int = 60
    depth: int = 7
    top_k: int = 10
    threshold: float = 1.0
    
    # 采样设置
    temperature: float = 0.0
    top_p: float = 0.0
    repetition_penalty: float = 0.0
    
    # 语义约束
    use_semantic_constraint: bool = False
    
    # 调试设置
    verbose: bool = False
    seed: int = 42
    
    # 扩展配置
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证配置"""
        self._validate()
    
    def _validate(self):
        """验证配置有效性"""
        if self.max_parallel < 1:
            raise ValueError("max_parallel 必须 >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须 >= 1")
        if self.total_tokens < 1:
            raise ValueError("total_tokens 必须 >= 1")
        if self.depth < 1:
            raise ValueError("depth 必须 >= 1")
        if self.top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p 必须在 [0, 1] 范围内")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SystemConfig":
        """从字典创建配置"""
        # 提取已知字段
        known_fields = {
            'enable_parallel', 'use_scheduling', 'use_bim_mode',
            'max_parallel', 'max_branches_per_batch',
            'max_new_tokens', 'max_skeleton_tokens',
            'total_tokens', 'depth', 'top_k', 'threshold',
            'temperature', 'top_p', 'repetition_penalty',
            'use_semantic_constraint', 'verbose', 'seed',
        }
        
        init_kwargs = {}
        extra = {}
        
        for key, value in config_dict.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra[key] = value
        
        init_kwargs['extra'] = extra
        return cls(**init_kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'enable_parallel': self.enable_parallel,
            'use_scheduling': self.use_scheduling,
            'use_bim_mode': self.use_bim_mode,
            'max_parallel': self.max_parallel,
            'max_branches_per_batch': self.max_branches_per_batch,
            'max_new_tokens': self.max_new_tokens,
            'max_skeleton_tokens': self.max_skeleton_tokens,
            'total_tokens': self.total_tokens,
            'depth': self.depth,
            'top_k': self.top_k,
            'threshold': self.threshold,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'use_semantic_constraint': self.use_semantic_constraint,
            'verbose': self.verbose,
            'seed': self.seed,
        }
        result.update(self.extra)
        return result
    
    def update(self, **kwargs) -> "SystemConfig":
        """更新配置，返回新实例"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return SystemConfig.from_dict(config_dict)
    
    @property
    def is_greedy(self) -> bool:
        """是否使用 greedy 采样"""
        return self.temperature < 1e-5
    
    @property
    def use_sampling(self) -> bool:
        """是否使用采样（非 greedy）"""
        return self.temperature >= 1e-5
