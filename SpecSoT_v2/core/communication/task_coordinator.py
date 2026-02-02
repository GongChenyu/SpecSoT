# coding=utf-8
"""
分布式任务协调器 (Distributed Task Coordinator)

该模块负责 Master 和 Worker 之间的任务分发和结果收集，
将通信逻辑从主模型中分离出来，提高代码的模块化和可维护性。

主要功能：
1. Master -> Worker: 分发调度计划和分支任务
2. Worker -> Master: 发送计算结果
3. Master <-> Worker: 同步完成信号

使用示例：
    # Master 侧
    coordinator = DistributedTaskCoordinator(comm, logger, rank=0, is_master=True)
    coordinator.distribute_tasks(schedule_result, parse_result)
    results = coordinator.collect_results(schedule_result, num_branches)
    coordinator.notify_complete()
    
    # Worker 侧
    coordinator = DistributedTaskCoordinator(comm, logger, rank=1, is_master=False)
    plan, branches = coordinator.receive_tasks()
    coordinator.send_results(branch_outputs)
"""

from typing import Dict, List, Tuple, Optional, Any
import logging

# 导入调度相关的数据结构
from ..scheduling import (
    BranchInfo,
    DeviceExecutionPlan,
    SchedulePlan,
)


class DistributedTaskCoordinator:
    """
    分布式任务协调器
    
    封装 Master 和 Worker 之间的所有通信逻辑，包括：
    - 任务分发 (Master -> Workers)
    - 结果收集 (Workers -> Master)
    - 完成信号同步
    
    Attributes:
        comm: 通信管理器实例 (ZMQCommManagerBase)
        logger: 日志记录器
        rank: 当前节点的 rank
        is_master: 是否为 Master 节点
    """
    
    def __init__(
        self,
        comm,  # ZMQCommManagerBase 实例
        logger: logging.Logger,
        rank: int = 0,
        is_master: bool = True,
    ):
        """
        初始化任务协调器
        
        Args:
            comm: 通信管理器实例
            logger: 日志记录器
            rank: 当前节点的 rank
            is_master: 是否为 Master 节点
        """
        self.comm = comm
        self.logger = logger
        self.rank = rank
        self.is_master = is_master
    
    # =========================================================================
    # Master 侧方法
    # =========================================================================
    
    def distribute_tasks(
        self,
        schedule_result,  # ScheduleResult 对象
        parse_result,     # SkeletonParseResult 对象
    ) -> None:
        """
        Master 分发任务给 Workers
        
        该方法将调度计划和分支 Prompt 发送给所有 Worker 节点。
        
        Args:
            schedule_result: 调度结果，包含调度计划和分支信息
            parse_result: 骨架解析结果（当前未使用，预留接口）
        """
        if not self.is_master:
            self.logger.warning("只有 Master 可以调用 distribute_tasks")
            return
        
        schedule_plan = schedule_result.schedule_plan
        branch_infos = schedule_result.branch_infos
        
        # 序列化调度计划
        schedule_plan_data = {
            'num_branches': schedule_plan.num_branches,
            'num_devices': schedule_plan.num_devices,
            'branch_to_device': schedule_plan.branch_to_device,
            'device_plans': {
                did: {
                    'device_id': plan.device_id,
                    'execution_batches': plan.execution_batches,
                    'assigned_branches': plan.assigned_branches,
                    'max_parallel': plan.max_parallel,
                }
                for did, plan in schedule_plan.device_plans.items()
            },
            'scheduler_type': schedule_plan.scheduler_type,
        }
        
        # 广播调度计划
        self.logger.info(f"广播调度计划到所有 Worker...")
        self.comm.send_schedule_plan_async(schedule_plan_data, dst_rank=-1)
        
        # 发送分支 Prompt 到各 Worker
        device_branches = {}
        for info in branch_infos:
            device_id = schedule_plan.get_device_for_branch(info.branch_id)
            if device_id not in device_branches:
                device_branches[device_id] = []
            device_branches[device_id].append(info)
        
        for device_id, infos in device_branches.items():
            if device_id == 0:
                continue  # Master 不发给自己
            branch_data = [
                {
                    'branch_id': info.branch_id,
                    'title': info.title,
                    'predicted_length': info.predicted_length,
                    'prompt_tokens': info.prompt_tokens,
                }
                for info in infos
            ]
            self.logger.info(f"发送 {len(infos)} 个分支任务到设备 {device_id}")
            self.comm.send_branch_prompt_async(branch_data, device_id)
    
    def collect_results(
        self,
        schedule_result,  # ScheduleResult 对象
        num_branches: int,
    ) -> Dict[int, List[int]]:
        """
        Master 收集 Workers 的计算结果
        
        Args:
            schedule_result: 调度结果（当前未使用，预留接口）
            num_branches: 总分支数
            
        Returns:
            outputs: {branch_id: output_tokens} 字典
        """
        if not self.is_master:
            self.logger.warning("只有 Master 可以调用 collect_results")
            return {}
        
        if num_branches == 0:
            return {}
        
        self.logger.info(f"收集其他设备的分支输出 ({num_branches} 个)...")
        
        outputs = self.comm.collect_all_branch_outputs(
            num_branches=num_branches,
            timeout=300.0
        )
        return outputs
    
    def notify_complete(self) -> None:
        """Master 通知所有 Worker 任务完成"""
        if not self.is_master:
            self.logger.warning("只有 Master 可以调用 notify_complete")
            return
        
        self.logger.info("通知所有 Worker 任务完成")
        self.comm.broadcast_parallel_complete_signal()
    
    # =========================================================================
    # Worker 侧方法
    # =========================================================================
    
    def receive_tasks(self) -> Optional[Tuple[SchedulePlan, List[BranchInfo]]]:
        """
        Worker 接收 Master 分发的任务
        
        Returns:
            (schedule_plan, branch_infos): 调度计划和分支信息列表
            None: 如果收到完成信号或超时
        """
        if self.is_master:
            self.logger.warning("Master 不应该调用 receive_tasks")
            return None
        
        # 接收调度计划
        self.logger.info(f"Worker {self.rank} 等待接收调度计划...")
        schedule_plan_data = self.comm.recv_schedule_plan(timeout=60.0)
        if schedule_plan_data is None:
            self.logger.info(f"Worker {self.rank} 未收到调度计划（超时或完成信号）")
            return None
        
        # 反序列化调度计划
        device_plans = {}
        for did, pdata in schedule_plan_data['device_plans'].items():
            device_plans[int(did)] = DeviceExecutionPlan(
                device_id=pdata['device_id'],
                execution_batches=pdata['execution_batches'],
                assigned_branches=pdata['assigned_branches'],
                max_parallel=pdata['max_parallel'],
            )
        schedule_plan = SchedulePlan(
            num_branches=schedule_plan_data['num_branches'],
            num_devices=schedule_plan_data['num_devices'],
            branch_to_device=schedule_plan_data['branch_to_device'],
            device_plans=device_plans,
            scheduler_type=schedule_plan_data['scheduler_type'],
        )
        
        # 接收分支 Prompt
        self.logger.info(f"Worker {self.rank} 等待接收分支 Prompt...")
        branch_data_list = self.comm.recv_branch_prompts(timeout=60.0)
        if not branch_data_list:
            self.logger.warning(f"Worker {self.rank} 未收到分支数据")
            return None
        
        branch_infos = [
            BranchInfo(
                branch_id=data['branch_id'],
                title=data['title'],
                predicted_length=data['predicted_length'],
                prompt_tokens=data['prompt_tokens'],
            )
            for data in branch_data_list
        ]
        
        self.logger.info(f"Worker {self.rank} 收到 {len(branch_infos)} 个分支任务")
        return schedule_plan, branch_infos
    
    def send_results(self, branch_outputs: Dict[int, List[int]]) -> None:
        """
        Worker 发送计算结果给 Master
        
        Args:
            branch_outputs: {branch_id: output_tokens} 字典
        """
        if self.is_master:
            self.logger.warning("Master 不应该调用 send_results")
            return
        
        self.logger.info(f"Worker {self.rank} 发送 {len(branch_outputs)} 个分支结果到 Master")
        for bid, tokens in branch_outputs.items():
            self.comm.send_branch_output_async(
                branch_id=bid,
                output_tokens=tokens,
                dst_rank=0,
            )
    
    def wait_complete_signal(self, timeout: float = 300.0) -> bool:
        """
        Worker 等待 Master 的完成信号
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功收到完成信号
        """
        if self.is_master:
            self.logger.warning("Master 不应该调用 wait_complete_signal")
            return True
        
        self.logger.info(f"Worker {self.rank} 等待完成信号...")
        success = self.comm.recv_complete_signal(timeout=timeout)
        if success:
            self.logger.info(f"Worker {self.rank} 收到完成信号")
        else:
            self.logger.warning(f"Worker {self.rank} 等待完成信号超时")
        return success
