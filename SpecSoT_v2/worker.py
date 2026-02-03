# coding=utf-8
"""
SpecSoT Worker Engine - 推理执行器

该模块实现实际的推理执行逻辑：
1. 加载模型
2. 执行推理（支持标准模式和骨架并行模式）
3. 保存结果

设计原则：
- Worker 负责实际的推理计算
- 支持单机和分布式两种模式
- 日志简洁：控制台只显示关键信息，详细信息写入文件
"""

import os
import time
import torch
import pandas as pd
import logging
from typing import Dict, Any, Optional
from threading import Thread, Event
from dataclasses import dataclass

from .engine.generator import SpecSoTGenerator
from .config.distributed_config import DistributedConfig
from .utils.gpu_monitor import GPUMemoryMonitor
from .utils.logging import get_logger


# =============================================================================
# 数据加载
# =============================================================================

def load_dataset(task: str, project_dir: str):
    """加载评估数据集"""
    # 数据集在 parallel-decoding-in-one-sequence/datasets/ 目录下
    datasets_dir = os.path.join(project_dir, "parallel-decoding-in-one-sequence/datasets")

    if task == "retrieval":
        path = os.path.join(datasets_dir, "student_resume_logic_retrieval/logic_gpa_resume_10.jsonl")
        df = pd.read_json(path, lines=True)
        df['task_prompt'] = df['prompt']
    elif task == "planning":
        path = os.path.join(datasets_dir, "planning/industry_tasks.jsonl")
        df = pd.read_json(path, lines=True)
        df['task_prompt'] = df['task']
    else:  # multi-doc-qa
        path = os.path.join(datasets_dir, "multi-doc-qa/2wikimqa.jsonl")
        df = pd.read_json(path, lines=True)
        df['task_prompt'] = df['input']
    return df, path


# =============================================================================
# Worker Engine
# =============================================================================

class WorkerEngine:
    """
    Worker 引擎 - 负责实际的推理执行
    
    日志策略：
    - INFO: 配置信息、阶段开始/完成、统计结果
    - DEBUG: 详细的中间状态、tensor信息、通信细节
    """

    def __init__(self, args):
        self.args = args
        # 路径计算: engine/worker.py -> engine -> SpecSoT_v2 -> SpecSoT
        specsot_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_dir = os.path.dirname(specsot_v2_dir)  # SpecSoT/
        self.log_dir = os.path.join(self.project_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志
        rank = getattr(args, 'rank', 0)
        self.logger = get_logger(rank=rank, log_dir=self.log_dir)
        
        # 模型（延迟初始化）
        self.model = None
        self.tokenizer = None
        self.distributed_config = None
        
        # 结果存储
        self.results = []

    def run(self):
        """运行 Worker 主流程"""
        self._log_config()
        
        # 设置分布式
        self.distributed_config = self._setup_distributed()
        
        # 加载模型
        self._load_model()
        
        # 加载数据集
        df = self._load_dataset()
        
        # 执行推理
        self._run_inference_loop(df)
        
        # 保存结果
        self._save_results()
        
        # 清理
        self._cleanup()
        
        self.logger.info("Worker 执行完成")

    def _log_config(self):
        """打印配置信息"""
        args = self.args
        self.logger.info("=" * 50)
        self.logger.info("SpecSoT Worker 启动")
        self.logger.info("=" * 50)
        self.logger.info(f"模型: {os.path.basename(args.base_model_path)}")
        self.logger.info(f"Eagle: {'Eagle3' if args.use_eagle3 else 'Eagle2'}")
        self.logger.info(f"任务: {args.task} | 样本数: {args.num_samples}")
        
        # 执行模式说明
        distributed = getattr(args, 'distributed', False)
        scheduling = getattr(args, 'use_scheduling', False)
        parallel = getattr(args, 'enable_parallel', True)
        
        if not distributed and not scheduling:
            mode_desc = "单机标准" + ("(骨架并行)" if parallel else "(纯EAGLE)")
        elif not distributed and scheduling:
            mode_desc = "单机调度(Continuous Batching)"
        elif distributed and not scheduling:
            mode_desc = "分布式标准(贪心任务分配)"
        else:
            mode_desc = "分布式调度(分布式Scheduling)"
        
        self.logger.info(f"执行模式: {mode_desc}")
        self.logger.info(f"  - 分布式: {distributed} | 调度: {scheduling} | 骨架并行: {parallel}")
        
        if distributed:
            self.logger.info(f"  - rank={args.rank}/{args.world_size-1}")
        else:
            device = getattr(args, 'device', 'cuda:0')
            self.logger.info(f"  - 设备: {device}")
        
        self.logger.debug(f"完整配置: {vars(args)}")

    def _setup_distributed(self):
        """设置分布式配置"""
        if not self.args.distributed:
            return None
        
        from SpecSoT_v2.config.distributed_config import DistributedConfig
        
        config = DistributedConfig.from_layer_splits_str(
            layer_splits_str=self.args.layer_splits,
            rank=self.args.rank,
            world_size=self.args.world_size,
            base_port=self.args.base_port,
            comm_mode=self.args.comm_mode,
            chunk_size=self.args.chunk_size,
        )
        self.logger.debug(f"分布式配置: {config}")
        return config

    def _load_model(self):
        """加载模型"""

        self.logger.info("正在加载模型...")
        start = time.time()

        # 设备选择：分布式模式由环境变量控制，非分布式使用 --device 参数
        device = getattr(self.args, 'device', 'cuda:0')
        
        self.model = SpecSoTGenerator.from_pretrained(
            base_model_path=self.args.base_model_path,
            ea_model_path=self.args.eagle_model_path,
            use_eagle3=self.args.use_eagle3,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
            total_token=40,
            depth=4,
            top_k=6,
            seed=self.args.seed,
            distributed_config=self.distributed_config,
            use_bim_mode=getattr(self.args, 'use_bim_mode', True),
        )

        elapsed = time.time() - start
        self.logger.info(f"模型加载完成 ({elapsed:.1f}s)")

        self.tokenizer = self.model.tokenizer
        self.tokenizer.padding_side = "left"

    def _load_dataset(self):
        """加载数据集"""
        df, path = load_dataset(self.args.task, self.project_dir)
        df = df[:self.args.num_samples]
        self.logger.info(f"数据集: {os.path.basename(path)} ({len(df)} 条)")
        return df

    def _run_inference_loop(self, df):
        """执行推理循环"""
        monitor_device = torch.cuda.current_device() if torch.cuda.is_available() else 0

        for i in range(len(df)):
            # task_prompt = df.loc[i, "task_prompt"]
            task_prompt = "Please explain the benefits of playing basketball from six aspects, with each explanation within 100 words."
            self.logger.info(f"\n[样本 {i+1}/{len(df)}] {task_prompt[:60]}...")
            
            result = self._run_single_inference(task_prompt, monitor_device)
            result["prompt"] = task_prompt
            self.results.append(result)
            
            self._log_sample_result(result)

    def _run_single_inference(self, task_prompt: str, device: int) -> Dict[str, Any]:
        """
        执行单个样本的推理

        使用 SpecSoTGenerator.generate() 接口。
        """
        args = self.args
        is_distributed_worker = args.distributed and args.rank != 0

        with GPUMemoryMonitor(device_index=device) as monitor:
            # 调用 model.generate()
            result = self.model.generate(
                task_prompt=task_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=getattr(args, 'temperature', 0.0),
                top_p=getattr(args, 'top_p', 0.0),
                top_k=getattr(args, 'top_k', 0),
                enable_parallel=args.enable_parallel,
                use_scheduling=args.use_scheduling,
                max_parallel=args.max_parallel,
                use_semantic_constraint=args.use_semantic_constraint,
            )
            
            # 从 GenerateResult 中提取信息
            stats = result.stats
            stats['total_time'] = result.total_time

            # 提取统计信息
            inference_time = stats.get('total_time', 0)
            skeleton_time = stats.get('skeleton_time', 0)
            parallel_time = stats.get('parallel_time', 0)
            num_para = stats.get('num_branches', 0)
            avg_accept_len = stats.get('avg_accept_len', 0.0)

        # 分布式 Worker 跳过
        if is_distributed_worker:
            response = ""
            length = 0
        else:
            response = result.output_text
            self.logger.info(f"  响应: {response}...")
            length = result.total_tokens

        throughput = length / inference_time if inference_time > 0 else 0

        return {
            "response": response,
            "inference_time": inference_time,
            "skeleton_time": skeleton_time,
            "parallel_time": parallel_time,
            "memory": monitor.peak_usage,
            "length": length,
            "throughput": throughput,
            "num_para": num_para,
            "avg_accept_len": avg_accept_len,
            "avg_draft_time": stats.get('avg_draft_time', 0.0),
            "avg_verify_time": stats.get('avg_verify_time', 0.0),
            "avg_update_time": stats.get('avg_update_time', 0.0),
            "mode": stats.get('mode', 'unknown'),
            "skeleton_text": stats.get('skeleton_text', ''),
            # 配置信息
            "use_bim_mode": getattr(args, 'use_bim_mode', True),
            "use_scheduling": args.use_scheduling,
            "distributed": args.distributed,
            "seed": args.seed,
        }

    def _log_sample_result(self, result: Dict[str, Any]):
        """打印单个样本的结果"""
        self.logger.info(
            f"  结果: {result['length']} tokens | "
            f"{result['throughput']:.1f} tok/s | "
            f"显存 {result['memory']:.0f}MB"
        )
        if self.args.enable_parallel:
            self.logger.info(
                f"  阶段: skeleton={result['skeleton_time']:.2f}s | "
                f"parallel={result['parallel_time']:.2f}s | "
                f"分支数={result['num_para']}"
            )
        
        # 详细信息写入文件
        self.logger.debug(f"  平均接受长度: {result['avg_accept_len']:.2f}")
        self.logger.debug(f"  平均Draft时间: {result['avg_draft_time']*1000:.2f}ms")
        self.logger.debug(f"  平均Verify时间: {result['avg_verify_time']*1000:.2f}ms")
        self.logger.debug(f"  响应预览: {result['response']}...")

    def _save_results(self):
        """保存结果"""
        if self.args.distributed and self.args.rank != 0:
            return

        import json
        from datetime import datetime

        save_dir = os.path.join(self.project_dir, "results")
        os.makedirs(save_dir, exist_ok=True)

        args = self.args
        mode = "parallel" if args.enable_parallel else "standard"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 移除不可序列化的字段
        save_results = [
            {k: v for k, v in r.items() if k != 'output_ids'}
            for r in self.results
        ]

        # 保存详细结果
        save_path = os.path.join(save_dir, f"results_{mode}_{args.task}.json")
        pd.DataFrame(save_results).to_json(
            save_path, orient='records', indent=4, force_ascii=False
        )
        self.logger.info(f"\n结果已保存: {save_path}")

        # 保存汇总统计
        self._save_summary(save_dir, mode, timestamp, save_results)

        # 打印汇总
        self._log_summary()

    def _save_summary(self, save_dir: str, mode: str, timestamp: str, save_results: list):
        """保存汇总统计文件"""
        import json

        if not self.results:
            return

        args = self.args
        n = len(self.results)

        # 计算汇总统计
        summary = {
            "config": {
                "model": os.path.basename(args.base_model_path),
                "eagle": "EAGLE3" if args.use_eagle3 else "EAGLE2",
                "task": args.task,
                "num_samples": n,
                "seed": args.seed,
                "enable_parallel": args.enable_parallel,
                "use_scheduling": args.use_scheduling,
                "use_bim_mode": getattr(args, 'use_bim_mode', True),
                "distributed": args.distributed,
            },
            "summary": {
                "avg_total_time": sum(r['inference_time'] for r in self.results) / n,
                "avg_throughput": sum(r['throughput'] for r in self.results) / n,
                "avg_output_tokens": sum(r['length'] for r in self.results) / n,
                "avg_memory_mb": sum(r['memory'] for r in self.results) / n,
                "avg_branches": sum(r['num_para'] for r in self.results) / n,
                "avg_accept_len": sum(r['avg_accept_len'] for r in self.results) / n,
            },
            "samples": save_results,
        }

        # 保存汇总文件
        summary_path = os.path.join(
            save_dir, f"summary_{mode}_{args.task}_{timestamp}.json"
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        self.logger.info(f"汇总统计已保存: {summary_path}")

    def _log_summary(self):
        """打印汇总统计"""
        if not self.results:
            return
        
        n = len(self.results)
        avg_time = sum(r['inference_time'] for r in self.results) / n
        avg_throughput = sum(r['throughput'] for r in self.results) / n
        avg_length = sum(r['length'] for r in self.results) / n
        avg_memory = sum(r['memory'] for r in self.results) / n
        
        self.logger.info("\n" + "=" * 50)
        self.logger.info("汇总统计")
        self.logger.info("=" * 50)
        self.logger.info(f"样本数: {n}")
        self.logger.info(f"平均推理时间: {avg_time:.2f}s")
        self.logger.info(f"平均吞吐量: {avg_throughput:.1f} tokens/s")
        self.logger.info(f"平均输出长度: {avg_length:.0f} tokens")
        self.logger.info(f"平均峰值显存: {avg_memory:.0f} MB")
        
        if self.args.enable_parallel:
            avg_skel = sum(r['skeleton_time'] for r in self.results) / n
            avg_para = sum(r['parallel_time'] for r in self.results) / n
            self.logger.info(f"平均Skeleton时间: {avg_skel:.2f}s")
            self.logger.info(f"平均Parallel时间: {avg_para:.2f}s")
        
        self.logger.info("=" * 50)

    def _cleanup(self):
        """清理资源"""
        if self.args.distributed and self.distributed_config:
            self.logger.debug("清理分布式资源...")
            # orchestrator 不需要显式清理
