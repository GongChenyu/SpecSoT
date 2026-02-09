# coding=utf-8
"""
结果保存器

统一的结果保存格式，使用 JSONL 格式
每个数据集对应一个文件夹，与 datasets 目录结构对应
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_result_filename(
    enable_parallel: bool = False,
    distributed: bool = False,
    use_scheduling: bool = False,
    use_eagle3: bool = True,
) -> str:
    """
    根据运行模式生成结果文件名
    
    命名规则:
    - eagle_ 或 specsot_: 基于是否启用并行
    - _distributed: 如果启用分布式
    - _scheduling: 如果启用调度
    
    Args:
        enable_parallel: 是否启用骨架并行 (True: specsot, False: eagle)
        distributed: 是否启用分布式
        use_scheduling: 是否启用调度
        use_eagle3: 是否使用 Eagle3 模型
        
    Returns:
        str: 文件名 (不含路径和扩展名)
    """
    # 基础前缀
    if enable_parallel:
        prefix = "specsot"
    else:
        prefix = "eagle3" if use_eagle3 else "eagle2"
    
    # 添加模式后缀
    parts = [prefix]
    if distributed:
        parts.append("distributed")
    if use_scheduling:
        parts.append("scheduling")
    
    return "_".join(parts)


def save_results(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: str,
    enable_parallel: bool = False,
    distributed: bool = False,
    use_scheduling: bool = False,
    use_eagle3: bool = True,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    保存推理结果到 JSONL 格式文件
    
    文件结构: {output_dir}/{task}/{filename}.jsonl
    每行一个结果，包含 question_id/task_id, response, 以及性能指标
    
    Args:
        results: 推理结果列表
        task: 任务类型 (mt_bench/vicuna_bench/planning 等)
        output_dir: 输出根目录
        enable_parallel: 是否启用骨架并行
        distributed: 是否启用分布式
        use_scheduling: 是否启用调度
        use_eagle3: 是否使用 Eagle3 模型
        extra_info: 额外信息
        
    Returns:
        str: 保存的文件路径
    """
    # 创建任务对应的子目录
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    
    # 生成文件名
    base_name = get_result_filename(
        enable_parallel=enable_parallel,
        distributed=distributed,
        use_scheduling=use_scheduling,
        use_eagle3=use_eagle3,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.jsonl"
    filepath = os.path.join(task_dir, filename)
    
    # 清理并保存结果
    cleaned_results = _clean_results(results, task)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for result in cleaned_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"结果已保存: {filepath}")
    
    # 同时保存元数据到单独的 JSON 文件
    meta_filepath = filepath.replace(".jsonl", "_meta.json")
    metadata = {
        "task": task,
        "timestamp": timestamp,
        "num_samples": len(results),
        "enable_parallel": enable_parallel,
        "distributed": distributed,
        "use_scheduling": use_scheduling,
        "use_eagle3": use_eagle3,
        **(extra_info or {}),
    }
    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return filepath


def _clean_results(results: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """
    清理结果，提取关键字段
    
    每条结果包含:
    - question_id 或 task_id (根据数据集类型)
    - response: 模型输出
    - inference_time: 总推理时间
    - skeleton_time: 骨架生成时间 (如果启用并行)
    - parallel_time: 并行生成时间 (如果启用并行)
    - memory: 峰值显存 (MB)
    - num_para: 并行分支数
    - length: 输出 token 数
    - throughput: 吞吐量 (tokens/s)
    
    Args:
        results: 原始结果列表
        task: 任务类型
        
    Returns:
        List[Dict]: 清理后的结果
    """
    cleaned = []
    for r in results:
        clean_r = {}
        
        # 提取任务 ID (根据数据集类型)
        raw_data = r.get("raw_data", {})
        if task in ["mt_bench", "vicuna_bench"]:
            clean_r["question_id"] = raw_data.get("question_id", r.get("question_id", 0))
            clean_r["category"] = raw_data.get("category", r.get("category", ""))
        elif task == "planning":
            clean_r["task_id"] = raw_data.get("task_id", r.get("task_id", 0))
        else:
            # 其他数据集尝试提取通用 ID
            if "question_id" in raw_data:
                clean_r["question_id"] = raw_data["question_id"]
            elif "task_id" in raw_data:
                clean_r["task_id"] = raw_data["task_id"]
        
        # 提取核心字段
        clean_r["response"] = r.get("response", "")
        clean_r["prompt"] = r.get("prompt", "")
        
        # 性能指标
        clean_r["inference_time"] = r.get("inference_time", 0)
        clean_r["skeleton_time"] = r.get("skeleton_time", 0)
        clean_r["parallel_time"] = r.get("parallel_time", 0)
        clean_r["memory"] = r.get("memory", 0)
        clean_r["num_para"] = r.get("num_para", 0)
        clean_r["length"] = r.get("length", 0)
        clean_r["throughput"] = r.get("throughput", 0)
        clean_r["avg_accept_len"] = r.get("avg_accept_len", 0)
        clean_r["mode"] = r.get("mode", "unknown")
        
        cleaned.append(clean_r)
    return cleaned


def save_summary(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: str,
    enable_parallel: bool = False,
    distributed: bool = False,
    use_scheduling: bool = False,
    use_eagle3: bool = True,
) -> str:
    """
    保存汇总统计
    
    Args:
        results: 推理结果列表
        task: 任务类型
        output_dir: 输出目录
        enable_parallel: 是否启用骨架并行
        distributed: 是否启用分布式
        use_scheduling: 是否启用调度
        use_eagle3: 是否使用 Eagle3 模型
        
    Returns:
        str: 保存的文件路径
    """
    if not results:
        return ""
    
    n = len(results)
    
    summary = {
        "task": task,
        "num_samples": n,
        "enable_parallel": enable_parallel,
        "distributed": distributed,
        "use_scheduling": use_scheduling,
        "avg_inference_time": sum(r.get("inference_time", 0) for r in results) / n,
        "avg_throughput": sum(r.get("throughput", 0) for r in results) / n,
        "avg_length": sum(r.get("length", 0) for r in results) / n,
        "avg_memory": sum(r.get("memory", 0) for r in results) / n,
    }
    
    # 如果有骨架并行统计
    if enable_parallel:
        summary["avg_skeleton_time"] = sum(r.get("skeleton_time", 0) for r in results) / n
        summary["avg_parallel_time"] = sum(r.get("parallel_time", 0) for r in results) / n
        summary["avg_num_branches"] = sum(r.get("num_para", 0) for r in results) / n
        summary["avg_accept_len"] = sum(r.get("avg_accept_len", 0) for r in results) / n
    
    # 保存到任务对应的子目录
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    
    base_name = get_result_filename(
        enable_parallel=enable_parallel,
        distributed=distributed,
        use_scheduling=use_scheduling,
        use_eagle3=use_eagle3,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(task_dir, f"{base_name}_summary_{timestamp}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return filepath
