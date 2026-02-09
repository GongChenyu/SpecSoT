# coding=utf-8
"""
结果保存器

统一的结果保存格式
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


def save_results(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: str,
    mode: str = "evaluation",
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    保存推理结果
    
    Args:
        results: 推理结果列表
        task: 任务类型
        output_dir: 输出目录
        mode: 运行模式 (inference/evaluation)
        extra_info: 额外信息（如模型配置等）
        
    Returns:
        str: 保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{task}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 构建保存数据
    save_data = {
        "metadata": {
            "task": task,
            "mode": mode,
            "timestamp": timestamp,
            "num_samples": len(results),
            **(extra_info or {}),
        },
        "results": _clean_results(results),
    }
    
    # 写入文件
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存: {filepath}")
    return filepath


def _clean_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清理结果中不可序列化的字段
    
    Args:
        results: 原始结果列表
        
    Returns:
        List[Dict]: 清理后的结果
    """
    cleaned = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            # 跳过 tensor 类型
            if k == "output_ids":
                continue
            # 其他类型尝试序列化
            try:
                json.dumps(v)
                clean_r[k] = v
            except (TypeError, ValueError):
                clean_r[k] = str(v)
        cleaned.append(clean_r)
    return cleaned


def save_summary(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: str,
) -> str:
    """
    保存汇总统计
    
    Args:
        results: 推理结果列表
        task: 任务类型
        output_dir: 输出目录
        
    Returns:
        str: 保存的文件路径
    """
    if not results:
        return ""
    
    n = len(results)
    
    summary = {
        "task": task,
        "num_samples": n,
        "avg_inference_time": sum(r.get("inference_time", 0) for r in results) / n,
        "avg_throughput": sum(r.get("throughput", 0) for r in results) / n,
        "avg_length": sum(r.get("length", 0) for r in results) / n,
        "avg_memory": sum(r.get("memory", 0) for r in results) / n,
    }
    
    # 如果有骨架并行统计
    if any("skeleton_time" in r for r in results):
        summary["avg_skeleton_time"] = sum(r.get("skeleton_time", 0) for r in results) / n
        summary["avg_parallel_time"] = sum(r.get("parallel_time", 0) for r in results) / n
        summary["avg_num_branches"] = sum(r.get("num_para", 0) for r in results) / n
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"summary_{task}_{timestamp}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return filepath
