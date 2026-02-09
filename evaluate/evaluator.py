# coding=utf-8
"""
评估器

计算各任务的评估指标
"""

from typing import List, Dict, Any


def evaluate_results(
    results: List[Dict[str, Any]],
    task: str,
) -> Dict[str, float]:
    """
    评估推理结果
    
    Args:
        results: 推理结果列表
        task: 任务类型
        
    Returns:
        Dict[str, float]: 评估指标
    """
    if task == "planning":
        return _evaluate_planning(results)
    elif task == "retrieval":
        return _evaluate_retrieval(results)
    elif task == "multi-doc-qa":
        return _evaluate_multidoc_qa(results)
    elif task == "bfcl":
        return _evaluate_bfcl(results)
    else:
        return _evaluate_default(results)


def _evaluate_planning(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """评估 Planning 任务"""
    # TODO: 实现 Planning 任务评估逻辑
    # 可能的指标: 步骤完整性、逻辑一致性等
    return _compute_throughput_metrics(results)


def _evaluate_retrieval(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """评估 Retrieval 任务"""
    # TODO: 实现 Retrieval 任务评估逻辑
    # 可能的指标: 准确率、召回率等
    return _compute_throughput_metrics(results)


def _evaluate_multidoc_qa(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """评估 Multi-Doc QA 任务"""
    # TODO: 实现 Multi-Doc QA 评估逻辑
    # 可能的指标: F1, EM (Exact Match) 等
    return _compute_throughput_metrics(results)


def _evaluate_bfcl(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """评估 BFCL 任务"""
    # TODO: 实现 BFCL 评估逻辑
    # 可能的指标: 函数调用准确率、参数正确率等
    return _compute_throughput_metrics(results)


def _evaluate_default(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """默认评估（仅计算吞吐量指标）"""
    return _compute_throughput_metrics(results)


def _compute_throughput_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算吞吐量相关指标
    
    Args:
        results: 推理结果列表
        
    Returns:
        Dict[str, float]: 吞吐量指标
    """
    if not results:
        return {}
    
    n = len(results)
    
    metrics = {
        "num_samples": n,
        "avg_inference_time": sum(r.get("inference_time", 0) for r in results) / n,
        "avg_throughput": sum(r.get("throughput", 0) for r in results) / n,
        "avg_output_length": sum(r.get("length", 0) for r in results) / n,
        "avg_peak_memory_mb": sum(r.get("memory", 0) for r in results) / n,
    }
    
    # 骨架并行相关指标
    if any("skeleton_time" in r for r in results):
        metrics["avg_skeleton_time"] = sum(r.get("skeleton_time", 0) for r in results) / n
        metrics["avg_parallel_time"] = sum(r.get("parallel_time", 0) for r in results) / n
        metrics["avg_num_branches"] = sum(r.get("num_para", 0) for r in results) / n
        metrics["avg_accept_length"] = sum(r.get("avg_accept_len", 0) for r in results) / n
    
    return metrics


def print_evaluation_summary(metrics: Dict[str, float], task: str):
    """
    打印评估汇总
    
    Args:
        metrics: 评估指标
        task: 任务类型
    """
    print("\n" + "=" * 60)
    print(f"评估汇总 - {task}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)
