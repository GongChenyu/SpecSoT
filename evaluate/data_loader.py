# coding=utf-8
"""
数据集加载器

根据任务类型加载对应的评估数据集
"""

import os
import random
from typing import List, Dict, Any, Optional


def load_task_dataset(
    task: str,
    num_samples: int,
    seed: int,
    project_dir: str,
) -> List[Dict[str, Any]]:
    """
    加载评估数据集
    
    Args:
        task: 任务类型 (planning/retrieval/multi-doc-qa/bfcl)
        num_samples: 采样数量
        seed: 随机种子
        project_dir: 项目根目录
        
    Returns:
        List[Dict]: 数据集列表，每个元素包含 'task_prompt' 键
    """
    random.seed(seed)
    
    dataset_dir = os.path.join(project_dir, "evaluate", "datasets")
    
    if task == "planning":
        return _load_planning_dataset(dataset_dir, num_samples)
    elif task == "retrieval":
        return _load_retrieval_dataset(dataset_dir, num_samples)
    elif task == "multi-doc-qa":
        return _load_multidoc_qa_dataset(dataset_dir, num_samples)
    elif task == "bfcl":
        return _load_bfcl_dataset(dataset_dir, num_samples)
    else:
        raise ValueError(f"未知任务类型: {task}")


def _load_planning_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 Planning 任务数据集"""
    # TODO: 实现具体的数据集加载逻辑
    # 数据集路径: datasets/planning/industry_tasks_no_ascii.jsonl
    import pandas as pd
    
    path = os.path.join(dataset_dir, "planning", "industry_tasks_no_ascii.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    df = pd.read_json(path, lines=True)
    df = df[:num_samples]
    
    return [
        {"task_prompt": row.get("task", ""), "raw_data": row.to_dict()}
        for _, row in df.iterrows()
    ]


def _load_retrieval_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 Retrieval 任务数据集"""
    # TODO: 实现具体的数据集加载逻辑
    import pandas as pd
    
    path = os.path.join(dataset_dir, "student_resume_logic_retrieval", "logic_gpa_resume_10.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    df = pd.read_json(path, lines=True)
    df = df[:num_samples]
    
    return [
        {"task_prompt": row.get("prompt", ""), "raw_data": row.to_dict()}
        for _, row in df.iterrows()
    ]


def _load_multidoc_qa_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 Multi-Doc QA 任务数据集"""
    # TODO: 实现具体的数据集加载逻辑
    import pandas as pd
    
    path = os.path.join(dataset_dir, "multi-doc-qa", "2wikimqa.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    df = pd.read_json(path, lines=True)
    df = df[:num_samples]
    
    return [
        {"task_prompt": row.get("input", ""), "raw_data": row.to_dict()}
        for _, row in df.iterrows()
    ]


def _load_bfcl_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 BFCL (Berkeley Function Calling Leaderboard) 数据集"""
    # TODO: 实现具体的数据集加载逻辑
    # 数据集路径: datasets/bfcl/
    import pandas as pd
    
    path = os.path.join(dataset_dir, "bfcl", "BFCL_v4_parallel_multiple.json")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    # BFCL 使用 JSON Lines 格式
    df = pd.read_json(path, lines=True)
    df = df[:num_samples]
    
    results = []
    for _, row in df.iterrows():
        # 提取 question 中的用户 prompt
        question = row.get("question", [[]])[0]
        if question and len(question) > 0:
            user_content = question[0].get("content", "") if isinstance(question[0], dict) else ""
        else:
            user_content = ""
        
        results.append({
            "task_prompt": user_content,
            "functions": row.get("function", []),
            "raw_data": row.to_dict(),
        })
    
    return results
