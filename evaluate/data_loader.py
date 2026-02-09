# coding=utf-8
"""
数据集加载器

根据任务类型加载对应的评估数据集
支持的数据集: mt_bench, vicuna_bench, planning
"""

import os
import random
from typing import List, Dict, Any, Optional
import pandas as pd


def load_task_dataset(
    task: str,
    num_samples: int,
    project_dir: str,
) -> List[Dict[str, Any]]:
    """
    加载评估数据集
    
    Args:
        task: 任务类型 (mt_bench/vicuna_bench/planning/retrieval/multi-doc-qa/bfcl)
        num_samples: 采样数量
        project_dir: 项目根目录
        
    Returns:
        List[Dict]: 数据集列表，每个元素包含:
            - 'task_prompt': 任务输入
            - 'task_id' 或 'question_id': 任务标识符
            - 'raw_data': 原始数据
    """
    dataset_dir = os.path.join(project_dir, "evaluate", "datasets")
    
    if task == "mt_bench":
        return _load_mt_bench_dataset(dataset_dir, num_samples)
    elif task == "vicuna_bench":
        return _load_vicuna_bench_dataset(dataset_dir, num_samples)
    elif task == "planning":
        return _load_planning_dataset(dataset_dir, num_samples)
    elif task == "retrieval":
        return _load_retrieval_dataset(dataset_dir, num_samples)
    elif task == "multi-doc-qa":
        return _load_multidoc_qa_dataset(dataset_dir, num_samples)
    elif task == "bfcl":
        return _load_bfcl_dataset(dataset_dir, num_samples)
    else:
        raise ValueError(f"未知任务类型: {task}")


def _load_mt_bench_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """
    加载 MT-Bench 数据集
    
    字段: question_id, category, turns, [reference]
    - turns: 多轮对话列表，只使用第一轮
    - reference: 可选字段，如果存在则将 reference[0] 与 turns[0] 拼接
    """
    import json
    
    path = os.path.join(dataset_dir, "mt_bench", "question.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(results) >= num_samples:
                break
            
            try:
                row = json.loads(line)
                question_id = row.get("question_id", idx)
                category = row.get("category", "")
                turns = row.get("turns", [])
                reference = row.get("reference", None)
                
                # 只使用第一轮对话
                first_turn = turns[0] if turns else ""
                
                # 如果有 reference，拼接 reference[0] 到 prompt
                if reference and len(reference) > 0:
                    task_prompt = f"{first_turn}\n\nReference: {reference[0]}"
                else:
                    task_prompt = first_turn
                
                results.append({
                    "question_id": question_id,
                    "category": category,
                    "task_prompt": task_prompt,
                    "raw_data": row,
                })
            except json.JSONDecodeError as e:
                print(f"警告: MT-Bench 第 {idx+1} 行 JSON 格式错误，跳过: {e}")
                continue
    
    return results


def _load_vicuna_bench_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """
    加载 Vicuna-Bench 数据集
    
    字段: question_id, category, turns
    - turns: 多轮对话列表，只使用第一轮
    """
    import json
    
    path = os.path.join(dataset_dir, "vicuna_bench", "question.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(results) >= num_samples:
                break
            
            try:
                row = json.loads(line)
                question_id = row.get("question_id", idx)
                category = row.get("category", "")
                turns = row.get("turns", [])
                reference = row.get("reference", None)
                
                # 只使用第一轮对话
                first_turn = turns[0] if turns else ""
                
                # 如果有 reference，拼接 reference[0] 到 prompt
                if reference and len(reference) > 0:
                    task_prompt = f"{first_turn}\n\nReference: {reference[0]}"
                else:
                    task_prompt = first_turn
                
                results.append({
                    "question_id": question_id,
                    "category": category,
                    "task_prompt": task_prompt,
                    "raw_data": row,
                })
            except json.JSONDecodeError as e:
                print(f"警告: Vicuna-Bench 第 {idx+1} 行 JSON 格式错误，跳过: {e}")
                continue
    
    return results


def _load_planning_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """
    加载 Planning 任务数据集
    
    字段: task_id, task, answer
    - task: 任务的 prompt
    """
    import json
    
    path = os.path.join(dataset_dir, "planning", "industry_tasks_converted.jsonl")
    if not os.path.exists(path):
        print(f"警告: 数据集文件不存在: {path}")
        return []
    
    # 手动逐行读取 JSONL，处理格式错误
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if len(results) >= num_samples:
                break
            
            try:
                row = json.loads(line)
                results.append({
                    "task_id": row.get("task_id", idx),
                    "task_prompt": row.get("task", ""),
                    "raw_data": row,
                })
            except json.JSONDecodeError as e:
                print(f"警告: 第 {idx+1} 行 JSON 格式错误，跳过: {e}")
                continue
    
    return results


def _load_retrieval_dataset(dataset_dir: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 Retrieval 任务数据集"""
    # TODO: 实现具体的数据集加载逻辑
    
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
