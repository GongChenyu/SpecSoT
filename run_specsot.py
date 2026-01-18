# coding=utf-8
"""
SpecSoT: Speculative Decoding + Skeleton-of-Thought 运行脚本

该脚本演示如何使用 SpecSoT 模型进行推理：
1. 加载 SpecSoT 模型（Base Model + Eagle Layer）
2. 准备特殊 token IDs（用于骨架解析）
3. 执行推理（支持普通模式和骨架并行模式）
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device_index = 7  # 修改为所需的 GPU 设备索引
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

import random
import numpy as np
import torch
import argparse
import pandas as pd
import time
from tqdm import tqdm
import pynvml
from threading import Thread, Event

from SpecSoT import SpecSoTModel


# =============================================================================
# GPU 显存监控工具
# =============================================================================

class GPUMemoryMonitor:
    """
    GPU 显存监控器
    
    使用上下文管理器模式，在推理过程中后台监控显存峰值。
    
    Example:
        >>> with GPUMemoryMonitor(device_index=0) as monitor:
        ...     # 执行推理
        ...     pass
        >>> print(f"Peak memory: {monitor.peak_usage:.2f} MB")
    """
    
    def __init__(self, device_index: int = 0, interval: float = 0.01):
        """
        Args:
            device_index: GPU 设备索引
            interval: 采样间隔（秒）
        """
        self.device_index = device_index
        self.interval = interval
        self.peak_usage = 0
        self.monitor_thread = None
        self.stop_event = Event()

    def _monitor(self):
        """后台监控线程"""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        while not self.stop_event.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_usage = info.used / (1024 ** 2)  # 转换为 MB
            if current_usage > self.peak_usage:
                self.peak_usage = current_usage
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def __enter__(self):
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.monitor_thread.join()
        return False


# =============================================================================
# 数据加载
# =============================================================================

def load_dataset(task: str) -> pd.DataFrame:
    """
    加载评估数据集
    
    Args:
        task: 任务类型 ["retrieval", "planning", "multi-doc-qa"]
        
    Returns:
        DataFrame: 包含 task_prompt 列的数据集
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    if task == "retrieval":
        df_path = os.path.join(
            project_dir, 
            "datasets/student_resume_logic_retrieval", 
            "logic_gpa_resume_10.jsonl"
        )
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['prompt']
        
    elif task == "planning":
        df_path = os.path.join(
            project_dir, 
            "datasets/planning", 
            "industry_tasks_no_ascii.jsonl"
        )
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['task']
        
    else:  # multi-doc-qa
        df_path = os.path.join(
            project_dir, 
            "datasets/multi-doc-qa", 
            "2wikimqa.jsonl"
        )
        df = pd.read_json(df_path, lines=True)
        df['task_prompt'] = df['input']
        
    return df


def get_special_token_ids(tokenizer) -> dict:
    """
    获取骨架解析所需的特殊 token IDs
    
    骨架格式：
    ####标题1:...
    ####标题2:...
    ####%%%%
    
    Args:
        tokenizer: 分词器
        
    Returns:
        dict: 特殊 token ID 映射
    """
    return {
        "para_begin_token_id": tokenizer.encode("####")[0],
        "para_end_token_id": tokenizer.encode("%%%%")[0],
        "ellipsis_token_id": tokenizer.encode("......")[0],
        "half_ellipsis_token_id": tokenizer.encode("...")[0],
        "line_break_token_id": tokenizer.encode("\n\n")[0],
        "colon_token_id": tokenizer.encode(":")[0],
        "cn_colon_token_id": tokenizer.encode("：")[0],  # 中文冒号
        "colon_new_line_token_id": tokenizer.encode(":\n")[0],
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpecSoT Inference Script")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        # default="/data/home/chenyu/Coding/SD+SoT/models/Llama-3.1-8B-Instruct",
        default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B",
        help="Base Model 路径"
    )
    parser.add_argument(
        "--eagle_model_path", 
        type=str, 
        # default="/data/home/chenyu/Coding/SD+SoT/models/EAGLE3-LLaMA3.1-Instruct-8B",
        default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3",
        help="Eagle Model 路径"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="planning",
        choices=["retrieval", "planning", "multi-doc-qa"],
        help="评估任务类型"
    )
    parser.add_argument(
        "--enable_parallel", 
        type=bool, 
        default=True,
        help="是否启用骨架并行模式"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1000,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1,
        help="测试样本数量"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SpecSoT: Speculative Decoding + Skeleton-of-Thought")
    print("=" * 60)
    print(f"Base Model: {args.base_model_path}")
    print(f"Eagle Model: {args.eagle_model_path}")
    print(f"Task: {args.task}")
    print(f"Enable Parallel: {args.enable_parallel}")
    print("=" * 60)

    # =========================================================================
    # 1. 加载 SpecSoT 模型
    # =========================================================================
    print("\n>>> Loading SpecSoT Model...")
    model = SpecSoTModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.eagle_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=40,
        depth=4,
        top_k=6,
    )
    
    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"
    print("Model loaded successfully!")

    # =========================================================================
    # 2. 准备特殊 Token IDs
    # =========================================================================
    para_token_ids = get_special_token_ids(tokenizer)
    print("\nSpecial Token IDs:")
    for key, value in para_token_ids.items():
        token_str = tokenizer.decode([value])
        print(f"  {key}: {value} -> '{token_str}'")

    # =========================================================================
    # 3. 加载数据集
    # =========================================================================
    df = load_dataset(args.task)
    df = df[:args.num_samples]
    print(f"\nLoaded {len(df)} samples for evaluation.")

    # =========================================================================
    # 4. 执行推理
    # =========================================================================
    results = []
    
    for i in tqdm(range(len(df)), desc="Generating"):
        # task_prompt = df.loc[i, "task_prompt"]
        task_prompt = "请问打篮球时，如何提高投篮命中率？请给出详细的建议。"  
        # task_prompt = "How to improve your shooting percentage when playing basketball? Please give detailed suggestions."
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: {task_prompt[:100]}...")
        print("=" * 60)

        with GPUMemoryMonitor(device_index=device_index) as monitor:
            start_time = time.time()
            
            # 调用 SpecSoT 生成
            output_ids, avg_accept_len, num_para, avg_draft_time, avg_update_time, avg_verify_time = \
                model.generate(
                    task_prompt=task_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    enable_parallel=args.enable_parallel,
                    para_token_ids=para_token_ids,
                )
            
            total_time = time.time() - start_time

        # 解码响应
        response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        # 打印统计信息
        print(f"\n[Statistics]")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Peak Memory: {monitor.peak_usage:.2f} MB")
        print(f"  Output Length: {len(output_ids[0])} tokens")
        print(f"  Parallel Branches: {num_para}")
        print(f"  Avg Accept Length: {avg_accept_len:.2f}")
        print(f"  Avg Draft Time: {avg_draft_time*1000:.2f} ms")
        print(f"  Avg Verify Time: {avg_verify_time*1000:.2f} ms")
        print(f"  Avg Update Time: {avg_update_time*1000:.2f} ms")
        
        # 打印响应预览
        print(f"\n[Response Preview]")
        print(response)
        # print(response[:500] + "..." if len(response) > 500 else response)

        results.append({
            "prompt": task_prompt,
            "response": response,
            "time": total_time,
            "memory": monitor.peak_usage,
            "length": len(output_ids[0]),
            "num_para": num_para,
            "avg_accept_len": avg_accept_len,
            "avg_draft_time": avg_draft_time,
            "avg_verify_time": avg_verify_time,
            "avg_update_time": avg_update_time,
        })

    # =========================================================================
    # 5. 保存结果
    # =========================================================================
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(project_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    mode = "parallel" if args.enable_parallel else "standard"
    save_path = os.path.join(save_dir, f"results_specsot_{mode}_{args.task}.json")
    pd.DataFrame(results).to_json(save_path, orient='records', indent=4, force_ascii=False)
    print(f"\nResults saved to: {save_path}")

    # 打印汇总统计
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_length = sum(r['length'] for r in results) / len(results)
    avg_memory = sum(r['memory'] for r in results) / len(results)
    print(f"  Average Time: {avg_time:.2f}s")
    print(f"  Average Length: {avg_length:.1f} tokens")
    print(f"  Average Memory: {avg_memory:.2f} MB")
    print(f"  Throughput: {avg_length/avg_time:.1f} tokens/s")


if __name__ == "__main__":
    # 设置随机种子
    seed = 35
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    main()
