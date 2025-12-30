# run_parallel.py
# coding=utf-8
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device_index = 7  # 修改为所需的 GPU 设备索引
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

import gc
import random
import numpy as np
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import pynvml
from threading import Thread, Event
from eagle.model.ea_model import EaModel
from eagle.prompts import *


# ================= 显存监控工具 (复用自 generate.py) =================
class GPUMemoryMonitor:
	def __init__(self, device_index=0, interval=0.01):
		self.device_index = device_index
		self.interval = interval
		self.peak_usage = 0
		self.monitor_thread = None
		self.stop_event = Event()

	def _monitor(self):
		pynvml.nvmlInit()
		handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
		while not self.stop_event.is_set():
			info = pynvml.nvmlDeviceGetMemoryInfo(handle)
			current_usage = info.used / (1024 ** 2)
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


def load_dateset(task):
	# 数据集加载逻辑 (保持原样)
	project_dir = os.path.dirname(os.path.abspath(__file__))
	if task == "retrieval":
		df_path = os.path.join(project_dir, "datasets/student_resume_logic_retrieval", "logic_gpa_resume_10.jsonl")
		df = pd.read_json(df_path, lines=True)
		df['task_prompt'] = df['prompt']
	elif task == "planning": 
		df_path = os.path.join(project_dir, "datasets/planning", "industry_tasks_no_ascii.jsonl")
		df = pd.read_json(df_path, lines=True)
		df['task_prompt'] = df['task']
	else: # multi-doc-qa
		df_path = os.path.join(project_dir, "datasets/multi-doc-qa", "2wikimqa.jsonl")
		df = pd.read_json(df_path, lines=True)
		df['task_prompt'] = df['input']
	return df


def get_special_token_ids(tokenizer):
	para_token_ids = {
		"para_begin_token_id": tokenizer.encode("####")[0],
		"para_end_token_id": tokenizer.encode("%%%%")[0],
		"ellipsis_token_id": tokenizer.encode("......")[0],
		"half_ellipsis_token_id": tokenizer.encode("...")[0],
		"line_break_token_id": tokenizer.encode("\n\n")[0],
		"colon_token_id": tokenizer.encode(":")[0],
		"cn_colon_token_id": tokenizer.encode("：")[0], # cn ： 
		"colon_new_line_token_id": tokenizer.encode(":\n")[0]
	}
	return para_token_ids


# ================= 数据保存逻辑 (已移除绘图部分) =================
def save_results(raw_results_df, output_dir):
    print("\nStarting Data Processing (No Plotting)...")
    
    # 1. 保存原始有效数据 (包含 response 文本)
    raw_path = os.path.join(output_dir, "benchmark_valid_raw_data.csv")
    # 使用 escapechar 处理文本中可能存在的复杂字符
    raw_results_df.to_csv(raw_path, index=False, escapechar='\\')
    print(f"Raw data saved to {raw_path}")

    # 2. 数据聚合
    # 注意：聚合时会自动忽略非数值列 (response 文本列会被自动忽略)
    grouped_df = raw_results_df.groupby(['total_token_setting', 'num_para']).agg({
        'avg_accept_len': 'mean',
        'throughput': 'mean',
        'time': 'mean',
        'length': 'mean',
        'memory': 'max',
        'sample_id': 'count'
    }).reset_index()
    
    grouped_df = grouped_df.rename(columns={'sample_id': 'valid_sample_count'})
    grouped_df = grouped_df.sort_values(by=['total_token_setting', 'num_para'])

    agg_path = os.path.join(output_dir, "benchmark_aggregated_table.csv")
    grouped_df.to_csv(agg_path, index=False)
    print(f"Aggregated table saved to {agg_path}")
    print("Data processing complete.")


# ================= 推理执行函数 (已修改：增加 tokenizer 和 response 返回) =================
def run_inference(model, tokenizer, task_prompt, para_token_ids, is_warmup=False):
    """
    执行单次推理。
    修改：增加 tokenizer 参数，解码并返回 response 文本。
    """
    try:
        with GPUMemoryMonitor(device_index=device_index) as monitor:
            start_time = time.time()
            
            output_ids, avg_accept_len, num_para = model.eagenerate(
                task_prompt,
                max_new_tokens=3000,
                temperature=0.0,
                enable_parallel=True,
                para_token_ids=para_token_ids
            )
            
            total_time = time.time() - start_time
            gen_len = len(output_ids[0])
            throughput = gen_len / total_time if total_time > 0 else 0
            
            if is_warmup:
                return None
            
            # === 解码并保存 Response ===
            response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return {
                "prompt": task_prompt,
                "num_para": int(num_para),
                "avg_accept_len": avg_accept_len,
                "time": total_time,
                "length": gen_len,
                "throughput": throughput,
                "memory": monitor.peak_usage,
                "response": response_text  # <--- 新增字段
            }
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B")
    parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3")
    parser.add_argument("--task", type=str, default="planning", choices=["retrieval", "planning", "multi-doc-qa"])
    parser.add_argument("--token_configs", type=str, default="10,20,30,40,50", help="Comma separated list of total_tokens")
    parser.add_argument("--test_rounds", type=int, default=2, help="Number of test rounds per sample")
    args = parser.parse_args()

    token_configs = [int(x) for x in args.token_configs.split(",")]
    print(f"Benchmark Configs: Total Tokens={token_configs}, Test Rounds={args.test_rounds}")

    full_df = load_dateset(args.task)
    
    # 切分预热和正式数据
    if len(full_df) > 3:
        warmup_df = full_df.iloc[:2].reset_index(drop=True)
        test_df = full_df.iloc[2:].reset_index(drop=True)
    else:
        print("Warning: Dataset too small (<3). Using 1st sample for warmup.")
        warmup_df = full_df.iloc[:1]
        test_df = full_df 

    print(f"Dataset Split: {len(warmup_df)} Warmup samples, {len(test_df)} Test samples.")

    all_results = []
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    total_attempts = 0
    valid_attempts = 0

    for tt in token_configs:
        print(f"\n{'='*20} Config: Total Token = {tt} {'='*20}")
        
        torch.cuda.empty_cache()
        gc.collect()

        try:
            model = EaModel.from_pretrained(
                base_model_path=args.base_model_path,
                ea_model_path=args.eagle_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                total_token=tt,
                depth=4, 
                top_k=15
            )
            model.eval()
            tokenizer = model.get_tokenizer()
            tokenizer.padding_side = "left"
            para_token_ids = get_special_token_ids(tokenizer)
        except Exception as e:
            print(f"Init Error: {e}")
            continue

        # === Phase 1: Warmup ===
        print(f"  [Warmup] Running {len(warmup_df)} samples...")
        for i in range(len(warmup_df)):
            task_prompt = warmup_df.loc[i, "task_prompt"]
            # 这里的调用也加了 tokenizer，虽然 warmup 不返回结果
            run_inference(model, tokenizer, task_prompt, para_token_ids, is_warmup=True)

        # === Phase 2: Testing ===
        print(f"  [Testing] Running {len(test_df)} samples x {args.test_rounds} rounds...")
        
        for i in tqdm(range(len(test_df)), desc=f"TT={tt}"):
            task_prompt = test_df.loc[i, "task_prompt"]
            print("===========================")
            print(f"\n[Sample {i}] Prompt: {task_prompt[:50]}...")  
            
            # 获取预期分支数
            expected_branches = int(test_df.loc[i, "num_branches"]) if "num_branches" in test_df.columns else -1
            
            for r in range(args.test_rounds):
                # 传入 tokenizer
                res = run_inference(model, tokenizer, task_prompt, para_token_ids, is_warmup=False)
                
                if res:
                    total_attempts += 1
                    actual_para = res["num_para"]
                    
                    # === 核心校验逻辑 ===
                    if expected_branches != -1 and actual_para != expected_branches:
                        continue
                    
                    valid_attempts += 1
                    res["total_token_setting"] = tt
                    res["sample_id"] = i
                    res["round_id"] = r
                    res["expected_branches"] = expected_branches
                    all_results.append(res)

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # === 统计摘要 ===
    print("\n" + "="*30)
    print("BENCHMARK SUMMARY")
    print("="*30)
    print(f"Total Attempts: {total_attempts}")
    print(f"Valid Results : {valid_attempts}")
    if total_attempts > 0:
        print(f"Success Rate  : {(valid_attempts/total_attempts)*100:.2f}%")
    else:
        print("Success Rate  : N/A")
    print("="*30)

    if not all_results:
        print("No valid results collected (maybe all failed the branch check?).")
        return

    df_results = pd.DataFrame(all_results)
    # 调用新的保存函数，而不是原来的绘图函数
    save_results(df_results, output_dir)
    print("\nBenchmark Completed Successfully!")

if __name__ == "__main__":
    seed = 35
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    main()

