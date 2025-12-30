# run_parallel.py
# coding=utf-8
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


# ================= 主函数 =================
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B")
	parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3")
	parser.add_argument("--task", type=str, default="planning", choices=["retrieval", "planning", "multi-doc-qa"])
	# parser.add_argument("--device_index", type=int, default=7)
	args = parser.parse_args()

	print(f"Loading Base Model from: {args.base_model_path}")
	print(f"Loading EAGLE Model from: {args.eagle_model_path}")

	# 初始化 EAGLE 模型
	model = EaModel.from_pretrained(
		base_model_path=args.base_model_path,
		ea_model_path=args.eagle_model_path,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		device_map="auto",
		total_token=60, 
		depth=7, 
		top_k=10
	)
    
	tokenizer = model.get_tokenizer()
	tokenizer.padding_side = "left"

	# 准备特殊 Token ID (用于语义并行)
	para_token_ids = get_special_token_ids(tokenizer)
	print("Semantic Parallel Special Token IDs:", para_token_ids)

	df = load_dateset(args.task)
	# 测试前 5 条
	df = df[0:1]
    
	results = []

	for i in tqdm(range(len(df))):
		task_prompt = df.loc[i, "task_prompt"]
		# task_prompt = "请分析运动的好处"
		# task_prompt = "请从两个方面分析运动的好处"
		# task_prompt = "请从三个方面分析运动的好处"
		# task_prompt = "请从四个方面分析运动的好处"
		# task_prompt = "请从五个方面分析篮球的好处"
		# task_prompt = "请从六个方面分析篮球的好处"
		# task_prompt = "请从七个方面分析篮球的好处"
		# task_prompt = "请从八个方面分析篮球的好处"
		# task_prompt = "请从九个方面分析篮球的好处"

		print(f"prompt:{task_prompt}")
		# input_ids = tokenizer([task_prompt], return_tensors="pt").input_ids.to(model.base_model.device)

		with GPUMemoryMonitor(device_index=device_index) as monitor:
			start_time = time.time()
            
			# === 调用核心方法: eagenerate (支持语义并行) ===
			output_ids, avg_accept_len, num_para = model.eagenerate(
				task_prompt,
				max_new_tokens=3000,
				temperature=0.0,
				enable_parallel=True,   # 开启并行开关
				para_token_ids=para_token_ids        # 传入特殊 Token
			)
            
			total_time = time.time() - start_time
            
		response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
		print(f"\n[Sample {i}] Time: {total_time:.2f}s, Memory: {monitor.peak_usage:.2f}MB")
		# print("Response Preview:", response[:200])
        
		results.append({
			"response": response,
			"time": total_time,
			"memory": monitor.peak_usage,
			"length": len(output_ids[0])
		})
		print(f"time: {total_time}, memory: {monitor.peak_usage}, length: {len(output_ids[0])}")

	# # 保存结果
	# save_dir = os.path.join(project_dir, "results")
	# save_path = os.path.join(save_dir, f"results_eagle_parallel_{args.task}.json")
	# pd.DataFrame(results).to_json(save_path, orient='records', indent=4, force_ascii=False)
	# print(f"Results saved to {save_path}")

if __name__ == "__main__":
	seed = 35
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	main()


