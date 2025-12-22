# run_parallel.py
# coding=utf-8
import random
import numpy as np
import torch
import os
import argparse
import pandas as pd
import time
from tqdm import tqdm
import pynvml
from threading import Thread, Event
from eagle.model.ea_model import EaModel
from prompts import *

# ================= 显存监控工具 (复用自 generate.py) =================
class GPUMemoryMonitor:
	def __init__(self, device_index=0, interval=0.1):
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


skeleton_prompt_zh = (
	"解决以上任务共需要两个步骤：\n"
	"步骤一目标：简短的骨架生成。任务指令：请对上述问题进行分析，判断是否可以拆分为多个独立的平行分支（Parallel Branches）进行处理。\n"
	"1. 判断逻辑：如果任务必须顺序执行，请直接回答，无需拆分与生成骨架；如果任务可以拆分（如不同文档、不同方面、不同步骤等），请务必简洁的输出骨架。\n"
	"2. 输出格式：当进行拆分时，必须严格遵守以下 Skeleton-of-Thought 格式：\n"
	" - 每个分支以 '####' 开头，紧接分支标题，以英文冒号 ':' 结尾。\n"
	" - 冒号后使用省略号，绝对不要在此阶段生成具体内容。\n"
	" - 所有分支列举完毕后，必须输出 '####%%%%' 作为结束标记。\n"
	"3. 示例：\n"
	" ####分支一标题:......\n"
	" ####分支二标题:......\n"
	" ####%%%%\n"
	"4. 注意：现在只执行步骤一（生成骨架）。请直接输出回答或者直接输出骨架，禁止包含任何开场白或解释。\n"
	"开始回答（直接回答或者骨架）：\n"
)


# ================= 主函数 =================
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B")
	parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3")
	parser.add_argument("--task", type=str, default="planning", choices=["retrieval", "planning", "multi-doc-qa"])
	parser.add_argument("--device_index", type=int, default=7)
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    
	print(f"Loading Base Model from: {args.base_model_path}")
	print(f"Loading EAGLE Model from: {args.eagle_model_path}")

	# 初始化 EAGLE 模型
	model = EaModel.from_pretrained(
		base_model_path=args.base_model_path,
		ea_model_path=args.eagle_model_path,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		device_map="auto"
	)
    
	tokenizer = model.get_tokenizer()
	tokenizer.padding_side = "left"

	# 准备特殊 Token ID (用于语义并行)
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
	print("Semantic Parallel Special Token IDs:", para_token_ids)

	# 数据集加载逻辑 (保持原样)
	project_dir = os.path.dirname(os.path.abspath(__file__))
	if args.task == "retrieval":
		df_path = os.path.join(project_dir, "datasets/student_resume_logic_retrieval", "logic_gpa_resume_10.jsonl")
		df = pd.read_json(df_path, lines=True)
		addition_prompt = "You should check every student to judge whether he meets the requirement in your reasoning process."
		df['full_prompt'] = df['prompt'].apply(lambda x: x + "\n" + addition_prompt + "\n" + skeleton_prompt_zh)
	elif args.task == "planning":
		df_path = os.path.join(project_dir, "datasets/planning", "industry_tasks.jsonl")
		df = pd.read_json(df_path, lines=True)
		df['full_prompt'] = df['task'].apply(lambda x: "任务：" + x + "\n" + skeleton_prompt_zh)
	else: # multi-doc-qa
		df_path = os.path.join(project_dir, "datasets/multi-doc-qa", "2wikimqa.jsonl")
		df = pd.read_json(df_path, lines=True)
		df = df[df["context"].apply(lambda x: len(tokenizer.encode(x))) < 7500].reset_index(drop=True)
		addition_prompt = "You should check each document one by one..."
		df['full_prompt'] = df.apply(lambda x: x["context"] + "\n\nQuestion: " + x['input'] + "\n" + addition_prompt + "\n" + skeleton_prompt_zh, axis=1)

	# 测试前 5 条
	df = df[0:1]
    
	results = []

	for i in tqdm(range(len(df))):
		full_prompt = df.loc[i, "full_prompt"]
		# full_prompt = "你叫什么名字，给我介绍华为。"
		# full_prompt = "你好"
		print(f"prompt:{full_prompt}")
		input_ids = tokenizer([full_prompt], return_tensors="pt").input_ids.to(model.base_model.device)

		with GPUMemoryMonitor(device_index=args.device_index) as monitor:
			start_time = time.time()
            
			# === 调用核心方法: eagenerate (支持语义并行) ===
			output_ids = model.eagenerate(
				input_ids,
				max_new_tokens=5000,
				temperature=0.0,
				enable_parallel=True,   # 开启并行开关
				**para_token_ids        # 传入特殊 Token
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
		print(results)

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


