
# coding=utf-8
import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
import time
import threading
from threading import Thread, Event


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
        self.monitor_thread = Thread(target=self._monitor)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.monitor_thread.join()
        return False
    
def main():
    # 配置模型路径
    base_model_path = "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    # 加载数据
    data_dir = "/data/home/chenyu/Coding/SD+SoT/parallel-decoding-in-one-sequence"
    df_path = os.path.join(data_dir, "datasets/planning", "industry_tasks.jsonl")
    df = pd.read_json(df_path, lines=True)
    df['full_prompt'] = df['task'].apply(lambda x: "任务：" + x)
    df = df[0:1]  # 只取一条测试


    total_time = 0.0
    total_tokens = 0
    peak_memory = 0.0
    start_time = time.time()
    device_index = 0  # 默认用0号卡
    with GPUMemoryMonitor(device_index=device_index) as monitor:
        for i in range(len(df)):
            full_prompt = df.loc[i, "full_prompt"]
            print(f"Prompt: {full_prompt}")
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.0,
                    do_sample=False
                )
            gen_time = time.time() - gen_start
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Response: {response}")
            gen_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
            total_tokens += gen_tokens
            total_time += gen_time
    total_elapsed = time.time() - start_time
    peak_memory = monitor.peak_usage
    throughput = total_tokens / total_time if total_time > 0 else 0
    print("\n===== 测试指标 =====")
    print(f"总推理时间: {total_elapsed:.2f} 秒")
    print(f"生成总token数: {total_tokens}")
    print(f"总显存占用: {peak_memory:.2f} MB")
    print(f"总吞吐 (token/s): {throughput:.2f}")

if __name__ == "__main__":
    main()



