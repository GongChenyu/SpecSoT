import json
import os

def convert_jsonl_to_readable(input_file_path, output_file_path):
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"输入文件不存在: {input_file_path}")
            return

        # 读取 .jsonl 文件，每行一个 JSON 对象
        data = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # 写入为标准 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"转换成功！\n原始文件: {input_file_path}\n新文件: {output_file_path}")

    except Exception as e:
        print(f"转换过程中出错: {e}")

input_path = "/data/home/chenyu/Coding/SD+SoT/Speculative-Decoding-Enabled-Skeleton-of-Thought/datasets/planning/industry_tasks.jsonl"
output_path = "/data/home/chenyu/Coding/SD+SoT/Speculative-Decoding-Enabled-Skeleton-of-Thought/datasets/planning/industry_tasks_no_ascii.json"

# 调用函数进行转换  
convert_jsonl_to_readable(input_path, output_path)