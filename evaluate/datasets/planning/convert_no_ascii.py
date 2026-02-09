import json
import os

file_path_dir = os.path.dirname(os.path.abspath(__file__))

def convert_jsonl_to_readable(input_file_path, output_file_path):
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"输入文件不存在: {input_file_path}")
            return

        # 使用 'w' 模式打开输出文件，准备写入
        # 同时打开输入文件进行读取（流式处理，节省内存）
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            count = 0
            for line in f_in:
                line = line.strip()
                if line:
                    # 1. json.loads 会自动把 \uXXXX 这种 ASCII 转义符变成正常的中文
                    json_obj = json.loads(line)
                    
                    # 2. 使用 json.dumps 生成字符串
                    # ensure_ascii=False 是关键，它让中文直接显示，而不是显示为 \u 编码
                    # 3. 手动加上换行符 \n 保持 JSONL 格式
                    new_line = json.dumps(json_obj, ensure_ascii=False)
                    f_out.write(new_line + "\n")
                    
                    count += 1

        print(f"转换成功！共处理 {count} 行数据。")
        print(f"新文件: {output_file_path}")

    except Exception as e:
        print(f"转换过程中出错: {e}")

input_path = os.path.join(file_path_dir, "industry_tasks.jsonl")
output_path = os.path.join(file_path_dir, "industry_tasks_converted.jsonl")

# 调用函数进行转换  
convert_jsonl_to_readable(input_path, output_path)