以下是关于 Vicuna、Qwen (2.5/3) 和 Llama 3.1 三种主流模型 Prompt Template 的详细整理，包含格式说明、从 Tokenizer 获取的方法以及手动构建的代码示例。

LLM 推理 Prompt Template 详解指南
1. Vicuna (v1.3)
Vicuna 是早期的经典模型，基于 Llama 2 微调，它的模板非常简单且经典，类似于对话脚本。

模板格式
Vicuna v1.3 使用 USER: 和 ASSISTANT: 作为角色标识，中间用空格分隔。系统提示词（System Prompt）通常放在最前面。

结构示意：

Plaintext
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {用户输入}
ASSISTANT:
多轮对话结构：

Plaintext
{System Prompt}

USER: {用户输入1}
ASSISTANT: {模型回答1}</s>
USER: {用户输入2}
ASSISTANT:
注意：Vicuna 通常使用 </s> (EOS token) 作为轮次结束符。

2. Qwen 系列 (Qwen2.5 / Qwen3)
注：截至目前（2025/2026），Qwen3 可能沿用 Qwen2.5 的 ChatML 格式。Qwen 系列主要使用 ChatML 格式，这是一种更结构化的格式。

模板格式 (ChatML)
它使用 <|im_start|> 和 <|im_end|> 特殊 token 来包裹每一句话，并明确指定角色（system, user, assistant）。

结构示意：

Plaintext
<|im_start|>system
{System Prompt}<|im_end|>
<|im_start|>user
{用户输入}<|im_end|>
<|im_start|>assistant
多轮对话结构：

Plaintext
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮你的吗？<|im_end|>
<|im_start|>user
讲个笑话<|im_end|>
<|im_start|>assistant
3. Llama 3.1 (8B/70B/405B)
Llama 3 和 3.1 引入了一套全新的、基于 Reserved Special Tokens 的格式，不再像 Llama 2 那样只使用 [INST]。

模板格式
它使用 <|begin_of_text|> 开始，使用 <|start_header_id|> 和 <|end_header_id|> 来标记角色，内容存放在 <|eot_id|> (End of Turn) 之前。

结构示意：

Plaintext
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{System Prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{用户输入}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
多轮对话结构：

Plaintext
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|><|start_header_id|>user<|end_header_id|>

Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
4. 如何自动获取与使用 (推荐)
现在绝大多数 HuggingFace 模型都在 tokenizer_config.json 中内置了 chat_template 属性。强烈建议直接使用 Tokenizer 自带的模板应用方法，而不是手动拼接字符串，这样可以避免空格、换行符和特殊 token 的细微错误。

使用 tokenizer.apply_chat_template
这是最标准、最通用的方法，适用于上述所有模型。

Python
from transformers import AutoTokenizer

# 1. 加载 Tokenizer (自动加载对应的 template)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
# 或者 "Qwen/Qwen2.5-7B-Instruct", "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 构建消息列表 (标准 OpenAI 格式)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "解释一下量子力学"},
    {"role": "assistant", "content": "量子力学是物理学的一个分支..."}, # 多轮对话示例
    {"role": "user", "content": "那它和相对论有什么区别？"}
]

# 3. 应用模板
# tokenize=False: 返回拼接好的字符串（用于检查格式）
# tokenize=True: 直接返回 input_ids (用于模型输入)
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("=== 最终生成的 Prompt ===")
print(prompt_str)

# 4. 转换为 input_ids 并送入模型
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

# model.generate(inputs, ...)
关键参数说明：
add_generation_prompt=True: 这一点非常关键。它会自动在最后添加模型生成的引导符（例如 Llama 3 的 <|start_header_id|>assistant<|end_header_id|> 或 Qwen 的 <|im_start|>assistant），提示模型开始生成。

5. 如何手动构建 (如果不使用 HF Tokenizer)
如果你在特定环境（如 C++ 推理、纯 PyTorch 实现）无法调用 apply_chat_template，可以使用 Python 字符串拼接实现。

通用手动构建代码示例
Python
def build_prompt(model_type, messages):
    prompt = ""
    
    if model_type == "vicuna":
        # Vicuna 格式
        system_prompt = "A chat between a curious user and an artificial intelligence assistant..."
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
        
        prompt = f"{system_prompt}\n\n"
        for msg in messages:
            if msg['role'] == 'user':
                prompt += f"USER: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"ASSISTANT: {msg['content']}</s>\n"
        prompt += "ASSISTANT:" # 引导生成

    elif model_type == "qwen":
        # Qwen ChatML 格式
        # 注意：实际编码时 <|im_start|> 等通常是特殊 token ID，这里演示文本拼接
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    elif model_type == "llama3":
        # Llama 3 格式
        prompt = "<|begin_of_text|>"
        for msg in messages:
            prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            prompt += f"{msg['content']}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt

# 测试数据
msgs = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"}
]

# print(build_prompt("llama3", msgs))