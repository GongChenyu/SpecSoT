# coding=utf-8
"""
SpecSoT Prompts - Multi-Turn Dialogue Prompt System

This module provides a multi-turn dialogue based prompt system for SpecSoT.
All prompt-related content and methods are centralized here.

Architecture:
- System Prompt: Base prompt containing user input, serves as PREFIX for both phases
- Skeleton Prompt: First turn user input for skeleton generation phase  
- Parallel Prompt: Second turn user input for parallel branch execution phase

Chat Template Structure:
- Vicuna:  {Vicuna Header}\n\nUSER: {system_prompt}\n{user_prompt}\nASSISTANT:
- Qwen:    <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
- Llama3:  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>...

Key Design:
1. PREFIX = System Prompt with user input (reused in both phases)
2. Skeleton Phase: PREFIX + USER: skeleton_prompt + ASSISTANT:
3. Parallel Phase: PREFIX + USER: parallel_prompt (per branch) + ASSISTANT:
4. IMPORTANT: Remove [PLAN]/[END] markers from skeleton context to avoid early stopping
"""

import re
from typing import List, Tuple, Optional, Dict
import torch


# =============================================================================
# System Prompt (Base Prompt) - Contains User Input, Used as PREFIX
# =============================================================================

system_prompt = (
    "[System Directive] You are a logic expert specializing in complex task planning and execution.\n"
    "Your workflow has two strictly separated phases:\n"
    "1. **Planning Phase**: Decompose complex queries into independent parallel sub-tasks.\n"
    "2. **Execution Phase**: Execute a specific step based on the established plan.\n"
    "You must strictly adhere to the instructions for the current phase.\n"
    "[User Input]:\n{user_input}\n"
)


# =============================================================================
# Skeleton Prompt - For Skeleton Generation Phase (First Turn)
# =============================================================================

skeleton_prompt = (
    "[Current Task: Skeleton Generation]\n"
    "Analyze the input and output strictly in ONE of the following formats:\n\n"
    "### Format 1: Direct Answer (For simple/atomic/non-parallelizable tasks)\n"
    "Output:\n"
    "[DIRECT]\n"
    "(Write your answer content here...)\n"
    "[END]\n\n"
    "### Format 2: Decomposition Plan (For tasks with independent parallel branches)\n"
    "Output (compact format, no spaces between fields):\n"
    "[PLAN]\n"
    "1.<Est_Tokens><Tool_Name>[-]Branch 1 Title\n"
    "2.<Est_Tokens><Tool_Name>[-]Branch 2 Title\n"
    "...\n"
    "[END]\n\n"
    "**Constraints**:\n"
    "1. **Length**: Must be a number (supports multi-digit like <127>, <1500>).\n"
    "2. **Tool**: Use <None> if no tool is needed, otherwise <Search>, etc.\n"
    "3. **Topology**: Strictly use [-] for now to indicate parallel execution.\n"
    "4. **Branch Title**: The brief title should be as short as possible.\n"
    "5. Start immediately with [DIRECT] or [PLAN].\n"
    "6. Both formats MUST end with [END]. [END] must not be followed by any content.\n"
    "7. **Compact Format**: No spaces between fields.\n\n"
    "### Example 1: Direct Answer\n"
    "User asks: What is 1+1?\n"
    "Output:\n"
    "[DIRECT]\n"
    "1+1 equals 2.\n"
    "[END]\n\n"
    "### Example 2: Decomposition Plan\n"
    "User asks: Introduce the top 3 cities in China separately.\n"
    "Output:\n"
    "[PLAN]\n"
    "1.<154><None>[-]Introduce Beijing\n"
    "2.<543><None>[-]Introduce Shanghai\n"
    "3.<456><None>[-]Introduce Guangzhou\n"
    "[END]\n\n"
    "[Output]:"
)


# =============================================================================
# Parallel Prompt - For Parallel Execution Phase (Second Turn per Branch)
# =============================================================================

parallel_prompt = (
    "[Current Task: Execute Specific Step]\n"
    "Overall Plan:\n"
    "{skeleton_context}\n"
    "Your task is to execute Step **{current_id}**: **{current_point}**\n"
    "**Requirements**:\n"
    "1. Target Length: Approx **{target_length}** tokens.\n"
    "2. The output must start with [ANSWER] and please output the body content directly. DO NOT include IDs, titles.\n"
    "3. If you finished the answer of this branch, the answer must end with [END]. Do not output [END] at the beginning.\n"
    "4. Ensure logical flow with the [User Input].\n"
    "[ANSWER]"
)


# =============================================================================
# Chat Template Builders - Build prompts according to model-specific templates
# =============================================================================


def build_prompt(model_type: str, system_content: str, user_content: str) -> str:
    """
    构建完整的 prompt 字符串（包含 system + user + assistant 标记）。
    用于 skeleton 阶段的完整输入。
    
    Args:
        model_type: 'vicuna', 'qwen', 'llama', or 'other'
        system_content: System prompt 内容（包含用户输入）
        user_content: User prompt 内容（skeleton prompt）
        
    Returns:
        完整的 prompt 字符串
    """
    if model_type == 'vicuna':
        # Vicuna 格式: {system}\nUSER: {user}\nASSISTANT:
        return f"{system_content}\nUSER: {user_content}\nASSISTANT:"
    
    elif model_type == 'qwen':
        # Qwen ChatML 格式
        return (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    elif model_type == 'llama':
        # Llama 3 格式
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    else:
        # 默认格式：简单拼接
        return f"{system_content}\n{user_content}"


def build_prefix(model_type: str, system_content: str) -> str:
    """
    构建 PREFIX 部分（仅 system prompt）。
    这部分在 skeleton 和 parallel 阶段复用其 KV cache。
    
    Args:
        model_type: 'vicuna', 'qwen', 'llama', or 'other'
        system_content: System prompt 内容（已填充用户输入）
        
    Returns:
        PREFIX 字符串
    """
    if model_type == 'vicuna':
        # Vicuna: system content + 换行
        return f"{system_content}\n"
    
    elif model_type == 'qwen':
        # Qwen: system 消息完整格式
        return f"<|im_start|>system\n{system_content}<|im_end|>\n"
    
    elif model_type == 'llama':
        # Llama 3: system 消息完整格式
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_content}<|eot_id|>"
        )
    
    else:
        return system_content


# =============================================================================
# Utility Functions - Extract skeleton context without markers
# =============================================================================

def extract_skeleton_context(skeleton_text: str) -> str:
    """
    Extract skeleton content without [PLAN]/[END] or [DIRECT]/[END] markers.
    
    IMPORTANT: This is critical to avoid early stopping in parallel phase.
    The [END] marker in the skeleton would cause the model to stop prematurely.
    
    Args:
        skeleton_text: Raw skeleton output from model
        
    Returns:
        Clean skeleton context without markers
    """
    skeleton_context = skeleton_text.strip()
    
    if "[PLAN]" in skeleton_text:
        try:
            start = skeleton_text.index("[PLAN]") + 6
            end = skeleton_text.index("[END]") if "[END]" in skeleton_text else len(skeleton_text)
            skeleton_context = skeleton_text[start:end].strip()
        except ValueError:
            pass
    elif "[DIRECT]" in skeleton_text:
        skeleton_context = skeleton_text.replace("[DIRECT]", "").replace("[END]", "").strip()
    
    return skeleton_context


# =============================================================================
# Main API Functions - Prepare inputs for inference
# =============================================================================

def prepare_skeleton_input(
    tokenizer,
    task_prompt: str,
    model_type: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare skeleton generation phase input.
    
    Structure (example for Vicuna):
        {Vicuna Header}
        
        USER: {system_prompt with user_input}
        {skeleton_prompt}
        ASSISTANT:
    
    PREFIX is: {Vicuna Header}\n\nUSER: {system_prompt with user_input}\n
    
    Args:
        tokenizer: Tokenizer instance
        task_prompt: User's original input/question
        model_type: 'vicuna', 'qwen', 'llama', or 'other'
        device: Target device
        
    Returns:
        input_ids: Complete input sequence [1, seq_len]
        prefix_ids: PREFIX part [1, prefix_len] (for reuse in parallel phase)
    """
    # Build system content (PREFIX content, contains user input)
    system_content = system_prompt.format(user_input=task_prompt)
    
    # Build user content (skeleton prompt)
    user_content = skeleton_prompt
    
    if model_type == 'llama':
        # Use tokenizer's apply_chat_template for Llama
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        # For prefix, only include system message (without add_generation_prompt)
        prefix_messages = [{"role": "system", "content": system_content}]
        prefix_ids = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(device)
        
        print(f"[Llama] Using apply_chat_template | input_len={input_ids.shape[1]}, prefix_len={prefix_ids.shape[1]}")
        
    elif model_type == 'qwen':
        # Build Qwen ChatML format
        full_prompt = build_prompt(model_type, system_content, user_content)
        input_ids = tokenizer([full_prompt], return_tensors="pt").input_ids.to(device)
        
        # Build prefix (system part only)
        prefix_text = build_prefix(model_type, system_content)
        prefix_ids = tokenizer([prefix_text], return_tensors="pt").input_ids.to(device)
        
        print(f"[Qwen] Using ChatML format | input_len={input_ids.shape[1]}, prefix_len={prefix_ids.shape[1]}")
        
    elif model_type == 'vicuna':
        # Build Vicuna format
        full_prompt = build_prompt(model_type, system_content, user_content)
        input_ids = tokenizer([full_prompt], return_tensors="pt").input_ids.to(device)
        
        # Build prefix
        prefix_text = build_prefix(model_type, system_content)
        prefix_ids = tokenizer([prefix_text], return_tensors="pt").input_ids.to(device)
        
        print(f"[Vicuna] Using Vicuna chat format | input_len={input_ids.shape[1]}, prefix_len={prefix_ids.shape[1]}")
        
    else:
        # Default: simple concatenation
        full_prompt = build_prompt(model_type, system_content, user_content)
        input_ids = tokenizer([full_prompt], return_tensors="pt").input_ids.to(device)
        
        prefix_text = build_prefix(model_type, system_content)
        prefix_ids = tokenizer([prefix_text], return_tensors="pt").input_ids.to(device)
        
        print(f"[Other] Using default format | input_len={input_ids.shape[1]}, prefix_len={prefix_ids.shape[1]}")
    
    return input_ids, prefix_ids


def prepare_parallel_inputs(
    tokenizer,
    tasks: List[Dict],
    skeleton_text: str,
    model_type: str = 'qwen',
    task_prompt: str = "",
) -> Tuple[List[List[int]], List[int]]:
    """
    准备 parallel 阶段的分支输入。
    
    关键设计：
    =========
    在 skeleton 阶段，我们 prefill 了：
      1. system prompt 的 KV cache
      2. skeleton prompt 的 KV cache  
      3. skeleton 生成内容的 KV cache
    
    在 parallel 阶段：
      - 我们需要删除 skeleton prompt 和 skeleton 生成内容的 cache
      - 保留 system prompt 的 cache 作为 prefix（复用已有 cache）
      - 只需要构建 parallel prompt（user turn）作为输入
      - **不需要**再次添加 system prompt，因为其 cache 已存在
    
    因此，这里构建的 token 序列只包含 user turn 部分：
      - Vicuna: "USER: {parallel_prompt}\nASSISTANT:"
      - Qwen: "<|im_start|>user\n{parallel_prompt}<|im_end|>\n<|im_start|>assistant\n"
      - Llama: "<|start_header_id|>user<|end_header_id|>\n\n{parallel_prompt}<|eot_id|>..."
    
    注意事项：
      - 提取 skeleton context 时需要移除 [PLAN]/[END] 标记，避免提前停止
    
    Args:
        tokenizer: Tokenizer 实例
        tasks: 从 parse_skeleton_output 解析出的任务列表
        skeleton_text: 原始 skeleton 输出（会被清理）
        model_type: 'vicuna', 'qwen', 'llama', or 'other'
        task_prompt: 原始用户输入（此参数在当前实现中不再使用，保留以兼容）
        
    Returns:
        branch_token_ids: 每个分支的 token ID 列表（仅包含 user turn）
        instruction_lengths: 每个分支的指令长度
    """
    # 提取干净的 skeleton context（移除 [PLAN]/[END] 标记）
    skeleton_context = extract_skeleton_context(skeleton_text)
    
    branch_token_ids = []
    instruction_lengths = []
    
    for task in tasks:
        # 构建 user content（parallel prompt，填充占位符）
        user_content = parallel_prompt.format(
            skeleton_context=skeleton_context,
            current_id=task['id'],
            current_point=task['title'],
            target_length=task['length'],
        )
        
        # 关键：只构建 user turn 部分，不包含 system prompt
        # 因为 system prompt 的 KV cache 已经存在，会作为 prefix 复用
        # 根据不同的模型类型构建 user turn prompt
        if model_type == 'vicuna':
            # Vicuna 格式: USER: {user}\nASSISTANT:
            user_turn_prompt = f"USER: {user_content}\nASSISTANT:"
        
        elif model_type == 'qwen':
            # Qwen ChatML 格式: user turn + assistant 开始
            user_turn_prompt = (
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        
        elif model_type == 'llama':
            # Llama 3 格式: user turn + assistant 开始
            user_turn_prompt = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        
        else:
            # 默认格式
            user_turn_prompt = f"\n{user_content}"
        
        token_ids = tokenizer.encode(user_turn_prompt, add_special_tokens=False)
        
        branch_token_ids.append(token_ids)
        instruction_lengths.append(len(token_ids))
    
    return branch_token_ids, instruction_lengths


# =============================================================================
# Skeleton Parser - Parse skeleton output to extract tasks
# =============================================================================

def parse_skeleton_output(text: str) -> Tuple[str, object]:
    """
    Parse skeleton format output.
    
    Format:
    - Mode A: [DIRECT]...[END] - Direct answer
    - Mode B: [PLAN]...[END] - Decomposition plan
    
    Line format: ID.<Length><Tool>[Deps]Title
    
    Args:
        text: Model generated text
        
    Returns:
        Tuple[mode, content]:
        - mode: "direct" | "plan" | "error"
        - content: 
            - direct: str (answer content)
            - plan: List[dict] (task list)
            - error: str (error message)
    """
    text = text.strip()
    
    # Mode A: Direct answer
    if text.startswith("[DIRECT]"):
        content = text.replace("[DIRECT]", "", 1).strip()
        if "[END]" in content:
            content = content[:content.index("[END]")].strip()
        return "direct", content
    
    # Mode B: Plan mode
    elif "[PLAN]" in text:
        try:
            start = text.index("[PLAN]") + 6
            end = text.index("[END]") if "[END]" in text else len(text)
            plan_body = text[start:end].strip()
        except ValueError:
            return "error", "Malformed tags: [PLAN] or [END] not found properly"

        tasks = []
        # Regex pattern supporting both compact and loose formats
        pattern = re.compile(
            r"(\d+)[.:、]\s*"                     # ID + separator
            r"[<（]?(\d+)[>）]?\s*"               # Length
            r"[<（]?(\S+?)[>）]?\s*"              # Tool
            r"[\[【](.*?)[\]】]\s*"               # [Deps]
            r"(.+)"                               # Title
        )
        
        for line in plan_body.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = pattern.search(line)
            if match:
                t_id = int(match.group(1))
                t_len = int(match.group(2))
                t_tool = match.group(3).strip()
                t_deps_str = match.group(4).strip()
                t_title = match.group(5).strip()
                
                if t_tool.lower() == "none":
                    t_tool = None
                
                tasks.append({
                    "id": t_id,
                    "length": t_len,
                    "tool": t_tool,
                    "deps": t_deps_str,
                    "title": t_title,
                    "raw_line": line,
                })
        
        if not tasks:
            return "error", f"No valid tasks parsed from plan body: {plan_body[:200]}"
            
        return "plan", tasks
    
    # Default: treat as direct answer
    else:
        return "direct", text

