# coding=utf-8

# =============================================================================
# Chinese Prompts (中文提示词) - Optimized for Qwen
# =============================================================================

# 基础Prompt：确立专家人设，强调逻辑判断
base_prompt_zh = (
    "【系统指令：思维骨架生成引擎】\n"
    "你是一个擅长处理复杂任务的逻辑分析专家。你的核心能力是能够瞬间判断一个问题是应该“直接回答”还是需要“拆解规划”。\n"
    # "其中“拆解规划”的工作流程严格分为两个独立步骤： 1. **步骤一（规划）**：判断任务类型并生成骨架。 2. **步骤二（执行）**：根据骨架并行填充内容。\n"
    # "你只会收到执行**其中一个步骤**的指令。 **严禁越界**：如果你处于“步骤一”，绝对不要生成正文内容。如果你处于“步骤二”，绝对不要重复生成骨架。\n"
    "【用户输入】内容：\n"
    "<user_input>\n"
    "{user_inputs}\n"
    "</user_input>\n\n"
)

# 阶段一：骨架生成触发器
# 优化点：
# 1. 增加了“判断标准”部分，教模型如何分类。
# 2. 强化了格式的正则约束，确保小模型不乱加空格或换行。
skeleton_trigger_zh = (
    "【当前任务：输入分析与骨架生成】\n"
    "请对 <user_input> 进行逻辑评估，并从以下两种方法中选择一条执行：\n\n"

    "方法 A：直接回答（线性/简单任务）\n"
    "判定标准：\n"
    "1. 问题简单，一两句话能说清（如：常识问答、简单翻译）。\n"
    "2. 具有强烈的**逻辑先后顺序**，步骤B必须等步骤A完成后才能进行，**无法并行处理**。\n"
    "执行操作：直接输出答案，**禁止**生成骨架或“####”。\n\n"

    "方法 B：思维骨架拆分（并行/复杂任务）\n"
    "判定标准：回答需要多角度分析、多模块列举、长文写作，且各部分内容**相对独立**，可以同时撰写。\n"
    "执行操作：请按严格格式输出骨架。\n\n"

    "**骨架格式严格约束（机器读取专用）：**\n"
    "1. 每一行代表一个独立的分支，格式必须为：`#### 标题【预估Token数】:...`\n"
    "2. `####` 开头，紧跟标题。\n"
    "3. `【数字】` 代表预估该段落需要的Token数量。\n"
    "4. 必须以英文冒号 `:` 和省略号 `...` 结尾。**严禁**在此冒号后写任何正文内容！\n"
    "5. 所有分支列举完后，skeleton生成结束，结束符是 `####%%%%`。\n\n"
    
    "<user_input>: prompt...\n"
    "**标准骨架输出示例：**\n"
    "#### 分支标题【342】:...\n"
    "#### 分支标题【521】:...\n"
    "####%%%%\n\n"

    "<user_input>: 分析新能源汽车的发展前景\n"
    "**具体骨架输出示例：**\n"
    "#### 技术维度【329】:...\n"
    "#### 市场维度【127】:...\n"
    "#### 政策维度【663】:...\n"
    "####%%%%\n\n"

    "**警告**：\n"
    "- 无论用户输入多短（例如仅有“写个大纲”），只要任务适合拆分，就必须执行方法B。\n"
    "- 绝对不要输出任何分析过程、开场白或“好的”、“明白”等废话。\n"
    "- 直接开始输出答案或骨架。\n\n"
    "再次重复【用户输入】内容：\n"
    "<user_input>\n"
    "{user_inputs}\n"
    "</user_input>\n\n"
    "禁止输出开场白，【回答】：\n"
)

# 阶段二：并行填充触发器
# 优化点：
# 1. 强调上下文的联系，防止子回答显得没头没尾。
# 2. 明确结束符，防止小模型喋喋不休。
parallel_trigger_zh = (
    "【系统指令：并行内容生成】\n"
    "整体回答结构已规划如下：\n"
    "<skeleton>\n"
    "{skeleton_context}"
    "</skeleton>\n"
    "【当前具体任务】请撰写【当前分支】的详细正文内容。\n"
    "【撰写要求】：\n"
    "1. **格式**：直接输出正文。**禁止**包含 `####`、标题、序号或开场白（如“关于这一点...”）。\n"
    "2. **长度**：参考规划中的Token预估值，不要在这一个分支上输出偏离预测值。\n"
    "3. **逻辑**：内容必须完整，且能与上下文自然衔接。\n"
    "4. **结束**：本部分内容写完后，请立即停止，禁止重复输出。结束符号为<|im_end|> \n\n"
    "你只被允许针对【当前分支】回答，【当前分支】：\n"
    "{current_point} "
    "禁止回答其他分支，禁止输出开场白，【回答】：\n"
)


# =============================================================================
# English Prompts (英文提示词) - Optimized for Llama
# =============================================================================

# Base Prompt: Establish expert persona, emphasize logical judgment and step isolation
base_prompt_en = (
    "[System Directive: Skeleton-of-Thought Generation Engine]\n"
    "You are an expert logical analyst skilled in handling complex tasks. Your core capability is to instantly determine whether a problem requires a 'Direct Answer' or 'Decomposition Planning'.\n"
    "The 'Decomposition Planning' workflow is strictly divided into two independent steps: 1. **Step 1 (Planning)**: Determine task type and generate the skeleton. 2. **Step 2 (Execution)**: Fill in content in parallel based on the skeleton.\n"
    "You will receive instructions to execute **ONLY ONE** of these steps at a time. **STRICT ISOLATION**: If you are in 'Step 1', DO NOT generate body content. If you are in 'Step 2', DO NOT regenerate the skeleton.\n"
    "[User Input] Content:\n"
    "<user_input>\n"
    "{user_inputs}\n"
    "</user_input>\n\n"
)

skeleton_trigger_en = (
    "[Task: Input Analysis & Path Selection]\n"
    "Evaluate the <user_input> and execute ONE of the following two paths:\n\n"

    "### Path A: Direct Answer (Linear/Simple Task)\n"
    "Criteria:\n"
    "1. The query is simple (e.g., factual, short translation).\n"
    "2. The task has strict **sequential dependency** (Step B requires Step A to be finished first). It CANNOT be processed in parallel.\n"
    "Action: Output the answer directly. **DO NOT** generate a skeleton or use '####'.\n\n"

    "### Path B: Skeleton Decomposition (Parallel/Complex Task)\n"
    "Criteria: The answer requires multi-angle analysis, independent modules, or long-form writing where parts are **independent** and can be written simultaneously.\n"
    "Action: Output a structural skeleton using the strict format below.\n\n"

    "**Strict Skeleton Format (For Machine Parsing):**\n"
    "1. Each branch is a separate line: `#### Title【TokenCount】:...`\n"
    "2. Must start with `####` followed by a brief title.\n"
    "3. `【Number】` indicates estimated tokens for that section.\n"
    "4. Must end with a colon `:` and ellipsis `...`. **DO NOT** generate content after the colon.\n"
    "5. After all branches, output the specific end token: `####%%%%`\n\n"
    
    "**Standard Example:**\n"
    "prompt: xxx...\n"
    "Skeleton:\n"
    "#### branch【342】:...\n"
    "#### branch【521】:...\n"
    "####%%%%\n\n"

    "**Example:**\n"
    "prompt: Analyze the impact of remote work.\n"
    "Skeleton:\n"
    "#### Economic Impact【329】:...\n"
    "#### Social Impact【127】:...\n"
    "#### Mental Health【663】:...\n"
    "####%%%%\n\n"

    "**WARNING**:\n"
    "- Even for short inputs (e.g., 'Plan a trip'), if the task is divisible, execute Path B.\n"
    "- Do NOT output preambles, analysis thoughts, or conversational fillers like 'Sure' or 'Okay'.\n"
    "- Start the output directly.\n\n"
    "[Answer Directly]:\n"
)

parallel_trigger_en = (
    "[System Directive: Parallel Content Generation]\n"
    "Context: We are answering the query \"{user_inputs}\".\n"
    "The overall structure has been planned as follows:\n"
    "{skeleton_context}\n\n"
    "--------------------------------------------------\n"
    "[Current Micro-Task]\n"
    "Write the detailed content for the branch: **[ {current_point} ]**\n\n"

    "[Writing Constraints]:\n"
    "1. **Format**: Output the body content directly. **FORBIDDEN** to use `####`, titles, or preambles.\n"
    "2. **Length**: Adhere to the estimated token count in the plan.\n"
    "3. **Logic**: Ensure the content is self-contained but flows well with the context.\n"
    "4. **Stop**: Stop writing immediately after covering this specific point.\n\n"
    "[Answer Directly]:\n"
)


# =============================================================================
# Vicuna Chat Template
# =============================================================================

# Vicuna 模型使用特定的对话格式
# 格式: A chat between a curious user and an AI assistant.
#       USER: {prompt}
#       ASSISTANT:
vicuna_chat_template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    "USER: {prompt}\n"
    "ASSISTANT:"
)