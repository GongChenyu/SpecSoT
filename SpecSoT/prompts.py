# coding=utf-8

# =============================================================================
# Chinese Prompts (中文提示词) - for Qwen models
# =============================================================================

base_prompt = (
    "【系统角色】\n"
    "你是一个擅长处理各种简单和复杂问题的 AI 助手。"
    "针对简单问题或者是不可拆分的具有逻辑性的问题，请直接给出简洁明了的答案；"
    "针对可以拆分的问题，我们需要采用“思维骨架”（Skeleton-of-Thought）的方法：先规划结构，再并行填充内容。\n\n"
    "【用户输入】\n"
    "{user_question}\n\n"
)

skeleton_trigger_zh = (
    "【步骤一指令：输入分析与骨架拆分】\n"
    " 1. 具体要求：当回复复杂且适合**并行处理**（如多工具和智能体调用、多角度分析、多方面列举、长文写作规划）时，请按以下格式输出骨架：\n"
    "   - 每个分支独立一行，以 '####' 开头。\n"
    "   - 紧接着是简短的【分支标题】。\n"
    "   - 在标题后的括号 '()' 内，标注你预测该分支生成内容的【Token数量】（纯数字）。\n"
    "   - 以英文冒号 ':' 结尾。\n"
    "   - 冒号后必须紧跟省略号 '...'，禁止在此生成内容。\n"
    "   - 结尾输出 '####%%%%'。\n"
    "   - 示例：\n"
    "     ####标题1(169):...\n"
    "     ####标题2(431):...\n"
    "     ####%%%%\n\n"
    " 2. 如果用户输入的回答很简单或者具备强烈的逻辑顺序依赖，请直接回答问题，无需生成骨架。\n"
    "【回答】：禁止输出分析和开场白，回答：\n"
)

parallel_trigger_zh = (
    "【步骤二指令：内容填充】\n"
    "我们已经制定了该问题的整体回答骨架（含预计生成token长度）如下：\n"
    "{skeleton_context}\n\n" 
    "你现在的具体任务是撰写其中一个分支的内容。\n"
    "当前任务分支：【 {current_point} 】\n\n"
    "严格约束：\n"
    "1. 直接开始撰写该分支的正文内容。\n"
    "2. 参考骨架中的预估token长度进行撰写，不要过短或过长。\n" 
    "3. 禁止输出标题或“####”，禁止输出开场白。\n"
    "4. 内容需独立完整，表达简洁，逻辑上能衔接整体骨架。\n"
    "5. 禁止重复输出内容，适时结束子分支的回答,结束符号是<|im_end|>。\n\n"
    "【直接开始回答】：\n"
)

# =============================================================================
# English Prompts (英文提示词) - for Llama and other models
# =============================================================================

base_prompt_en = (
    "[System Role]\n"
    "You are an AI assistant skilled at handling both simple and complex questions. "
    "For simple questions or indivisible logical problems, provide a concise and clear answer directly. "
    "For divisible problems, we use the 'Skeleton-of-Thought' approach: plan the structure first, then fill in the content in parallel.\n\n"
    "[User Input]\n"
    "{user_question}\n\n"
)

skeleton_trigger_en = (
    "[Step 1 Instruction: Input Analysis and Skeleton Decomposition]\n"
    " 1. Specific Requirements: When the response is complex and suitable for **parallel processing** (such as multi-tool and agent calls, multi-perspective analysis, multi-aspect enumeration, long-form writing planning), please output the skeleton in the following format:\n"
    "   - Each branch on a separate line, starting with '####'.\n"
    "   - Followed by a concise [Branch Title].\n"
    "   - In parentheses '()' after the title, indicate your predicted [Token Count] for this branch's content (pure number).\n"
    "   - End with an English colon ':'.\n"
    "   - The colon must be immediately followed by an ellipsis '...', and you are forbidden to generate content here.\n"
    "   - End with '####%%%%'.\n"
    "   - Example:\n"
    "     ####Title 1(169):...\n"
    "     ####Title 2(431):...\n"
    "     ####%%%%\n\n"
    " 2. If the user input is very simple or has strong logical sequence dependencies, please answer the question directly without generating a skeleton.\n"
    "[Answer]: No analysis or preamble allowed, answer:\n"
)

parallel_trigger_en = (
    "[Step 2 Instruction: Content Filling]\n"
    "We have established the overall answer skeleton for this question (with estimated token length) as follows:\n"
    "{skeleton_context}\n\n"
    "Your specific task now is to write the content for one of the branches.\n"
    "Current Task Branch: [ {current_point} ]\n\n"
    "Strict Constraints:\n"
    "1. Start writing the main content for this branch directly.\n"
    "2. Refer to the estimated token length in the skeleton for writing, avoiding being too short or too long.\n"
    "3. Do not output titles or '####', and do not output preambles.\n"
    "4. The content should be independently complete, concisely expressed, and logically connected to the overall skeleton.\n"
    "5. Do not repeat content, end the sub-branch's answer appropriately, the end symbol is <|im_end|>.\n\n"
    "[Start Answering Directly]:\n"
)



















