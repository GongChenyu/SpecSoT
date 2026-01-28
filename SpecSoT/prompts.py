# coding=utf-8
"""
SpecSoT Prompts - Topology-Aware Skeleton Protocol

骨架协议定义：
- 模式 A：直接回答 [DIRECT] - 用于简单、原子或无法拆分的任务
- 模式 B：规划模式 [PLAN]...[END] - 用于复杂任务的并行拆解

简化行格式：ID.<Length><Tool>[-]Title (无空格)
- ID: 数字加点 (1. 2. ...)
- Length: <数字> 预估 Token 数量 (支持多位数如 <127>, <1500>)
- Tool: <工具名> 或 <None>
- Deps: [-] 表示纯并行
- Title: 任务描述
"""

# =============================================================================
# Chinese Prompts (中文提示词) - Protocol
# =============================================================================

# 中文 Base Prompt
base_prompt_zh = (
    "【系统指令】你是一个精通复杂任务规划与执行的逻辑专家。\n"
    "你的工作模式分为两个阶段：\n"
    "1. **规划阶段 ([PLAN])**：将复杂问题拆解为多个独立的并行子任务。\n"
    "2. **执行阶段 ([WORK])**：根据既定计划，执行其中某一个具体步骤。\n"
    "你必须严格遵守指令给出的当前阶段要求，严禁越界。\n"
    "【用户输入】：\n{user_inputs}\n\n"
)

# 中文 Skeleton Trigger (骨架生成阶段)
skeleton_trigger_zh = (
    "【当前任务：生成骨架】\n"
    "请判断用户输入适合「直接回答」还是「拆解规划」。请从以下两种格式中严格选择一种输出：\n\n"
    "### 格式一：直接回答（适用于简单/原子/不可并行任务）\n"
    "输出格式：\n"
    "[DIRECT]\n"
    "(在此处直接写出答案内容...)\n"
    "[END]\n\n"
    "### 格式二：拆解规划（适用于分支独立的可并行任务）\n"
    "输出格式（紧凑格式，无空格）：\n"
    "[PLAN]\n"
    "1.<预估长度><工具名>[-]分支一标题\n"
    "2.<预估长度><工具名>[-]分支二标题\n"
    "...\n"
    "[END]\n\n"
    "**严格约束**：\n"
    "1. **长度**：必须填写数字，如 <127><793><1500>等，支持三位数。\n"
    "2. **工具**：如果不需要外部工具，请填写 <None>；如果需要，填写工具名如 <Search>。\n"
    "3. **拓扑**：目前请强制使用 [-]，表示所有步骤并行执行，互不依赖。\n"
    "4. 直接以 [DIRECT] 或 [PLAN] 开头，不要废话。\n"
    "5. 两种格式都必须以 [END] 结尾。[END]后面禁止输出任何内容。\n"
    "6. **紧凑格式**：各字段之间不要有空格。\n\n"
    "### 示例一：直接回答\n"
    "用户问：1+1等于多少？\n"
    "输出：\n"
    "[DIRECT]\n"
    "1+1等于2。\n"
    "[END]\n\n"
    "### 示例二：拆解规划\n"
    "用户问：分别介绍中国的三大城市\n"
    "输出：\n"
    "[PLAN]\n"
    "1.<652><None>[-]介绍北京\n"
    "2.<234><None>[-]介绍上海\n"
    "3.<356><None>[-]介绍广州\n"
    "[END]\n\n"
    "【开始输出】：\n"
)

# 中文 Parallel Trigger (并行扩展阶段)
parallel_trigger_zh = (
    "【当前任务：分支扩展】\n"
    "整体计划如下：\n"
    "{skeleton_context}\n"
    "--------------------\n"
    "你的任务是执行步骤 **{current_id}**：\n"
    "**{current_point}**\n\n"
    "**撰写要求**：\n"
    "1. 目标长度：约 **{target_length}** tokens。\n"
    "2. 直接输出该步骤的正文内容，**不要**包含序号、标题或标签。\n"
    "3. 确保内容能与上下文逻辑衔接。\n"
    "【开始撰写】：\n"
)


# =============================================================================
# English Prompts (英文提示词) - Protocol
# =============================================================================

# English Base Prompt
base_prompt_en = (
    "[System Directive] You are a logic expert specializing in complex task planning and execution.\n"
    "Your workflow has two strictly separated phases:\n"
    "1. **Planning Phase **: Decompose complex queries into independent parallel sub-tasks.\n"
    "2. **Execution Phase **: Execute a specific step based on the established plan.\n"
    "You must strictly adhere to the instructions for the current phase.\n"
    "[User Input]:\n{user_inputs}\n\n"
)

# English Skeleton Trigger (Skeleton Generation Phase)
skeleton_trigger_en = (
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
    "4. **Branch Title**: The brief title should be as short as possible."
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
    "[Output]:\n"
)

# English Parallel Trigger (Parallel Execution Phase)
parallel_trigger_en = (
    "[Current Task: Execute Specific Step]\n"
    "Overall Plan:\n"
    "{skeleton_context}\n"
    "Your task is to execute Step **{current_id}**:**{current_point}**\n"
    "**Requirements**:\n"
    "1. Target Length: Approx **{target_length}** tokens.\n"
    "2. The output must start with [ANSWER] and please output the body content directly. DO NOT include IDs, titles.\n"
    "3. If you finished the answer of this branch, the answer must end with [END].\n"
    "4. Ensure logical flow with the [User Input].\n"
    "[ANSWER]"
)


# =============================================================================
# Vicuna Chat Template (保留兼容性)
# =============================================================================

vicuna_chat_template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    "USER: {prompt}\n"
    "ASSISTANT:"
)
