# coding=utf-8
"""
FSM Logits Processor 单元测试

测试新的 SemanticLogitsProcessor 的状态机逻辑
"""

import sys
sys.path.insert(0, '/data/home/chenyu/Coding/SD+SoT/SpecSoT')

from SpecSoT.logits_processor import SemanticLogitsProcessor, FSMState


class MockTokenizer:
    """模拟 tokenizer 用于测试"""
    
    def __init__(self):
        # 模拟 token 到 ID 的映射
        self.token_to_id = {
            "[": 91,
            "]": 93,
            "<": 60,
            ">": 62,
            ".": 46,
            "-": 45,
            " ": 32,
            "\n": 10,
            "0": 48, "1": 49, "2": 50, "3": 51, "4": 52,
            "5": 53, "6": 54, "7": 55, "8": 56, "9": 57,
            "P": 80, "L": 76, "A": 65, "N": 78,
            "D": 68, "I": 73, "R": 82, "E": 69, "C": 67, "T": 84,
            "PLAN": 1000,
            "DIRECT": 1001,
            "END": 1002,
        }
        # 单字母映射
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.token_to_id[c] = 97 + i
        for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            self.token_to_id[c] = 65 + i
    
    def encode(self, text, add_special_tokens=False):
        if text in self.token_to_id:
            return [self.token_to_id[text]]
        # 对于多字符，返回单个 token（模拟子词分词）
        if text == "PLAN":
            return [1000]
        if text == "DIRECT":
            return [1001]
        if text == "END":
            return [1002]
        # 否则按字符返回
        return [self.token_to_id.get(c, 0) for c in text if c in self.token_to_id]
    
    def decode(self, ids, skip_special_tokens=False):
        # 简单的反向映射（用于测试，实际不会用到）
        return ""


def test_fsm_states():
    """测试 FSM 状态转换逻辑"""
    tokenizer = MockTokenizer()
    processor = SemanticLogitsProcessor(tokenizer, prefix_len=100, enforce_format=True)
    
    # =========================================================================
    # 测试 Header 阶段
    # =========================================================================
    
    print("=" * 60)
    print("测试 Header 阶段")
    print("=" * 60)
    
    # 测试空文本
    state = processor._get_state_from_text("")
    assert state == FSMState.HEADER_LBRACKET, f"空文本应该返回 HEADER_LBRACKET, 实际: {state.name}"
    print(f"✓ 空文本 -> {state.name}")
    
    # 测试 [
    state = processor._get_state_from_text("[")
    assert state == FSMState.HEADER_TYPE, f"'[' 应该返回 HEADER_TYPE, 实际: {state.name}"
    print(f"✓ '[' -> {state.name}")
    
    # 测试 [PLAN
    state = processor._get_state_from_text("[PLAN")
    assert state == FSMState.HEADER_RBRACKET, f"'[PLAN' 应该返回 HEADER_RBRACKET, 实际: {state.name}"
    print(f"✓ '[PLAN' -> {state.name}")
    
    # 测试 [DIRECT
    state = processor._get_state_from_text("[DIRECT")
    assert state == FSMState.HEADER_RBRACKET, f"'[DIRECT' 应该返回 HEADER_RBRACKET, 实际: {state.name}"
    print(f"✓ '[DIRECT' -> {state.name}")
    
    # 测试 [PLAN]
    state = processor._get_state_from_text("[PLAN]")
    assert state == FSMState.HEADER_NEWLINE, f"'[PLAN]' 应该返回 HEADER_NEWLINE, 实际: {state.name}"
    print(f"✓ '[PLAN]' -> {state.name}")
    
    # 测试 [DIRECT]
    state = processor._get_state_from_text("[DIRECT]")
    assert state == FSMState.HEADER_NEWLINE, f"'[DIRECT]' 应该返回 HEADER_NEWLINE, 实际: {state.name}"
    print(f"✓ '[DIRECT]' -> {state.name}")
    
    # =========================================================================
    # 测试 DIRECT 模式
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("测试 DIRECT 模式")
    print("=" * 60)
    
    processor.reset()
    
    # 测试 DIRECT 内容（无约束）
    state = processor._get_state_from_text("[DIRECT]\n这是直接回答的内容")
    assert state == FSMState.DIRECT_CONTENT, f"DIRECT 内容应该返回 DIRECT_CONTENT, 实际: {state.name}"
    print(f"✓ '[DIRECT]\\n这是内容' -> {state.name}")
    
    # 测试 DIRECT 完成
    state = processor._get_state_from_text("[DIRECT]\n这是直接回答的内容\n[END]")
    assert state == FSMState.FINISHED, f"DIRECT 完成应该返回 FINISHED, 实际: {state.name}"
    print(f"✓ '[DIRECT]...\\n[END]' -> {state.name}")
    
    # =========================================================================
    # 测试 PLAN 模式 - 分支行解析
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("测试 PLAN 模式 - 分支行解析")
    print("=" * 60)
    
    processor.reset()
    
    # 测试新行开始
    state = processor._get_state_from_text("[PLAN]\n")
    assert state == FSMState.LINE_START, f"PLAN 新行应该返回 LINE_START, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n' -> {state.name}")
    
    # 测试 ID
    state = processor._get_state_from_text("[PLAN]\n1")
    assert state == FSMState.AFTER_ID, f"'1' 应该返回 AFTER_ID, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1' -> {state.name}")
    
    # 测试多位数 ID
    state = processor._get_state_from_text("[PLAN]\n12")
    assert state == FSMState.AFTER_ID, f"'12' 应该返回 AFTER_ID, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n12' -> {state.name}")
    
    # 测试点号后
    state = processor._get_state_from_text("[PLAN]\n1.")
    assert state == FSMState.AFTER_DOT, f"'1.' 应该返回 AFTER_DOT, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1.' -> {state.name}")
    
    # 测试点号后空格
    state = processor._get_state_from_text("[PLAN]\n1. ")
    assert state == FSMState.AFTER_DOT_SPACE, f"'1. ' 应该返回 AFTER_DOT_SPACE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. ' -> {state.name}")
    
    # 测试 <Length 开始
    state = processor._get_state_from_text("[PLAN]\n1. <")
    assert state == FSMState.LEN_OPEN, f"'1. <' 应该返回 LEN_OPEN, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <' -> {state.name}")
    
    # 测试 <Length 值
    state = processor._get_state_from_text("[PLAN]\n1. <200")
    assert state == FSMState.LEN_VAL, f"'1. <200' 应该返回 LEN_VAL, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200' -> {state.name}")
    
    # 测试 <Length> 关闭后
    state = processor._get_state_from_text("[PLAN]\n1. <200>")
    assert state == FSMState.LEN_CLOSE, f"'1. <200>' 应该返回 LEN_CLOSE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200>' -> {state.name}")
    
    # 测试 <Length> 后空格
    state = processor._get_state_from_text("[PLAN]\n1. <200> ")
    assert state == FSMState.AFTER_LEN_SPACE, f"'1. <200> ' 应该返回 AFTER_LEN_SPACE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> ' -> {state.name}")
    
    # 测试 <Tool 开始
    state = processor._get_state_from_text("[PLAN]\n1. <200> <")
    assert state == FSMState.TOOL_OPEN, f"'1. <200> <' 应该返回 TOOL_OPEN, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <' -> {state.name}")
    
    # 测试 <Tool 值
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None")
    assert state == FSMState.TOOL_VAL, f"'1. <200> <None' 应该返回 TOOL_VAL, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None' -> {state.name}")
    
    # 测试 <Tool 值 - Search
    state = processor._get_state_from_text("[PLAN]\n1. <200> <Search")
    assert state == FSMState.TOOL_VAL, f"'1. <200> <Search' 应该返回 TOOL_VAL, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <Search' -> {state.name}")
    
    # 测试 <Tool> 关闭后
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None>")
    assert state == FSMState.TOOL_CLOSE, f"'1. <200> <None>' 应该返回 TOOL_CLOSE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None>' -> {state.name}")
    
    # 测试 <Tool> 后空格
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> ")
    assert state == FSMState.AFTER_TOOL_SPACE, f"'1. <200> <None> ' 应该返回 AFTER_TOOL_SPACE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> ' -> {state.name}")
    
    # 测试 [Deps 开始
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [")
    assert state == FSMState.DEPS_OPEN, f"'[' 应该返回 DEPS_OPEN, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> [' -> {state.name}")
    
    # 测试 [Deps 值
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-")
    assert state == FSMState.DEPS_VAL, f"'[-' 应该返回 DEPS_VAL, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> [-' -> {state.name}")
    
    # 测试 [Deps] 关闭后
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-]")
    assert state == FSMState.DEPS_CLOSE, f"'[-]' 应该返回 DEPS_CLOSE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> [-]' -> {state.name}")
    
    # 测试 [Deps] 后空格
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] ")
    assert state == FSMState.AFTER_DEPS_SPACE, f"'[-] ' 应该返回 AFTER_DEPS_SPACE, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> [-] ' -> {state.name}")
    
    # 测试标题内容
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 这是标题")
    assert state == FSMState.CONTENT, f"标题应该返回 CONTENT, 实际: {state.name}"
    print(f"✓ '[PLAN]\\n1. <200> <None> [-] 这是标题' -> {state.name}")
    
    # =========================================================================
    # 测试 [END] 检测
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("测试 [END] 检测")
    print("=" * 60)
    
    processor.reset()
    
    # 测试完整的第一行后换行
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 标题\n")
    assert state == FSMState.LINE_START, f"换行后应该返回 LINE_START, 实际: {state.name}"
    print(f"✓ '...\\n' (第二行开始) -> {state.name}")
    
    # 测试 [ 开始 END
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 标题\n[")
    assert state == FSMState.END_TYPE, f"'[' 应该返回 END_TYPE, 实际: {state.name}"
    print(f"✓ '...\\n[' -> {state.name}")
    
    # 测试 [END
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 标题\n[END")
    assert state == FSMState.END_RBRACKET, f"'[END' 应该返回 END_RBRACKET, 实际: {state.name}"
    print(f"✓ '...\\n[END' -> {state.name}")
    
    # 测试 [END]
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 标题\n[END]")
    assert state == FSMState.FINISHED, f"'[END]' 应该返回 FINISHED, 实际: {state.name}"
    print(f"✓ '...\\n[END]' -> {state.name}")
    
    # =========================================================================
    # 测试多行 PLAN
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("测试多行 PLAN")
    print("=" * 60)
    
    processor.reset()
    
    # 完整的多行 PLAN
    full_plan = """[PLAN]
1. <200> <Search> [-] 搜索最新的篮球规则
2. <150> <None> [-] 分析投篮动作
3. <300> <None> [-] 总结训练技巧
[END]"""
    
    state = processor._get_state_from_text(full_plan)
    assert state == FSMState.FINISHED, f"完整 PLAN 应该返回 FINISHED, 实际: {state.name}"
    print(f"✓ 完整多行 PLAN -> {state.name}")
    
    # 正在输入第三行
    partial_plan = """[PLAN]
1. <200> <Search> [-] 搜索最新的篮球规则
2. <150> <None> [-] 分析投篮动作
3"""
    
    state = processor._get_state_from_text(partial_plan)
    assert state == FSMState.AFTER_ID, f"'3' 应该返回 AFTER_ID, 实际: {state.name}"
    print(f"✓ 第三行 '3' -> {state.name}")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


def test_edge_cases():
    """测试边界情况"""
    tokenizer = MockTokenizer()
    processor = SemanticLogitsProcessor(tokenizer, prefix_len=100, enforce_format=True)
    
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    # 测试带数字的工具名
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n1. <200> <Tool123")
    assert state == FSMState.TOOL_VAL, f"数字工具名应该返回 TOOL_VAL, 实际: {state.name}"
    print(f"✓ '<Tool123' (数字工具名) -> {state.name}")
    
    # 测试复杂依赖
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [1,2,3")
    assert state == FSMState.DEPS_VAL, f"复杂依赖应该返回 DEPS_VAL, 实际: {state.name}"
    print(f"✓ '[1,2,3' (复杂依赖) -> {state.name}")
    
    # 测试大数字 Length
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n1. <99999")
    assert state == FSMState.LEN_VAL, f"大数字 Length 应该返回 LEN_VAL, 实际: {state.name}"
    print(f"✓ '<99999' (大数字 Length) -> {state.name}")
    
    # 测试空行后新行
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 标题\n\n")
    # 注意：split('\n') 后最后是 ""
    assert state == FSMState.LINE_START, f"空行后应该返回 LINE_START, 实际: {state.name}"
    print(f"✓ '...\\n\\n' (空行) -> {state.name}")
    
    # 测试 DIRECT 中间包含 [ 字符
    processor.reset()
    state = processor._get_state_from_text("[DIRECT]\n这里有一个 [方括号] 内容")
    assert state == FSMState.DIRECT_CONTENT, f"DIRECT 中的方括号应该返回 DIRECT_CONTENT, 实际: {state.name}"
    print(f"✓ DIRECT 中包含 [方括号] -> {state.name}")
    
    # 测试多位数 ID（如 10, 11, ...）
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n10.")
    assert state == FSMState.AFTER_DOT, f"'10.' 应该返回 AFTER_DOT, 实际: {state.name}"
    print(f"✓ '10.' (多位数 ID) -> {state.name}")
    
    # 测试 Length 为单个数字
    processor.reset()
    state = processor._get_state_from_text("[PLAN]\n1. <5>")
    assert state == FSMState.LEN_CLOSE, f"'<5>' 应该返回 LEN_CLOSE, 实际: {state.name}"
    print(f"✓ '<5>' (单数字 Length) -> {state.name}")
    
    print("\n✓ 所有边界测试通过!")


def test_real_tokenizer_if_available():
    """使用真实 tokenizer 进行测试（如果可用）"""
    try:
        from transformers import AutoTokenizer
        print("\n" + "=" * 60)
        print("测试真实 Tokenizer")
        print("=" * 60)
        
        # 尝试加载 Qwen tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/home/chenyu/Coding/SD+SoT/models/Qwen2.5-7B-Instruct",
            use_fast=False,
            trust_remote_code=True
        )
        
        processor = SemanticLogitsProcessor(tokenizer, prefix_len=100, enforce_format=True)
        
        # 打印关键 token IDs
        print(f"Token IDs: {processor.token_ids}")
        print(f"Digit token count: {len(processor.digit_token_ids)}")
        print(f"Letter token count: {len(processor.letter_token_ids)}")
        print(f"PLAN tokens: {processor.plan_tokens}")
        print(f"DIRECT tokens: {processor.direct_tokens}")
        print(f"END tokens: {processor.end_tokens}")
        
        # 验证状态机
        state = processor._get_state_from_text("")
        print(f"\n空文本状态: {state.name}")
        
        state = processor._get_state_from_text("[PLAN]\n1. <200> <None> [-] 测试")
        print(f"完整行状态: {state.name}")
        
        print("\n✓ 真实 Tokenizer 测试完成!")
        
    except Exception as e:
        print(f"\n⚠ 跳过真实 Tokenizer 测试: {e}")


if __name__ == "__main__":
    test_fsm_states()
    test_edge_cases()
    test_real_tokenizer_if_available()
