# coding=utf-8
"""
V2 Protocol Test Script

测试 V2 协议的各个组件：
1. Tokenizer 兼容性测试
2. Parser 功能测试
3. FSM Logits Processor 测试
"""

import re
import sys
from typing import Tuple

# 添加项目路径
sys.path.insert(0, '/data/home/chenyu/Coding/SD+SoT/SpecSoT')


def test_parser():
    """测试 V2 Parser"""
    from SpecSoT.utils import parse_skeleton_output
    
    print("=" * 60)
    print("测试 V2 Parser")
    print("=" * 60)
    
    # 测试用例 1: 直接回答模式
    test_direct = """[DIRECT]
这是一个简单的问题，答案是42。
"""
    mode, content = parse_skeleton_output(test_direct)
    print(f"\n测试 1 (直接回答):")
    print(f"  Mode: {mode}")
    print(f"  Content: {content[:50]}...")
    assert mode == "direct", f"Expected 'direct', got '{mode}'"
    print("  ✓ PASSED")
    
    # 测试用例 2: 规划模式
    test_plan = """[PLAN]
1. <200> <None> [-] 分析问题背景
2. <150> <Search> [-] 搜索相关资料
3. <300> <None> [-] 总结并给出建议
[END]
"""
    mode, content = parse_skeleton_output(test_plan)
    print(f"\n测试 2 (规划模式):")
    print(f"  Mode: {mode}")
    print(f"  Tasks: {len(content)} tasks")
    for task in content:
        print(f"    - ID={task['id']}, Len={task['length']}, Tool={task['tool']}, Title={task['title']}")
    assert mode == "plan", f"Expected 'plan', got '{mode}'"
    assert len(content) == 3, f"Expected 3 tasks, got {len(content)}"
    assert content[0]['id'] == 1
    assert content[0]['length'] == 200
    assert content[0]['tool'] is None
    assert content[1]['tool'] == "Search"
    print("  ✓ PASSED")
    
    # 测试用例 3: 中文圆括号格式
    test_plan_cn = """[PLAN]
1。（200）（None）【-】技术维度分析
2。（150）（Search）【-】市场调研
[END]
"""
    mode, content = parse_skeleton_output(test_plan_cn)
    print(f"\n测试 3 (中文格式):")
    print(f"  Mode: {mode}")
    if mode == "plan":
        print(f"  Tasks: {len(content)} tasks")
        for task in content:
            print(f"    - ID={task['id']}, Title={task['title']}")
        print("  ✓ PASSED")
    else:
        print(f"  Warning: Chinese format not fully supported, got {mode}")
    
    # 测试用例 4: 无标签直接回答
    test_no_tag = "这是一个没有任何标签的回答。"
    mode, content = parse_skeleton_output(test_no_tag)
    print(f"\n测试 4 (无标签):")
    print(f"  Mode: {mode}")
    assert mode == "direct", f"Expected 'direct', got '{mode}'"
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("Parser 测试完成!")
    print("=" * 60)


def test_tokenizer_compatibility():
    """测试 Tokenizer 对关键符号的编码"""
    print("\n" + "=" * 60)
    print("测试 Tokenizer 兼容性")
    print("=" * 60)
    
    # 需要测试的符号
    key_symbols = {
        "数字 0-9": "0123456789",
        "点号 .": ".",
        "小于号 <": "<",
        "大于号 >": ">",
        "左方括号 [": "[",
        "右方括号 ]": "]",
        "减号 -": "-",
        "换行符 \\n": "\n",
        "空格": " ",
        "[PLAN] 标记": "[PLAN]",
        "[END] 标记": "[END]",
        "[DIRECT] 标记": "[DIRECT]",
        "完整行示例": "1. <200> <None> [-] 测试标题",
    }
    
    # 尝试加载不同的 tokenizer
    tokenizer_paths = [
        "/data/home/chenyu/Coding/SD+SoT/models/Llama-3.1-8B-Instruct",
        "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B",
        "/data/home/chenyu/Coding/SD+SoT/models/Qwen2.5-7B-Instruct",
    ]
    
    from transformers import AutoTokenizer
    
    for path in tokenizer_paths:
        try:
            print(f"\n测试 Tokenizer: {path.split('/')[-1]}")
            print("-" * 40)
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
            
            all_passed = True
            for name, symbol in key_symbols.items():
                try:
                    ids = tokenizer.encode(symbol, add_special_tokens=False)
                    decoded = tokenizer.decode(ids)
                    # 检查是否可以正确往返编码
                    match = decoded.strip() == symbol.strip()
                    status = "✓" if match else "⚠"
                    print(f"  {status} {name}: {symbol!r} -> {ids} -> {decoded!r}")
                    if not match:
                        all_passed = False
                except Exception as e:
                    print(f"  ✗ {name}: Error - {e}")
                    all_passed = False
            
            if all_passed:
                print(f"  ✓ 所有符号测试通过!")
            else:
                print(f"  ⚠ 部分符号存在问题，请检查")
                
        except Exception as e:
            print(f"  无法加载 tokenizer: {e}")
    
    print("\n" + "=" * 60)
    print("Tokenizer 兼容性测试完成!")
    print("=" * 60)


def test_prompts():
    """测试 V2 Prompts 模板"""
    print("\n" + "=" * 60)
    print("测试 V2 Prompts 模板")
    print("=" * 60)
    
    from SpecSoT.prompts import (
        base_prompt_zh, skeleton_trigger_zh, parallel_trigger_zh,
        base_prompt_en, skeleton_trigger_en, parallel_trigger_en,
    )
    
    # 测试中文 prompts
    test_input = "分析人工智能的发展趋势"
    
    print("\n中文 Base Prompt:")
    base = base_prompt_zh.format(user_inputs=test_input)
    print(base[:200] + "...")
    
    print("\n中文 Skeleton Trigger:")
    skeleton = skeleton_trigger_zh
    print(skeleton[:300] + "...")
    
    # 检查关键标记
    assert "[DIRECT]" in skeleton, "skeleton_trigger_zh 应包含 [DIRECT]"
    assert "[PLAN]" in skeleton, "skeleton_trigger_zh 应包含 [PLAN]"
    assert "[END]" in skeleton, "skeleton_trigger_zh 应包含 [END]"
    print("  ✓ 中文 prompts 包含所有关键标记")
    
    print("\n英文 Base Prompt:")
    base_en = base_prompt_en.format(user_inputs=test_input)
    print(base_en[:200] + "...")
    
    print("\n英文 Skeleton Trigger:")
    print(skeleton_trigger_en[:300] + "...")
    
    # 检查英文关键标记
    assert "[DIRECT]" in skeleton_trigger_en, "skeleton_trigger_en 应包含 [DIRECT]"
    assert "[PLAN]" in skeleton_trigger_en, "skeleton_trigger_en 应包含 [PLAN]"
    assert "[END]" in skeleton_trigger_en, "skeleton_trigger_en 应包含 [END]"
    print("  ✓ 英文 prompts 包含所有关键标记")
    
    # 测试 parallel trigger
    print("\n中文 Parallel Trigger 模板:")
    try:
        parallel = parallel_trigger_zh.format(
            skeleton_context="[PLAN]\n1. <200> <None> [-] 测试\n[END]",
            current_id=1,
            current_point="测试分支",
            target_length=200,
        )
        print(parallel[:300] + "...")
        print("  ✓ Parallel trigger 格式化成功")
    except KeyError as e:
        print(f"  ✗ Parallel trigger 缺少占位符: {e}")
    
    print("\n" + "=" * 60)
    print("Prompts 测试完成!")
    print("=" * 60)


def test_fsm_state():
    """测试 FSM 状态机逻辑"""
    print("\n" + "=" * 60)
    print("测试 FSM 状态机")
    print("=" * 60)
    
    from SpecSoT.logits_processor import FSMState, SemanticLogitsProcessor
    
    # 测试状态枚举
    print("\nFSM 状态列表:")
    for state in FSMState:
        print(f"  {state.name} = {state.value}")
    
    # 如果有 tokenizer，测试状态确定逻辑
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B",
            use_fast=False
        )
        
        processor = SemanticLogitsProcessor(tokenizer, prefix_len=10, enforce_format=False)
        
        # 测试状态确定
        test_cases = [
            ("", FSMState.LINE_START),
            ("1", FSMState.AFTER_ID),
            ("1.", FSMState.AFTER_DOT),
            ("1. <", FSMState.LEN_OPEN),
            ("1. <200", FSMState.LEN_VAL),
            ("1. <200>", FSMState.LEN_CLOSE),
            ("1. <200> <", FSMState.TOOL_OPEN),
            ("1. <200> <None", FSMState.TOOL_VAL),
            ("1. <200> <None>", FSMState.TOOL_CLOSE),
            ("1. <200> <None> [", FSMState.DEPS_OPEN),
            ("1. <200> <None> [-", FSMState.DEPS_VAL),
            ("1. <200> <None> [-]", FSMState.DEPS_CLOSE),
            ("1. <200> <None> [-] 标题", FSMState.CONTENT),
        ]
        
        print("\n状态确定测试:")
        for line, expected in test_cases:
            actual = processor._determine_state(line)
            status = "✓" if actual == expected else "✗"
            print(f"  {status} '{line}' -> {actual.name} (expected: {expected.name})")
        
        print("  ✓ FSM 状态确定测试完成")
        
    except Exception as e:
        print(f"  跳过 FSM 实际测试: {e}")
    
    print("\n" + "=" * 60)
    print("FSM 测试完成!")
    print("=" * 60)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SpecSoT V2 Protocol 测试套件")
    print("=" * 60)
    
    try:
        test_parser()
        test_prompts()
        test_fsm_state()
        test_tokenizer_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(run_all_tests())
