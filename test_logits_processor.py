#!/usr/bin/env python3
# coding=utf-8
"""
测试 SemanticLogitsProcessor 在不同分词器上的兼容性

测试内容：
1. VocabScanner 词表扫描功能
2. FSM 状态判定逻辑
3. 多位数长度支持 (如 <127>, <1500>)
4. 紧凑格式解析 (无空格)

使用方法:
    python test_logits_processor.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from SpecSoT.logits_processor import SemanticLogitsProcessor, VocabScanner, FSMState
from SpecSoT.utils import parse_skeleton_output


def get_state_from_text(processor: SemanticLogitsProcessor, text: str) -> FSMState:
    """
    辅助函数：通过增量状态机获取给定文本的最终状态
    
    这是为了测试新的增量状态机架构，模拟从头到尾处理文本后的状态
    """
    processor.reset()
    processor._update_state_incremental(text)
    return processor._current_state


def test_vocab_scanner(tokenizer, name: str):
    """测试词表扫描器"""
    print(f"\n{'='*60}")
    print(f"测试 VocabScanner: {name}")
    print(f"{'='*60}")
    
    scanner = VocabScanner(tokenizer)
    
    # 检查特殊字符类别
    categories = {
        'digit_tokens': scanner.digit_tokens,
        'letter_tokens': scanner.letter_tokens,
        'lt_tokens': scanner.lt_tokens,
        'gt_tokens': scanner.gt_tokens,
        'lbracket_tokens': scanner.lbracket_tokens,
        'rbracket_tokens': scanner.rbracket_tokens,
        'dot_tokens': scanner.dot_tokens,
        'dash_tokens': scanner.dash_tokens,
        'newline_tokens': scanner.newline_tokens,
    }
    
    print("\n字符类别 Token 数量:")
    for cat_name, tokens in categories.items():
        print(f"  {cat_name}: {len(tokens)} tokens")
        # 显示前 5 个
        if tokens:
            samples = list(tokens)[:5]
            decoded = [repr(scanner.decode_token(t)) for t in samples]
            print(f"    示例: {decoded}")
    
    # 测试关键字字符
    print("\n关键字字符测试:")
    for char in "PLANDIRECT":
        tokens = scanner.get_tokens_matching_char(char)
        if tokens:
            samples = list(tokens)[:3]
            decoded = [repr(scanner.decode_token(t)) for t in samples]
            print(f"  '{char}': {len(tokens)} tokens, 示例: {decoded}")
    
    return scanner


def test_fsm_states(processor: SemanticLogitsProcessor):
    """测试 FSM 状态判定"""
    print(f"\n{'='*60}")
    print("测试 FSM 状态判定")
    print(f"{'='*60}")
    
    test_cases = [
        # Header 阶段
        ("", FSMState.HEADER_LBRACKET),
        ("[", FSMState.HEADER_KEYWORD),
        ("[P", FSMState.HEADER_KEYWORD),
        ("[PL", FSMState.HEADER_KEYWORD),
        ("[PLA", FSMState.HEADER_KEYWORD),
        # 注意：在增量状态机中，[PLAN 仍然是 HEADER_KEYWORD，
        # 直到遇到 ] 才转换。约束应用时会检查 buffer 是否完整并允许 ]
        ("[PLAN", FSMState.HEADER_KEYWORD),
        ("[PLAN]", FSMState.HEADER_NEWLINE),
        ("[D", FSMState.HEADER_KEYWORD),
        ("[DIR", FSMState.HEADER_KEYWORD),
        ("[DIRECT", FSMState.HEADER_KEYWORD),
        ("[DIRECT]", FSMState.HEADER_NEWLINE),
        
        # PLAN 模式
        ("[PLAN]\n", FSMState.LINE_START),
        ("[PLAN]\n1", FSMState.AFTER_ID),
        ("[PLAN]\n1.", FSMState.AFTER_DOT),
        ("[PLAN]\n1.<", FSMState.LEN_OPEN),
        ("[PLAN]\n1.<2", FSMState.LEN_VAL),
        ("[PLAN]\n1.<20", FSMState.LEN_VAL),
        ("[PLAN]\n1.<200", FSMState.LEN_VAL),
        ("[PLAN]\n1.<200>", FSMState.AFTER_LEN_CLOSE),
        ("[PLAN]\n1.<200><", FSMState.TOOL_OPEN),
        ("[PLAN]\n1.<200><N", FSMState.TOOL_VAL),
        ("[PLAN]\n1.<200><None", FSMState.TOOL_VAL),
        ("[PLAN]\n1.<200><None>", FSMState.AFTER_TOOL_CLOSE),
        ("[PLAN]\n1.<200><None>[", FSMState.DEPS_OPEN),
        ("[PLAN]\n1.<200><None>[-", FSMState.DEPS_VAL),
        ("[PLAN]\n1.<200><None>[-]", FSMState.AFTER_DEPS_CLOSE),
        ("[PLAN]\n1.<200><None>[-]Hello", FSMState.CONTENT),
        ("[PLAN]\n1.<200><None>[-]Hello\n", FSMState.LINE_START),
        ("[PLAN]\n1.<200><None>[-]Hello\n[", FSMState.END_KEYWORD),
        ("[PLAN]\n1.<200><None>[-]Hello\n[E", FSMState.END_KEYWORD),
        ("[PLAN]\n1.<200><None>[-]Hello\n[EN", FSMState.END_KEYWORD),
        # [END 也保持 END_KEYWORD，直到遇到 ] 才完成
        ("[PLAN]\n1.<200><None>[-]Hello\n[END", FSMState.END_KEYWORD),
        ("[PLAN]\n1.<200><None>[-]Hello\n[END]", FSMState.FINISHED),
        
        # 多位数长度测试
        ("[PLAN]\n1.<1500>", FSMState.AFTER_LEN_CLOSE),
        ("[PLAN]\n1.<127><Search>[-]Title", FSMState.CONTENT),
        
        # DIRECT 模式
        ("[DIRECT]\n", FSMState.DIRECT_CONTENT),
        ("[DIRECT]\nHello world", FSMState.DIRECT_CONTENT),
        ("[DIRECT]\nHello\n[END]", FSMState.FINISHED),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_state in test_cases:
        processor.reset()
        actual_state = get_state_from_text(processor, text)
        
        if actual_state == expected_state:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"
        
        # 只显示失败的用例
        if status == "✗":
            print(f"  {status} '{text[:40]:<40}' -> 期望: {expected_state.name}, 实际: {actual_state.name}")
    
    print(f"\n状态判定测试: {passed}/{passed+failed} 通过")
    return failed == 0


def test_skeleton_parsing():
    """测试骨架解析"""
    print(f"\n{'='*60}")
    print("测试骨架解析 (parse_skeleton_output)")
    print(f"{'='*60}")
    
    test_cases = [
        # 紧凑格式
        (
            "[PLAN]\n1.<200><Search>[-]搜索内容\n2.<150><None>[-]分析结果\n[END]",
            "plan",
            2,
        ),
        # 带空格格式
        (
            "[PLAN]\n1. <200> <Search> [-] 搜索内容\n2. <150> <None> [-] 分析结果\n[END]",
            "plan",
            2,
        ),
        # 多位数长度
        (
            "[PLAN]\n1.<1500><None>[-]长任务\n[END]",
            "plan",
            1,
        ),
        # 直接回答
        (
            "[DIRECT]\n这是直接回答\n[END]",
            "direct",
            None,
        ),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_mode, expected_count in test_cases:
        mode, content = parse_skeleton_output(text)
        
        if mode == expected_mode:
            if expected_mode == "plan":
                if len(content) == expected_count:
                    passed += 1
                    print(f"  ✓ {mode} 模式, {len(content)} 个任务")
                    for task in content:
                        print(f"    - ID={task['id']}, Length={task['length']}, Tool={task['tool']}, Title={task['title'][:20]}")
                else:
                    failed += 1
                    print(f"  ✗ 期望 {expected_count} 个任务, 实际 {len(content)}")
            else:
                passed += 1
                print(f"  ✓ {mode} 模式")
        else:
            failed += 1
            print(f"  ✗ 期望 {expected_mode} 模式, 实际 {mode}")
    
    print(f"\n骨架解析测试: {passed}/{passed+failed} 通过")
    return failed == 0


def test_logits_masking(tokenizer, name: str):
    """测试 logits masking"""
    print(f"\n{'='*60}")
    print(f"测试 Logits Masking: {name}")
    print(f"{'='*60}")
    
    processor = SemanticLogitsProcessor(
        tokenizer=tokenizer,
        prefix_len=10,
        enforce_format=True,
    )
    
    vocab_size = tokenizer.vocab_size
    
    # 创建模拟的 input_ids 和 scores
    def make_scores():
        return torch.zeros(1, vocab_size)
    
    def test_constraint(description: str, input_ids: torch.Tensor, expected_category: str):
        scores = make_scores()
        result = processor(input_ids, scores)
        
        # 找出被允许的 token
        allowed = (result[0] != float('-inf')).nonzero().squeeze(-1).tolist()
        if isinstance(allowed, int):
            allowed = [allowed]
        
        # 验证
        vs = processor.vocab_scanner
        category_tokens = getattr(vs, expected_category, set())
        
        # 检查允许的 token 是否在预期类别中
        if allowed:
            in_category = sum(1 for t in allowed if t in category_tokens)
            ratio = in_category / len(allowed) if allowed else 0
            status = "✓" if ratio > 0.5 else "?"  # 至少50%在类别中
            print(f"  {status} {description}: 允许 {len(allowed)} tokens, {in_category} 在 {expected_category} 中 ({ratio:.1%})")
        else:
            print(f"  ? {description}: 无允许的 token")
    
    # 准备 prefix
    prefix = tokenizer.encode("Hello world test", add_special_tokens=False)
    while len(prefix) < 10:
        prefix = prefix + prefix
    prefix = prefix[:10]
    
    # 测试初始状态
    input_ids = torch.tensor([prefix])
    test_constraint("初始状态 -> 期望 [", input_ids, "lbracket_tokens")
    
    # 测试 [P 后
    input_ids = torch.tensor([prefix + tokenizer.encode("[P", add_special_tokens=False)])
    scores = make_scores()
    processor.reset()
    result = processor(input_ids, scores)
    allowed = (result[0] != float('-inf')).nonzero().squeeze(-1).tolist()
    if isinstance(allowed, int):
        allowed = [allowed]
    
    # 检查是否允许 L (PLAN 的下一个字符)
    l_tokens = processor.vocab_scanner.get_tokens_matching_char('L')
    has_l = any(t in l_tokens for t in allowed)
    status = "✓" if has_l else "✗"
    print(f"  {status} '[P' 后 -> 期望包含 'L': 允许 {len(allowed)} tokens, 包含 L: {has_l}")
    
    return True


def main():
    """主测试函数"""
    print("="*60)
    print("SemanticLogitsProcessor 兼容性测试")
    print("="*60)
    
    # 测试可用的模型
    test_models = [
        ("/data/home/chenyu/Coding/SD+SoT/models/vicuna-7b-v1.3", "Vicuna-7B"),
        ("/data/home/chenyu/Coding/SD+SoT/models/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
        ("/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B", "Qwen3-4B"),
    ]
    
    # 先测试骨架解析（不需要模型）
    test_skeleton_parsing()
    
    # 测试每个模型
    for model_path, model_name in test_models:
        if not os.path.exists(model_path):
            print(f"\n跳过 {model_name}: 模型路径不存在")
            continue
        
        try:
            print(f"\n加载 {model_name} tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            
            # 测试词表扫描
            scanner = test_vocab_scanner(tokenizer, model_name)
            
            # 测试 FSM 状态
            processor = SemanticLogitsProcessor(
                tokenizer=tokenizer,
                prefix_len=10,
                enforce_format=True,
            )
            test_fsm_states(processor)
            
            # 测试 logits masking
            test_logits_masking(tokenizer, model_name)
            
        except Exception as e:
            print(f"\n测试 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
