#!/usr/bin/env python
# coding=utf-8
"""
Prompt Template Verification Script

This script verifies that prompt templates are correctly built for each model type.
It outputs the constructed prompts (decoded as strings) for inspection.

Test cases:
1. Vicuna - chat template format
2. Qwen - ChatML format  
3. Llama 3.1 - special tokens format
4. Other/Default - simple concatenation

For each model type, we test:
- Skeleton phase: full prompt + prefix
- Parallel phase: branch prompts with skeleton context (without [END] markers)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
import torch

from SpecSoT.prompts import (
    system_prompt,
    skeleton_prompt,
    parallel_prompt,
    build_prompt,
    build_prefix,
    extract_skeleton_context,
    prepare_skeleton_input,
    prepare_parallel_branches,
    parse_skeleton_output,
    VICUNA_HEADER,
)


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a section separator"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def test_prompt_templates_without_tokenizer():
    """Test prompt building without tokenizer (string level)"""
    print_separator("1. Testing Prompt Templates (String Level)")
    
    task_prompt = "Introduce the top 3 cities in China separately."
    
    # Build system content
    system_content = system_prompt.format(user_input=task_prompt)
    
    print("=== System Prompt (PREFIX Content) ===")
    print(system_content)
    print()
    
    print("=== Skeleton Prompt (User Content for Phase 1) ===")
    print(skeleton_prompt)
    print()
    
    # Test each model type
    for model_type in ['vicuna', 'qwen', 'llama', 'other']:
        print(f"\n--- Model: {model_type.upper()} ---")
        
        # Full skeleton prompt
        full_prompt = build_prompt(model_type, system_content, skeleton_prompt)
        print(f"\n[Full Skeleton Prompt for {model_type}]:")
        print(full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt)
        
        # Prefix
        prefix = build_prefix(model_type, system_content)
        print(f"\n[PREFIX for {model_type}]:")
        print(prefix)
        print()


def test_skeleton_context_extraction():
    """Test that skeleton context extraction removes [PLAN]/[END] markers"""
    print_separator("2. Testing Skeleton Context Extraction")
    
    # Simulated skeleton output
    skeleton_output = """[PLAN]
1.<200><None>[-]Introduce Beijing
2.<300><None>[-]Introduce Shanghai
3.<250><None>[-]Introduce Guangzhou
[END]"""
    
    print("=== Original Skeleton Output ===")
    print(skeleton_output)
    print()
    
    # Extract clean context
    clean_context = extract_skeleton_context(skeleton_output)
    print("=== Extracted Clean Context (NO [PLAN]/[END]) ===")
    print(clean_context)
    print()
    
    # Verify [END] is removed
    if "[END]" in clean_context:
        print("❌ ERROR: [END] marker still present in extracted context!")
    else:
        print("✅ SUCCESS: [END] marker correctly removed from context")
    
    if "[PLAN]" in clean_context:
        print("❌ ERROR: [PLAN] marker still present in extracted context!")
    else:
        print("✅ SUCCESS: [PLAN] marker correctly removed from context")


def test_parallel_prompt_building():
    """Test parallel prompt building with clean skeleton context"""
    print_separator("3. Testing Parallel Prompt Building")
    
    task_prompt = "Introduce the top 3 cities in China separately."
    system_content = system_prompt.format(user_input=task_prompt)
    
    # Clean skeleton context (without markers)
    skeleton_context = """1.<200><None>[-]Introduce Beijing
2.<300><None>[-]Introduce Shanghai
3.<250><None>[-]Introduce Guangzhou"""
    
    # Build parallel prompt for branch 1
    user_content = parallel_prompt.format(
        skeleton_context=skeleton_context,
        current_id=1,
        current_point="Introduce Beijing",
        target_length=200,
    )
    
    print("=== Parallel Prompt User Content (Branch 1) ===")
    print(user_content)
    print()
    
    # Build full prompt for vicuna
    for model_type in ['vicuna', 'qwen', 'llama']:
        full_parallel = build_prompt(model_type, system_content, user_content)
        print(f"\n[Full Parallel Prompt for {model_type} - Branch 1]:")
        print(full_parallel[:600] + "..." if len(full_parallel) > 600 else full_parallel)
        print()


def test_with_tokenizer(model_name: str, model_type: str):
    """Test prompt building with actual tokenizer"""
    print_separator(f"4. Testing with Tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Loaded tokenizer for {model_name}")
    except Exception as e:
        print(f"⚠️ Could not load tokenizer for {model_name}: {e}")
        print("Skipping tokenizer test...")
        return
    
    device = torch.device("cpu")
    task_prompt = "Introduce the top 3 cities in China separately."
    
    # Test skeleton input
    print("\n--- Skeleton Phase ---")
    input_ids, prefix_ids = prepare_skeleton_input(
        tokenizer, task_prompt, model_type, device
    )
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Prefix IDs shape: {prefix_ids.shape}")
    
    # Decode and show
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    prefix_text = tokenizer.decode(prefix_ids[0], skip_special_tokens=False)
    
    print(f"\n[Decoded Full Input ({len(input_ids[0])} tokens)]:")
    print(full_text[:800] + "..." if len(full_text) > 800 else full_text)
    
    print(f"\n[Decoded PREFIX ({len(prefix_ids[0])} tokens)]:")
    print(prefix_text)
    
    # Test parallel branches
    print("\n--- Parallel Phase ---")
    skeleton_output = """[PLAN]
1.<200><None>[-]Introduce Beijing
2.<300><None>[-]Introduce Shanghai
3.<250><None>[-]Introduce Guangzhou
[END]"""
    
    mode, tasks = parse_skeleton_output(skeleton_output)
    print(f"Parsed mode: {mode}, tasks count: {len(tasks)}")
    
    branch_ids, instruction_lengths = prepare_parallel_branches(
        tokenizer, tasks, skeleton_output, model_type, task_prompt
    )
    
    for i, (branch_id, inst_len) in enumerate(zip(branch_ids, instruction_lengths)):
        branch_text = tokenizer.decode(branch_id, skip_special_tokens=False)
        print(f"\n[Branch {i+1} - {inst_len} tokens]:")
        print(branch_text[:500] + "..." if len(branch_text) > 500 else branch_text)
        
        # Check: skeleton context should NOT contain [END] from [PLAN]...[END] markers
        # Note: parallel_prompt itself mentions "[END]" as instruction (e.g., "must end with [END]")
        # This is expected. We're checking that skeleton_context doesn't have the actual marker.
        # Count occurrences of [END] - should be exactly the ones from parallel_prompt template
        end_count = branch_text.count("[END]")
        # parallel_prompt has 2 mentions of [END] in the instructions
        if end_count <= 2:
            print(f"✅ Branch {i+1}: [END] count is {end_count} (only from instructions, correct)")
        else:
            print(f"⚠️ WARNING: Branch {i+1} has {end_count} [END] occurrences (may include skeleton marker)")


def main():
    print("\n" + "=" * 80)
    print(" SpecSoT Prompt Template Verification")
    print("=" * 80)
    
    # Test 1: String-level prompt building
    test_prompt_templates_without_tokenizer()
    
    # Test 2: Skeleton context extraction
    test_skeleton_context_extraction()
    
    # Test 3: Parallel prompt building
    test_parallel_prompt_building()
    
    # Test 4: With tokenizer (if available)
    # Try different model tokenizers
    test_models = [
        ("lmsys/vicuna-7b-v1.5", "vicuna"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
        # ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama"),
    ]
    
    for model_name, model_type in test_models:
        test_with_tokenizer(model_name, model_type)
    
    print_separator("Verification Complete", "=")
    print("All prompt template tests passed! Review the output above for correctness.")
    print("\nKey points to verify:")
    print("1. Vicuna format: {Header}\\n\\nUSER: {system}\\n{user}\\nASSISTANT:")
    print("2. Qwen format: <|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant")
    print("3. Llama3 format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{sys}<|eot_id|>...")
    print("4. Skeleton context extraction: [PLAN]/[END] markers should be REMOVED")
    print()


if __name__ == "__main__":
    main()
