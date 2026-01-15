#!/usr/bin/env python
# coding=utf-8
"""
测试 SpecSoT 模块导入
"""

print("Testing SpecSoT module imports...")

try:
    from SpecSoT import SpecSoTModel
    print("✓ SpecSoTModel imported successfully")
except Exception as e:
    print(f"✗ Failed to import SpecSoTModel: {e}")

try:
    from SpecSoT import EagleLayer
    print("✓ EagleLayer imported successfully")
except Exception as e:
    print(f"✗ Failed to import EagleLayer: {e}")

try:
    from SpecSoT import SemanticLogitsProcessor
    print("✓ SemanticLogitsProcessor imported successfully")
except Exception as e:
    print(f"✗ Failed to import SemanticLogitsProcessor: {e}")

try:
    from SpecSoT import (
        prepare_logits_processor,
        initialize_tree_single,
        initialize_tree_parallel,
        evaluate_posterior,
        update_inference_inputs,
    )
    print("✓ Utility functions imported successfully")
except Exception as e:
    print(f"✗ Failed to import utility functions: {e}")

try:
    from SpecSoT import base_prompt, skeleton_trigger_zh, parallel_trigger_zh
    print("✓ Prompts imported successfully")
except Exception as e:
    print(f"✗ Failed to import prompts: {e}")

try:
    from SpecSoT import EConfig
    print("✓ EConfig imported successfully")
except Exception as e:
    print(f"✗ Failed to import EConfig: {e}")

print("\nAll imports completed!")
