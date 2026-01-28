# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Guidelines
- Env: conda activate sdsot
- Format: Follow PEP8
- Language: Read/Write in Chinese (中文)

## Commands

### Running Inference

```bash
# Baseline Single GPU inference
python run_specsot.py --distributed False

# Multi-GPU distributed inference (3 GPUs)
python run_specsot.py --world_size 3 --gpu_ids 5,6,7 --layer_splits 14,28

```

### Running Inference with different models

```bash
# Baseline Single GPU inference
python run_specsot.py --base_model_path "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B" --eagle_model_path "/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3" --use_eagle3 True

python run_specsot.py --base_model_path "/data/home/chenyu/Coding/SD+SoT/models/Llama-3.1-8B-Instruct" --eagle_model_path "/data/home/chenyu/Coding/SD+SoT/models/EAGLE3-LLaMA3.1-Instruct-8B" --use_eagle3 True

python run_specsot.py --base_model_path "/data/home/chenyu/Coding/SD+SoT/models/vicuna-7b-v1.3" --eagle_model_path "/data/home/chenyu/Coding/SD+SoT/models/EAGLE-Vicuna-7B-v1.3" --use_eagle3 False

# Multi-GPU distributed inference (3 GPUs)
python run_specsot.py --world_size 3 --gpu_ids 5,6,7 --layer_splits 14,28

```

### Key CLI Arguments for run_specsot.py

- `--distributed`: Enable distributed mode (True/False)
- `--world_size`: Number of GPUs/processes
- `--gpu_ids`: Comma-separated GPU IDs (e.g., "0,1,2")
- `--layer_splits`: Layer split points for model partitioning (e.g., "14,28" for 3 devices)
- `--task`: Evaluation task ("planning", "retrieval", "multi-doc-qa")
- `--enable_parallel`: Enable skeleton parallel mode (default: True)
- `--use_semantic_constraint`: Enable FSM semantic constraints (default: True)
- `--base_model_path` / `--eagle_model_path`: Model paths

## Architecture

SpecSoT combines Speculative Decoding with Skeleton-of-Thought for efficient LLM inference.

### Three-Stage Pipeline

1. **Skeleton Generation**: Uses EAGLE draft model with speculative decoding to quickly generate a response skeleton with format: `[PLAN] 1.<200><Search>[-]Task 2.<150><None>[-]Task [END]`

2. **Skeleton Parsing**: FSM-based SemanticLogitsProcessor parses skeleton to extract parallel branches

3. **Parallel Branch Decoding**: Decodes each branch in parallel, merges into final output

### Core Modules

- `SpecSoT/specsot_model.py`: Main `SpecSoTModel` class with `generate()`, `_generate_standard()`, `_generate_with_skeleton()`, `verify_step()`, `draft_step()`
- `SpecSoT/eagle_layer3.py`: EAGLE3 draft model for Qwen/LLaMA
- `SpecSoT/eagle_layer2.py`: EAGLE2 draft model for Vicuna-like models
- `SpecSoT/logits_processor.py`: GPU-optimized `SemanticLogitsProcessor` with tensor-based FSM
- `SpecSoT/utils.py`: Logits processing, mask building, skeleton parsing utilities
- `SpecSoT/kv_cache.py`: KV Cache management

### Base Model Implementations

Custom KV-cache-enabled implementations in `modeling_*_kv.py`:
- LLaMA (`modeling_llama_kv.py`)
- Qwen2 (`modeling_qwen2_kv.py`)
- Qwen3 (`modeling_qwen3_kv.py`)
- Mixtral (`modeling_mixtral_kv.py`)

### Distributed Module (`SpecSoT/distributed/`)

- `distributed_config.py`: Configuration for multi-GPU/multi-node setup
- `comm_manager.py`: ZMQ-based communication (P2P and Ring topologies)
- `distributed_prefill.py`: Distributed prefill phase management

## Dependencies

Key packages: torch 2.6.0, transformers>=4.53.1, accelerate, pynvml (for GPU monitoring)

Install: `pip install -r requirements.txt`
