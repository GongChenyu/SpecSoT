"""
单设备集中式推理脚本
- 仅使用一张GPU进行推理
- 输出生成的token id列表与解码文本

用法示例：
    python run_single_inference.py \
        --model_path /data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B \
        --gpu 4 \
        --prompt "请详细介绍一下人工智能的发展历史。" \
        --max_new_tokens 64
"""

import argparse
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

# 自定义模型实现
from modeling_qwen3_kv import Qwen3ForCausalLM
from kv_cache import initialize_past_key_values


def set_seed(seed: int) -> None:
    """设置随机种子，保证复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SingleInference")


def run_inference(
    model_path: str,
    prompt: str,
    gpu: int,
    max_new_tokens: int,
    seed: int,
) -> None:
    logger = setup_logger()

    # 选择GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda:0"

    # 设置随机种子
    set_seed(seed)

    logger.info(f"加载模型: {model_path}")
    logger.info(f"使用GPU: {gpu}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logger.info("开始生成...")
    input_ids = inputs["input_ids"]

    # 初始化KV Cache，容量按 prompt 长度 + 生成长度 预留，避免溢出
    prompt_len = input_ids.shape[1]
    kv_capacity = prompt_len + max_new_tokens + 8  # 预留少量余量
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
        model, max_length=kv_capacity, batch_size=1
    )

    def _forward_chunk(chunk_ids: torch.Tensor) -> torch.Tensor:
        """处理一个chunk（或单步），并更新KV Cache"""
        batch_size, seq_length = chunk_ids.shape

        # Embedding
        hidden_states = model.model.embed_tokens(chunk_ids)

        # 计算已有长度
        past_key_values_length = (
            past_key_values[0][0].current_length.item()
            if past_key_values[0][0].current_length > 0
            else 0
        )

        seq_length_with_past = seq_length + past_key_values_length

        # attention mask 与 position ids
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=device,
        )

        attention_mask = model.model._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        # 逐层前向，模型内部自动管理 KV cache 的拼接
        for layer_idx in range(model.config.num_hidden_layers):
            decoder_layer = model.model.layers[layer_idx]

            # 传入 past_key_value，模型内部会自动 cat 新的 KV
            # 见 modeling_qwen3_kv.py 中的实现：
            # key_states = past_key_value[0].cat(key_states, dim=2)
            # value_states = past_key_value[1].cat(value_states, dim=2)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx],
                output_attentions=False,
                use_cache=True,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            # past_key_value 已在模型内部更新，这里返回的是 None

        # 最后norm
        hidden_states = model.model.norm(hidden_states)
        return hidden_states

    generated_ids: List[int] = []

    with torch.no_grad():
        # Prefill：一次性处理完整 prompt
        last_hidden = _forward_chunk(input_ids)
        logits = model.lm_head(last_hidden)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_ids.append(next_token.item())

        # Decode：逐token自回归生成（贪心）
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].current_length.item()
            next_position_id = torch.tensor([[past_len]], device=device, dtype=torch.long)

            # 当前步输入为上一步token（第一步用prompt最后一个）
            if generated_ids:
                step_input = torch.tensor([[generated_ids[-1]]], device=device, dtype=torch.long)
            else:
                step_input = input_ids[:, -1:]

            hidden_states = _forward_chunk(step_input)

            logits = model.lm_head(hidden_states[:, -1:, :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            token_id = next_token.item()
            generated_ids.append(token_id)

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

    output_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    logger.info(f"生成的token id: {generated_ids}")
    logger.info("生成文本:")
    logger.info(generated_text)


def parse_args():
    parser = argparse.ArgumentParser(description="单设备集中式推理")
    parser.add_argument("--model_path", type=str, default='/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B', help="模型路径")
    parser.add_argument("--prompt", type=str, default='请详细介绍一下人工智能的发展历史。', help="输入prompt")
    parser.add_argument("--gpu", type=int, default=4, help="使用的GPU ID")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的新token数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        gpu=args.gpu,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
