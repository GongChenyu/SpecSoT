# coding=utf-8
"""
KV Cache 对齐验证脚本（SpecSoT 分布式版）

目的：对比单机与分布式 Prefill 得到的 KV Cache（包含 Base Model KV 与 Eagle Layer stable_kv），
      确认分布式逻辑是否正确同步。

使用方式：
1) 运行推理并保存 cache
   python verify_specsot_kv.py --mode run --world_size 3 --layer_splits 12,24 --gpu_ids 5 6 7 \
       --prompt "请详细介绍一下人工智能的发展历史。"

2) 仅分析已保存的 cache
   python verify_specsot_kv.py --mode analyze --cache_dir /path/to/cache_dir

3) 一步跑完（运行+分析）
   python verify_specsot_kv.py --mode all ...

保存文件：
- single_base.pkl / single_eagle.pkl
- dist_rank{r}_base.pkl / dist_rank{r}_eagle.pkl
- analyze_report.json（分析摘要）
"""

import argparse
import json
import logging
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from SpecSoT.specsot_model import SpecSoTModel
from SpecSoT.kv_cache import initialize_past_key_values
from SpecSoT.distributed.distributed_config import DistributedConfig


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _extract_kv_tensors(key_cache, value_cache):
    """提取 KV 数据到 CPU，自动裁剪到 current_length。"""
    # 兼容 KVCache / 直接 Tensor / tuple 情况
    if hasattr(key_cache, "data") and hasattr(key_cache, "current_length"):
        # KVCache 实例
        key_data = key_cache.data
        value_data = value_cache.data
        cur_len = key_cache.current_length
        if torch.is_tensor(cur_len):
            cur_len = cur_len.item()
        key_data = key_data[:, :, :cur_len, :]
        value_data = value_data[:, :, :cur_len, :]
    elif hasattr(key_cache, "get_data"):
        # 兼容带 get_data 接口的缓存
        key_data = key_cache.get_data()
        value_data = value_cache.get_data()
        cur_len = key_cache.current_length
        if torch.is_tensor(cur_len):
            cur_len = cur_len.item()
        key_data = key_data[:, :, :cur_len, :]
        value_data = value_data[:, :, :cur_len, :]
    else:
        # 已是 Tensor
        key_data = key_cache
        value_data = value_cache

    return key_data.detach().cpu(), value_data.detach().cpu()


def save_kv_cache(kv_cache_list: List, save_path: str, metadata: Dict = None):
    """保存 KV Cache 列表到 pickle（CPU Tensor）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "num_layers": len(kv_cache_list),
        "caches": [],
    }
    for idx, (k, v) in enumerate(kv_cache_list):
        key_data, value_data = _extract_kv_tensors(k, v)
        payload["caches"].append(
            {
                "layer_idx": idx,
                "key": key_data,
                "value": value_data,
                "key_shape": tuple(key_data.shape),
                "value_shape": tuple(value_data.shape),
            }
        )
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    logging.info(f"KV Cache saved to {save_path} ({payload['num_layers']} layers)")
    return payload


def save_obj(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Saved: {path}")


def load_kv_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def cleanup_ports(base_port: int, world_size: int):
    ports = set()
    for sender in range(world_size):
        for receiver in range(world_size):
            if sender != receiver:
                ports.add(base_port + sender * world_size + receiver)
    for port in ports:
        os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true")
    logging.info(f"cleaned ports: {min(ports)}-{max(ports)}")


# -----------------------------------------------------------------------------
# 运行逻辑
# -----------------------------------------------------------------------------

@dataclass
class RunArgs:
    base_model_path: str
    eagle_model_path: str
    prompt: str
    chunk_size: int
    world_size: int
    layer_splits: str
    gpu_ids: List[int]
    base_port: int
    comm_mode: str
    seed: int
    cache_dir: str
    max_new_tokens: int


def _prepare_model_single(run_args: RunArgs) -> Tuple[SpecSoTModel, torch.Tensor]:
    set_random_seed(run_args.seed)
    model = SpecSoTModel.from_pretrained(
        base_model_path=run_args.base_model_path,
        ea_model_path=run_args.eagle_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        total_token=40,
        depth=4,
        top_k=6,
        distributed_config=None,
    )
    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"
    input_ids = tokenizer(run_args.prompt, return_tensors="pt").input_ids.to(model.base_model.device)
    max_kv_len = input_ids.shape[1] + run_args.max_new_tokens + 100
    model.past_key_values, model.past_key_values_data, model.current_length_data = initialize_past_key_values(
        model.base_model, max_length=max_kv_len
    )
    return model, input_ids


def run_single(run_args: RunArgs):
    logging.info("==== Running single-device prefill ====")
    model, input_ids = _prepare_model_single(run_args)

    # 捕获传入 Eagle 的 hidden_states / input_ids 以及 draft 结果
    draft_capture = {}
    orig_generate = model.eagle_layer.generate_draft_tree

    def _capture_generate(hidden_states, input_ids_inner, *args, **kwargs):
        result = orig_generate(hidden_states, input_ids_inner, *args, **kwargs)
        draft_capture["hidden_states_for_eagle"] = hidden_states.detach().cpu()
        draft_capture["input_ids_for_eagle"] = input_ids_inner.detach().cpu()
        draft_capture["tree_result"] = tuple(x.detach().cpu() for x in result)
        return result

    model.eagle_layer.generate_draft_tree = _capture_generate  # type: ignore

    model.prefill_single(input_ids)

    # 保存 draft 结果
    draft_path = os.path.join(run_args.cache_dir, "single_draft.pkl")
    save_obj(draft_capture, draft_path)

    base_meta = {
        "mode": "single",
        "prompt": run_args.prompt,
        "seed": run_args.seed,
        "num_layers": model.base_model.config.num_hidden_layers,
    }
    base_path = os.path.join(run_args.cache_dir, "single_base.pkl")
    save_kv_cache(model.past_key_values, base_path, base_meta)

    if model.eagle_layer.stable_kv is not None:
        eagle_meta = {
            "mode": "single",
            "prompt": run_args.prompt,
            "seed": run_args.seed,
            "desc": "eagle stable kv",
        }
        eagle_path = os.path.join(run_args.cache_dir, "single_eagle.pkl")
        save_kv_cache([model.eagle_layer.stable_kv[0]], eagle_path, eagle_meta)
    else:
        logging.warning("single run: eagle_layer.stable_kv is None")


def _distributed_worker(rank: int, run_args: RunArgs):
    # 每个进程独立日志
    logging.basicConfig(level=logging.INFO, format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(run_args.gpu_ids[rank])
    set_random_seed(run_args.seed + rank)

    dist_cfg = DistributedConfig.from_layer_splits_str(
        layer_splits_str=run_args.layer_splits,
        rank=rank,
        world_size=run_args.world_size,
        base_port=run_args.base_port,
        comm_mode=run_args.comm_mode,
        chunk_size=run_args.chunk_size,
    )

    model = SpecSoTModel.from_pretrained(
        base_model_path=run_args.base_model_path,
        ea_model_path=run_args.eagle_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        total_token=40,
        depth=4,
        top_k=6,
        distributed_config=dist_cfg,
    )
    tokenizer = model.get_tokenizer()
    tokenizer.padding_side = "left"

    input_ids = tokenizer(run_args.prompt, return_tensors="pt").input_ids.to(model.base_model.device)
    max_kv_len = input_ids.shape[1] + run_args.max_new_tokens + 100
    model.past_key_values, model.past_key_values_data, model.current_length_data = initialize_past_key_values(
        model.base_model, max_length=max_kv_len
    )

    # 捕获分布式 draft 数据（只在最后一个 rank 有 tree_result，但每个 chunk 都记录输入）
    chunk_records: List[Dict] = []
    orig_generate_dist = model.eagle_layer.generate_draft_tree_dist_prefill

    def _capture_generate_dist(hidden_states_for_eagle, input_ids_this_chunk, *args, **kwargs):
        tree_result, incremental_kv = orig_generate_dist(
            hidden_states_for_eagle, input_ids_this_chunk, *args, **kwargs
        )
        rec = {
            "chunk_idx": kwargs.get("chunk_idx", 0),
            "is_last_chunk": kwargs.get("is_last_chunk", False),
            "hidden_states_for_eagle": hidden_states_for_eagle.detach().cpu(),
            "input_ids_this_chunk": input_ids_this_chunk.detach().cpu(),
            "tree_result": None,
            "incremental_kv": None,
        }
        if tree_result is not None:
            rec["tree_result"] = tuple(x.detach().cpu() for x in tree_result)
        if incremental_kv is not None:
            rec["incremental_kv"] = tuple(x.detach().cpu() for x in incremental_kv)
        chunk_records.append(rec)
        return tree_result, incremental_kv

    model.eagle_layer.generate_draft_tree_dist_prefill = _capture_generate_dist  # type: ignore

    # 仅执行分布式 Prefill（不进入解码环节）
    model.distributed_prefill_manager.prefill_single_distributed(input_ids, model.past_key_values, None)

    meta = {
        "mode": "distributed",
        "prompt": run_args.prompt,
        "seed": run_args.seed + rank,
        "rank": rank,
        "world_size": run_args.world_size,
        "layer_splits": run_args.layer_splits,
    }
    time.sleep(5)  # 确保cache顺利收到
    base_path = os.path.join(run_args.cache_dir, f"dist_rank{rank}_base.pkl")
    save_kv_cache(model.past_key_values, base_path, meta)

    if model.eagle_layer.stable_kv is not None:
        eagle_path = os.path.join(run_args.cache_dir, f"dist_rank{rank}_eagle.pkl")
        save_kv_cache([model.eagle_layer.stable_kv[0]], eagle_path, meta)
    else:
        logging.warning(f"rank {rank}: eagle_layer.stable_kv is None")

    # 保存 draft 捕获结果（所有 rank 都存，便于检查 chunk 拼接；tree_result 仅最后 rank 有）
    draft_path = os.path.join(run_args.cache_dir, f"dist_rank{rank}_draft.pkl")
    save_obj({"chunks": chunk_records}, draft_path)

    # 关闭通信
    model.cleanup_distributed()


def run_distributed(run_args: RunArgs):
    logging.info("==== Running distributed prefill ====")
    cleanup_ports(run_args.base_port, run_args.world_size)
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(run_args.world_size):
        p = ctx.Process(target=_distributed_worker, args=(r, run_args))
        p.start()
        procs.append(p)
        time.sleep(0.5)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"rank process exited with code {p.exitcode}")


# -----------------------------------------------------------------------------
# 分析逻辑
# -----------------------------------------------------------------------------

@dataclass
class CompareResult:
    ok: bool
    mismatched: List[Dict]


def _compare_two_layers(layer_idx: int, a: Dict, b: Dict, rtol: float, atol: float) -> Dict:
    ka, va = a["key"], a["value"]
    kb, vb = b["key"], b["value"]
    if ka.shape != kb.shape or va.shape != vb.shape:
        return {
            "layer": layer_idx,
            "match": False,
            "reason": "shape_mismatch",
            "key_shape_a": list(ka.shape),
            "key_shape_b": list(kb.shape),
            "value_shape_a": list(va.shape),
            "value_shape_b": list(vb.shape),
        }
    key_allclose = torch.allclose(ka, kb, rtol=rtol, atol=atol)
    val_allclose = torch.allclose(va, vb, rtol=rtol, atol=atol)
    max_key_diff = torch.max(torch.abs(ka - kb)).item()
    max_val_diff = torch.max(torch.abs(va - vb)).item()
    if key_allclose and val_allclose:
        return {"layer": layer_idx, "match": True, "max_key_diff": max_key_diff, "max_value_diff": max_val_diff}
    return {
        "layer": layer_idx,
        "match": False,
        "reason": "value_mismatch",
        "max_key_diff": max_key_diff,
        "max_value_diff": max_val_diff,
    }


def compare_cache_files(single_path: str, dist_path: str, rtol: float, atol: float) -> CompareResult:
    single = load_kv_cache(single_path)
    dist = load_kv_cache(dist_path)
    if single["num_layers"] != dist["num_layers"]:
        return CompareResult(False, [{"layer": -1, "match": False, "reason": "num_layers_mismatch"}])
    mismatched = []
    for i in range(single["num_layers"]):
        res = _compare_two_layers(i, single["caches"][i], dist["caches"][i], rtol, atol)
        if not res["match"]:
            mismatched.append(res)
    ok = len(mismatched) == 0
    return CompareResult(ok, mismatched)


def analyze(run_args: RunArgs, rtol: float, atol: float):
    report = {}
    single_base = os.path.join(run_args.cache_dir, "single_base.pkl")
    single_eagle = os.path.join(run_args.cache_dir, "single_eagle.pkl")
    dist_base = os.path.join(run_args.cache_dir, "dist_rank0_base.pkl")
    dist_eagle = os.path.join(run_args.cache_dir, f"dist_rank{run_args.world_size-1}_eagle.pkl")
    single_draft = os.path.join(run_args.cache_dir, "single_draft.pkl")
    dist_draft = os.path.join(run_args.cache_dir, f"dist_rank{run_args.world_size-1}_draft.pkl")

    logging.info("==== Analyzing Base Model KV ====")
    base_result = compare_cache_files(single_base, dist_base, rtol, atol)
    report["base_ok"] = base_result.ok
    report["base_mismatched"] = base_result.mismatched
    if base_result.ok:
        logging.info("Base KV: all layers match within tolerance")
    else:
        logging.warning(f"Base KV mismatches: {base_result.mismatched}")

    logging.info("==== Analyzing Eagle stable_kv ====")
    if not os.path.exists(single_eagle) or not os.path.exists(dist_eagle):
        logging.warning("Eagle cache file missing; skip eagle comparison")
        report["eagle_ok"] = False
        report["eagle_mismatched"] = [{"reason": "file_missing"}]
    else:
        eagle_result = compare_cache_files(single_eagle, dist_eagle, rtol, atol)
        report["eagle_ok"] = eagle_result.ok
        report["eagle_mismatched"] = eagle_result.mismatched
        if eagle_result.ok:
            logging.info("Eagle stable_kv matches")
        else:
            logging.warning(f"Eagle stable_kv mismatches: {eagle_result.mismatched}")

    logging.info("==== Analyzing Draft Outputs ====")
    if not (os.path.exists(single_draft) and os.path.exists(dist_draft)):
        logging.warning("Draft cache file missing; skip draft comparison")
        report["draft_ok"] = False
        report["draft_mismatched"] = [{"reason": "file_missing"}]
    else:
        with open(single_draft, "rb") as f:
            s_draft = pickle.load(f)
        with open(dist_draft, "rb") as f:
            d_draft = pickle.load(f)

        # 单机引用
        s_hidden = s_draft.get("hidden_states_for_eagle")
        s_input_ids = s_draft.get("input_ids_for_eagle")
        s_tree = s_draft.get("tree_result")

        # 分布式（按 chunk_idx 排序后拼接）
        d_chunks = sorted(d_draft.get("chunks", []), key=lambda x: x.get("chunk_idx", 0))
        d_hidden_list = [c["hidden_states_for_eagle"] for c in d_chunks]
        d_input_list = [c["input_ids_this_chunk"] for c in d_chunks]
        d_hidden = torch.cat(d_hidden_list, dim=1) if d_hidden_list else None
        d_input_ids = torch.cat(d_input_list, dim=1) if d_input_list else None
        d_tree = None
        for c in d_chunks:
            if c.get("is_last_chunk", False) and c.get("tree_result") is not None:
                d_tree = c["tree_result"]
                break

        draft_mismatched: List[Dict] = []

        def _cmp_tensor(name: str, a: torch.Tensor, b: torch.Tensor):
            if a.shape != b.shape:
                draft_mismatched.append({"item": name, "match": False, "reason": "shape_mismatch", "shape_a": list(a.shape), "shape_b": list(b.shape)})
                return
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                diff = (a - b).abs()
                draft_mismatched.append({"item": name, "match": False, "reason": "value_mismatch", "max_diff": diff.max().item()})

        if s_hidden is not None and d_hidden is not None:
            _cmp_tensor("hidden_states_for_eagle", s_hidden, d_hidden)
        if s_input_ids is not None and d_input_ids is not None:
            if not torch.equal(s_input_ids, d_input_ids):
                draft_mismatched.append({"item": "input_ids_for_eagle", "match": False, "reason": "value_mismatch"})

        if s_tree is not None and d_tree is not None:
            names = ["draft_tokens", "retrieve_indices", "tree_mask", "tree_position_ids"]
            for name, sa, sb in zip(names, s_tree, d_tree):
                _cmp_tensor(name, sa, sb)
        else:
            draft_mismatched.append({"item": "tree_result", "match": False, "reason": "missing"})

        report["draft_ok"] = len(draft_mismatched) == 0
        report["draft_mismatched"] = draft_mismatched
        if report["draft_ok"]:
            logging.info("Draft outputs match (hidden/input_ids/tree_result)")
        else:
            logging.warning(f"Draft mismatches: {draft_mismatched}")

    report_path = os.path.join(run_args.cache_dir, "analyze_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"Analyze report saved to {report_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate distributed KV cache correctness")
    parser.add_argument("--mode", type=str, default="all", choices=["run", "analyze", "all"], help="run=只运行, analyze=只分析, all=运行+分析")
    parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B")
    parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3")
    parser.add_argument("--prompt", type=str, default="请详细介绍一下人工智能的发展历史。")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--world_size", type=int, default=3)
    parser.add_argument("--layer_splits", type=str, default="12,24")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument("--base_port", type=int, default=45000)
    parser.add_argument("--comm_mode", type=str, default="p2p", choices=["p2p", "ring"])
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--cache_dir", type=str, default="/data/home/chenyu/Coding/SD+SoT/Speculative-Decoding-Enabled-Skeleton-of-Thought/prefill_cache_specsot_verify")
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_args = RunArgs(
        base_model_path=args.base_model_path,
        eagle_model_path=args.eagle_model_path,
        prompt=args.prompt,
        chunk_size=args.chunk_size,
        world_size=args.world_size,
        layer_splits=args.layer_splits,
        gpu_ids=args.gpu_ids,
        base_port=args.base_port,
        comm_mode=args.comm_mode,
        seed=args.seed,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
    )

    if args.mode in ["run", "all"]:
        run_single(run_args)
        run_distributed(run_args)

    if args.mode in ["analyze", "all"]:
        analyze(run_args, rtol=args.rtol, atol=args.atol)

    logging.info("Done")


if __name__ == "__main__":
    main()

# python verify_specsot_kv.py --mode all 

