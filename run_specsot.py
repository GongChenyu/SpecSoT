# coding=utf-8
"""
SpecSoT v2 统一推理入口

执行模式组合：
1. 单机模式 (--distributed False):
   - 纯推理: --use_scheduling False
   - 带调度: --use_scheduling True

2. 分布式模式 (--distributed True):
   - 纯分布式: --use_scheduling False
   - 分布式+调度: --use_scheduling True

使用示例:
    # 单机推理（最简单）
    python run_specsot.py --distributed False

    # 单机多卡分布式
    python run_specsot.py --distributed True --devices "127.0.0.1#1,127.0.0.1#2,127.0.0.1#3"

    # 启用调度
    python run_specsot.py --use_scheduling True --max_parallel 2
"""

import sys
import argparse

from SpecSoT_v2.master import MasterEngine
from SpecSoT_v2.worker import WorkerEngine


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False

# 每个 add_argument 仅仅占用一行
def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description="SpecSoT 推理系统", formatter_class=argparse.RawDescriptionHelpFormatter,)

    # 模型配置
    # Qwen3
    # parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B", help="Base Model 路径")
    # parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Qwen3-4B_eagle3", help="Eagle Model 路径")
    # parser.add_argument("--use_eagle3", type=str2bool, default=True, help="是否使用 Eagle3 模型")
    # LLaMA3
    parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/Llama-3.1-8B-Instruct", help="Base Model 路径")
    parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/EAGLE3-LLaMA3.1-Instruct-8B", help="Eagle Model 路径")
    parser.add_argument("--use_eagle3", type=str2bool, default=True, help="是否使用 Eagle3 模型")
    # Vicuna
    # parser.add_argument("--base_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/vicuna-7b-v1.3", help="Base Model 路径")
    # parser.add_argument("--eagle_model_path", type=str, default="/data/home/chenyu/Coding/SD+SoT/models/EAGLE-Vicuna-7B-v1.3", help="Eagle Model 路径")
    # parser.add_argument("--use_eagle3", type=str2bool, default=False, help="是否使用 Eagle3 模型")

    # 推理配置
    parser.add_argument("--enable_parallel", type=str2bool, default=True, help="启用骨架并行模式")
    parser.add_argument("--use_semantic_constraint", type=str2bool, default=True, help="是否使用 FSM 语义约束")
    parser.add_argument("--use_bim_mode", type=str2bool, default=True, help="是否使用 BIM 模式 (True: In-One-Sequence, False: Batching)")
    parser.add_argument("--max_new_tokens", type=int, default=3000, help="最大生成token数")
    
    # 分布式配置
    parser.add_argument("--distributed", type=str2bool, default=False, help="是否启用分布式模式")
    parser.add_argument("--world_size", type=int, default=3, help="分布式总进程数")
    parser.add_argument("--devices", type=str, default="127.0.0.1#1,127.0.0.1#2,127.0.0.1#3", help="设备列表，格式: ip#gpu_id,ip#gpu_id,...")
    parser.add_argument("--layer_splits", type=str, default="14,28", help="层分割策略")
    parser.add_argument("--base_port", type=int, default=45000, help="通信基础端口")
    parser.add_argument("--comm_mode", type=str, default="p2p", choices=["p2p", "ring"], help="通信模式")
    parser.add_argument("--chunk_size", type=int, default=128, help="Sequence Parallel chunk大小")

    # 角色配置
    parser.add_argument("--role", type=str, default="master", choices=["master", "worker"], help="运行角色: master(主控) 或 worker(计算节点)")
    parser.add_argument("--rank", type=int, default=0, help="当前进程的rank（内部使用）")

    # 调度配置
    parser.add_argument("--use_scheduling", type=str2bool, default=True, help="是否启用分支调度")
    parser.add_argument("--max_parallel", type=int, default=2, help="每设备最大并行分支数")

    # 任务配置
    parser.add_argument("--task", type=str, default="planning", choices=["retrieval", "planning", "multi-doc-qa"], help="评估任务类型")
    parser.add_argument("--num_samples", type=int, default=1, help="测试样本数量")
    parser.add_argument("--seed", type=int, default=50, help="随机种子")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # 分流逻辑
    if args.role == "worker":
        # 作为分布式 Worker 运行
        engine = WorkerEngine(args)
        engine.run()
    else:
        # 作为 Master 运行（自动处理单机/分布式）
        engine = MasterEngine(args)
        success = engine.run()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
