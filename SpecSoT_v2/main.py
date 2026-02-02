# coding=utf-8
"""
SpecSoT v2 统一命令行入口

支持三种运行模式：
- infer: 单次推理测试
- serve: 启动 API 服务
- eval: 批量评估

使用示例：
    # 单次推理
    python -m SpecSoT_v2 infer --prompt "Hello" --base_model_path ...

    # 启动服务
    python -m SpecSoT_v2 serve --host 0.0.0.0 --port 8000

    # 批量评估
    python -m SpecSoT_v2 eval --task planning --num_samples 10
"""

import argparse
import sys
import os


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False


def add_model_args(parser):
    """添加模型参数"""
    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Base Model 路径"
    )
    parser.add_argument(
        "--eagle_model_path", type=str, required=True,
        help="Eagle Model 路径"
    )
    parser.add_argument(
        "--use_eagle3", type=str2bool, default=True,
        help="是否使用 Eagle3 模型"
    )


def add_inference_args(parser):
    """添加推理参数"""
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="采样温度 (0 表示 greedy)"
    )
    parser.add_argument(
        "--enable_parallel", type=str2bool, default=True,
        help="启用骨架并行模式"
    )
    parser.add_argument(
        "--use_semantic_constraint", type=str2bool, default=True,
        help="使用 FSM 语义约束"
    )


def cmd_infer(args):
    """单次推理命令"""
    import torch
    from .engine import SpecSoTGenerator

    print("=" * 50)
    print("SpecSoT 单次推理")
    print("=" * 50)
    print(f"模型: {os.path.basename(args.base_model_path)}")
    print(f"Eagle: {'Eagle3' if args.use_eagle3 else 'Eagle2'}")
    print(f"Prompt: {args.prompt[:50]}...")
    print("=" * 50)

    # 加载模型
    print("正在加载模型...")
    model = SpecSoTGenerator.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.eagle_model_path,
        use_eagle3=args.use_eagle3,
        dtype=torch.float16,
        device_map="cuda:0",
    )

    # 执行推理
    print("正在生成...")
    output_ids, stats = model.generate(
        task_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_parallel=args.enable_parallel,
        use_semantic_constraint=args.use_semantic_constraint,
    )

    # 解码输出
    text = model.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print("\n" + "=" * 50)
    print("生成结果:")
    print("=" * 50)
    print(text)
    print("\n" + "=" * 50)
    print("统计信息:")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_serve(args):
    """启动 API 服务命令"""
    import torch
    from .engine import SpecSoTGenerator
    from .api import SpecSoTServer

    print("=" * 50)
    print("SpecSoT API 服务")
    print("=" * 50)
    print(f"模型: {os.path.basename(args.base_model_path)}")
    print(f"地址: http://{args.host}:{args.port}")
    print("=" * 50)

    # 加载模型
    print("正在加载模型...")
    model = SpecSoTGenerator.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.eagle_model_path,
        use_eagle3=args.use_eagle3,
        dtype=torch.float16,
        device_map="cuda:0",
    )

    # 启动服务
    server = SpecSoTServer(model, host=args.host, port=args.port)
    server.run()


def cmd_eval(args):
    """批量评估命令"""
    from .engine import MasterEngine

    # 构造兼容的 args
    args.distributed = getattr(args, 'distributed', False)
    args.world_size = getattr(args, 'world_size', 1)
    args.devices = getattr(args, 'devices', "127.0.0.1#0")
    args.layer_splits = getattr(args, 'layer_splits', "")
    args.base_port = getattr(args, 'base_port', 45000)
    args.comm_mode = getattr(args, 'comm_mode', "p2p")
    args.chunk_size = getattr(args, 'chunk_size', 128)
    args.role = "master"
    args.rank = 0
    args.use_scheduling = getattr(args, 'use_scheduling', False)
    args.max_parallel = getattr(args, 'max_parallel', 2)
    args.seed = getattr(args, 'seed', 42)

    engine = MasterEngine(args)
    success = engine.run()
    if not success:
        sys.exit(1)


def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(
        description="SpecSoT v2 - 投机解码 + 骨架并行推理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # infer 子命令
    infer_parser = subparsers.add_parser("infer", help="单次推理测试")
    add_model_args(infer_parser)
    add_inference_args(infer_parser)
    infer_parser.add_argument(
        "--prompt", type=str, required=True,
        help="输入提示文本"
    )

    # serve 子命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务")
    add_model_args(serve_parser)
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="监听地址"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000,
        help="监听端口"
    )

    # eval 子命令
    eval_parser = subparsers.add_parser("eval", help="批量评估")
    add_model_args(eval_parser)
    add_inference_args(eval_parser)
    eval_parser.add_argument(
        "--task", type=str, default="planning",
        choices=["planning", "retrieval", "multi-doc-qa"],
        help="评估任务类型"
    )
    eval_parser.add_argument(
        "--num_samples", type=int, default=10,
        help="测试样本数量"
    )

    return parser


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
