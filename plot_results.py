# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

def plot_throughput(df, output_dir):
    """绘制吞吐量图表"""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # 获取唯一的 total_token 设置，并排序
    token_settings = sorted(df['total_token_setting'].unique())
    
    # 使用 Seaborn 的调色板，确保颜色区分明显
    palette = sns.color_palette("tab10", n_colors=len(token_settings))

    for idx, tt in enumerate(token_settings):
        subset = df[df['total_token_setting'] == tt]
        if not subset.empty:
            plt.plot(
                subset['num_para'], 
                subset['throughput'], 
                marker='o', 
                linewidth=2,
                label=f'Total Token: {tt}',
                color=palette[idx]
            )

    plt.xlabel("Number of Parallel Paths (num_para)", fontsize=12, fontweight='bold')
    plt.ylabel("Throughput (tokens/s)", fontsize=12, fontweight='bold')
    plt.title("Throughput vs Parallelism Level", fontsize=14, pad=15)
    plt.legend(title="EAGLE Total Tokens", title_fontsize=10, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局防止标签被截断
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "plot_throughput_vs_parallelism.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Throughput plot saved to: {save_path}")
    plt.close()

def plot_avg_accept_len(df, output_dir):
    """绘制平均接受长度图表"""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    token_settings = sorted(df['total_token_setting'].unique())
    palette = sns.color_palette("tab10", n_colors=len(token_settings))

    for idx, tt in enumerate(token_settings):
        subset = df[df['total_token_setting'] == tt]
        if not subset.empty:
            plt.plot(
                subset['num_para'], 
                subset['avg_accept_len'], 
                marker='s',           # 使用方形标记区分
                linestyle='--',       # 使用虚线区分
                linewidth=2,
                label=f'Total Token: {tt}',
                color=palette[idx]
            )

    plt.xlabel("Number of Parallel Paths (num_para)", fontsize=12, fontweight='bold')
    plt.ylabel("Avg Accepted Length (tokens)", fontsize=12, fontweight='bold')
    plt.title("Average Accepted Length vs Parallelism Level", fontsize=14, pad=15)
    plt.legend(title="EAGLE Total Tokens", title_fontsize=10, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "plot_avg_accept_len.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Avg Accept Len plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV.")
    # 默认路径指向之前脚本生成的 benchmark_valid_raw_data.csv
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results", "benchmark_valid_raw_data.csv")
    
    parser.add_argument("--input", type=str, default=default_input, help="Path to the raw results CSV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots. Defaults to input file directory.")
    
    args = parser.parse_args()

    # 1. 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}")
        print("Please run the benchmark script first or provide the correct path.")
        sys.exit(1)

    # 2. 确定输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from: {args.input}")
    try:
        raw_df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        sys.exit(1)

    # 3. 数据聚合 (重新计算均值，确保数据最新)
    # 我们根据 total_token_setting 和 num_para 进行分组
    print("Aggregating data...")
    if 'throughput' not in raw_df.columns or 'avg_accept_len' not in raw_df.columns:
        print("[Error] Required columns ('throughput', 'avg_accept_len') missing in CSV.")
        sys.exit(1)

    grouped_df = raw_df.groupby(['total_token_setting', 'num_para']).agg({
        'throughput': 'mean',
        'avg_accept_len': 'mean'
    }).reset_index()

    # 排序以确保画图时的连线顺序正确
    grouped_df = grouped_df.sort_values(by=['total_token_setting', 'num_para'])

    # 4. 执行绘图
    plot_throughput(grouped_df, args.output_dir)
    plot_avg_accept_len(grouped_df, args.output_dir)

if __name__ == "__main__":
    main()