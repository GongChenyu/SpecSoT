# plot_results.py
# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys
import ast
import numpy as np

def plot_metric(df, metric_col, title, ylabel, filename, output_dir):
    # 检查列是否存在，防止报错
    if metric_col not in df.columns:
        print(f"[Warning] Column '{metric_col}' not found in data. Skipping plot {filename}.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    token_settings = sorted(df['total_token_setting'].unique())
    # 使用 tab10 调色板
    palette = sns.color_palette("tab10", n_colors=len(token_settings))

    for idx, tt in enumerate(token_settings):
        subset = df[df['total_token_setting'] == tt]
        if not subset.empty:
            plt.plot(
                subset['num_para'], 
                subset[metric_col], 
                marker='o', 
                linewidth=2,
                label=f'Total Token: {tt}',
                color=palette[idx]
            )

    plt.xlabel("Number of Parallel Paths (num_para)", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    plt.legend(title="EAGLE Total Tokens")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"[Success] {title} saved to: {save_path}")
    plt.close()

def plot_branches_distribution(df, output_dir):
    """
    绘制 branches_len 分布图，展示长度差异
    选择差值最大的 50 条数据，用误差范围图展示
    """
    print("\n[Processing] Analyzing branches_len distribution...")
    
    # 检查列是否存在
    if 'branches_len' not in df.columns:
        print("[Warning] 'branches_len' column not found. Skipping distribution plot.")
        return
    
    # 处理 branches_len（可能是字符串格式的列表）
    def parse_branches_len(val):
        if pd.isna(val):
            return []
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        elif isinstance(val, list):
            return val
        else:
            return []
    
    # 计算统计指标
    df_analysis = df.copy()
    df_analysis['branches_len_parsed'] = df_analysis['branches_len'].apply(parse_branches_len)
    
    # 过滤掉空列表
    df_analysis = df_analysis[df_analysis['branches_len_parsed'].apply(len) > 0].copy()
    
    if len(df_analysis) == 0:
        print("[Warning] No valid branches_len data found.")
        return
    
    # 计算统计值
    df_analysis['bl_max'] = df_analysis['branches_len_parsed'].apply(lambda x: max(x) if len(x) > 0 else 0)
    df_analysis['bl_min'] = df_analysis['branches_len_parsed'].apply(lambda x: min(x) if len(x) > 0 else 0)
    df_analysis['bl_mean'] = df_analysis['branches_len_parsed'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df_analysis['bl_range'] = df_analysis['bl_max'] - df_analysis['bl_min']
    
    # 选择差值最大的 20 条数据
    top_n = min(20, len(df_analysis))
    df_top = df_analysis.nlargest(top_n, 'bl_range').copy()
    
    print(f"[Info] Selected top {top_n} samples with largest range")
    print(f"[Info] Range stats - Max: {df_top['bl_range'].max():.2f}, "
          f"Min: {df_top['bl_range'].min():.2f}, "
          f"Mean: {df_top['bl_range'].mean():.2f}")
    
    # 重置索引用于绘图
    df_top = df_top.reset_index(drop=True)
    
    # ========== 绘图 ==========
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # 定义颜色：大于平均值用蓝色，小于平均值用橙色
    color_above = '#1f77b4'  # 蓝色
    color_below = '#ff7f0e'  # 橙色
    
    # 遍历每个样本
    for idx, row in df_top.iterrows():
        branches_lengths = row['branches_len_parsed']
        mean_len = row['bl_mean']
        
        if len(branches_lengths) == 0:
            continue
        
        # 绘制每个分支长度的点（归一化：除以平均值）
        for branch_len in branches_lengths:
            normalized_len = branch_len / mean_len if mean_len > 0 else 0
            
            # 根据是否大于平均值选择颜色
            point_color = color_above if branch_len >= mean_len else color_below
            
            # 绘制点
            ax.scatter(idx, normalized_len, color=point_color, s=60, 
                      marker='o', zorder=3, edgecolors='black', linewidths=0.6,
                      alpha=0.7)
        
        # 计算归一化后的最大最小值
        norm_max = row['bl_max'] / mean_len if mean_len > 0 else 0
        norm_min = row['bl_min'] / mean_len if mean_len > 0 else 0
        
        # 绘制连接最大值和最小值的竖线（浅灰色，作为背景）
        ax.plot([idx, idx], [norm_min, norm_max], 
               color='gray', linewidth=1.5, alpha=0.3, zorder=1)
        
        # 绘制平均值的五角星（归一化后为 1.0）
        ax.scatter(idx, 1.0, color='red', s=120, 
                  marker='*', zorder=4, edgecolors='darkred', linewidths=0.8)
        # # 绘制平均值的水平短线（标记平均值位置）
        # ax.plot([idx-0.2, idx+0.2], [mean_len, mean_len], 
        #        color='red', linewidth=2, alpha=0.6, zorder=2)
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_above, 
               markersize=8, markeredgecolor='black', markeredgewidth=0.6,
               label='Branch Length ≥ Mean', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_below, 
               markersize=8, markeredgecolor='black', markeredgewidth=0.6,
               label='Branch Length < Mean', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=12, markeredgecolor='darkred', markeredgewidth=0.8,
               label='Mean Length (=1.0)', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)
    
    # 设置标签和标题
    ax.set_xlabel("Sample Index (Top 20 by Range)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Normalized Branch Length (relative to mean)", fontsize=11, fontweight='bold')
    ax.set_title("Branch Length Distribution Variability", 
                fontsize=12, pad=12, fontweight='bold')
    
    # 设置 x 轴范围，减小数据间距
    ax.set_xlim(-0.5, top_n - 0.5)
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.grid(True, linestyle=':', alpha=0.2, axis='x')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, "plot_branches_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Branch distribution plot saved to: {save_path}")
    plt.close()
    
    # 保存处理后的统计数据
    stats_cols = ['sample_id', 'total_token_setting', 'num_para', 'bl_max', 'bl_min', 
                  'bl_mean', 'bl_range'] if 'sample_id' in df_top.columns else \
                 ['total_token_setting', 'num_para', 'bl_max', 'bl_min', 
                  'bl_mean', 'bl_range']
    
    # 只保留存在的列
    stats_cols = [col for col in stats_cols if col in df_top.columns]
    
    stats_path = os.path.join(output_dir, "branches_distribution_stats.xlsx")
    df_top[stats_cols].to_excel(stats_path, index=False)
    print(f"[Success] Distribution statistics saved to: {stats_path}")

def main():
    parser = argparse.ArgumentParser()
    # 默认路径
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results", "benchmark_valid_raw_data.xlsx")
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}")
        print("Please run 'python test_observation.py' first to generate the data.")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from: {args.input}")
    try:
        # 根据文件扩展名选择读取方法
        if args.input.endswith('.xlsx'):
            raw_df = pd.read_excel(args.input)
        else:
            raw_df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[Error] Failed to read file: {e}")
        sys.exit(1)

    # === 数据聚合 ===
    print("Aggregating data...")
    
    # 定义需要聚合的列
    cols_to_agg = [
        'avg_accept_len',
        'throughput',
        'time',
        'length',
        'avg_draft_time', 
        'avg_verify_time', 
        'avg_update_time',
        'avg_att_time',
        'avg_ffn_time',
    ]
    
    # 过滤掉 CSV 中不存在的列
    existing_cols = [c for c in cols_to_agg if c in raw_df.columns]
    missing_cols = set(cols_to_agg) - set(existing_cols)
    
    if missing_cols:
        print(f"[Warning] The following columns are missing in the CSV: {missing_cols}")
        print("          Plots for these metrics will be skipped.")

    if not existing_cols:
        print("[Error] No valid metric columns found in CSV.")
        sys.exit(1)

    # 执行聚合 - 添加 memory 的 max 聚合和 sample_id 计数
    agg_dict = {col: 'mean' for col in existing_cols}
    if 'memory' in raw_df.columns:
        agg_dict['memory'] = 'max'
    agg_dict['sample_id'] = 'count'
    
    grouped_df = raw_df.groupby(['total_token_setting', 'num_para']).agg(agg_dict).reset_index()
    grouped_df = grouped_df.rename(columns={'sample_id': 'valid_sample_count'})
    grouped_df = grouped_df.sort_values(by=['total_token_setting', 'num_para'])

    # === 保存聚合表格 ===
    agg_path = os.path.join(args.output_dir, "benchmark_aggregated_table.xlsx")
    grouped_df.to_excel(agg_path, index=False)
    print(f"Aggregated table saved to: {agg_path}")

    # === 绘图 ===
    print("Generating plots...")
    
    # 1. Throughput
    plot_metric(grouped_df, 'throughput', "Throughput vs Parallelism", "Throughput (tokens/s)", "plot_throughput.png", args.output_dir)
    
    # 2. Avg Accept Length
    plot_metric(grouped_df, 'avg_accept_len', "Avg Accepted Length vs Parallelism", "Avg Accepted Length", "plot_avg_accept_len.png", args.output_dir)
    
    # 3. Draft Time
    plot_metric(grouped_df, 'avg_draft_time', "Avg Draft Time vs Parallelism", "Draft Time (s)", "plot_draft_time.png", args.output_dir)
    
    # 4. Verify Time
    plot_metric(grouped_df, 'avg_verify_time', "Avg Verify Time vs Parallelism", "Verify Time (s)", "plot_verify_time.png", args.output_dir)

    # 5. Update Time (New!)
    plot_metric(grouped_df, 'avg_update_time', "Avg Update Time vs Parallelism", "Update Time (s)", "plot_update_time.png", args.output_dir)

    # 6. Attention Time
    plot_metric(grouped_df, 'avg_att_time', "Avg Attention Time vs Parallelism", "Attention Time (s)", "plot_att_time.png", args.output_dir)

    # 7. FFN Time
    plot_metric(grouped_df, 'avg_ffn_time', "Avg FFN Time vs Parallelism", "FFN Time (s)", "plot_ffn_time.png", args.output_dir)

    # 8. Branches Length Distribution (使用原始数据，不是聚合数据)
    plot_branches_distribution(raw_df, args.output_dir)

    print("\n[Success] All plots and aggregated data generated successfully!")

if __name__ == "__main__":
    main()