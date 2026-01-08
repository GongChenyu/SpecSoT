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

# 定义统一的科学配色方案
SCI_COLORS = [
    '#274753',  # 深墨绿
    '#297270',  # 蓝灰绿
    '#299d8f',  # 青翠绿
    '#8ab07c',  # 薄荷绿
    '#e7c66b',  # 暖鹅黄
    '#f3a361',  # 琥珀橙
    '#e66d50',  # 落日红
]

# 定义不同的线型用于区分
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.']

def plot_metric(df, metric_col, title, ylabel, filename, output_dir):
    # 检查列是否存在，防止报错
    if metric_col not in df.columns:
        print(f"[Warning] Column '{metric_col}' not found in data. Skipping plot {filename}.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    token_settings = sorted(df['total_token_setting'].unique())

    for idx, tt in enumerate(token_settings):
        subset = df[df['total_token_setting'] == tt]
        if not subset.empty:
            plt.plot(
                subset['num_para'], 
                subset[metric_col], 
                marker='o', 
                linewidth=2,
                label=f'Total Token: {tt}',
                color=SCI_COLORS[idx % len(SCI_COLORS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)]
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
    
    # 使用科学配色方案中的颜色
    color_above = SCI_COLORS[2]  # 青翠绿
    color_below = SCI_COLORS[5]  # 琥珀橙
    
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
        
        # 绘制平均值的五角星（归一化后为 1.0），使用科学配色
        ax.scatter(idx, 1.0, color=SCI_COLORS[6], s=120,  # 落日红
                  marker='*', zorder=4, edgecolors=SCI_COLORS[0], linewidths=0.8)
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_above, 
               markersize=8, markeredgecolor='black', markeredgewidth=0.6,
               label='Branch Length ≥ Mean', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_below, 
               markersize=8, markeredgecolor='black', markeredgewidth=0.6,
               label='Branch Length < Mean', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=SCI_COLORS[6], 
               markersize=12, markeredgecolor=SCI_COLORS[0], markeredgewidth=0.8,
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

def plot_memory_usage_comparison(df, output_dir):
    """
    绘制三种方法的内存占用对比图
    - SoT: 每个分支都拼接prompt cache，无前缀复用
    - PDOS: 使用prefix复用，但用padding补充未结束的分支
    - Ours: 使用prompt复用和提前退出
    
    选择5个长度差距最大的数据点
    """
    print("\n[Processing] Generating memory usage comparison plot...")
    
    # 检查列是否存在
    if 'branches_len' not in df.columns:
        print("[Warning] 'branches_len' column not found. Skipping memory usage plot.")
        return
    
    # 处理 branches_len
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
    
    # 分析数据
    df_analysis = df.copy()
    df_analysis['branches_len_parsed'] = df_analysis['branches_len'].apply(parse_branches_len)
    
    # 过滤掉空列表
    df_analysis = df_analysis[df_analysis['branches_len_parsed'].apply(len) > 0].copy()
    
    if len(df_analysis) == 0:
        print("[Warning] No valid branches_len data found.")
        return
    
    # 计算统计信息
    df_analysis['num_branches'] = df_analysis['branches_len_parsed'].apply(len)
    df_analysis['bl_max'] = df_analysis['branches_len_parsed'].apply(lambda x: max(x) if len(x) > 0 else 0)
    df_analysis['bl_min'] = df_analysis['branches_len_parsed'].apply(lambda x: min(x) if len(x) > 0 else 0)
    df_analysis['bl_range'] = df_analysis['bl_max'] - df_analysis['bl_min']
    
    # 选择5个长度差距最大的样本
    top_n = min(5, len(df_analysis))
    df_top = df_analysis.nlargest(top_n, 'bl_range')
    
    selected_samples = []
    for idx, row in df_top.iterrows():
        selected_samples.append({
            'num_branches': row['num_branches'],
            'branches': row['branches_len_parsed'],
            'range': row['bl_range']
        })
        print(f"[Info] Branch num={row['num_branches']}, range={row['bl_range']:.2f}, lengths={row['branches_len_parsed']}")
    
    # 固定prompt长度
    prompt_len = 1359
    
    # 配色方案 - 三种方法用不同颜色
    color_sot = '#299d8f'    # 青翠绿 - SoT
    color_pdos = '#8ab07c'   # 薄荷绿 - PDOS
    color_ours = '#e7c66b'   # 暖鹅黄 - Ours
    
    # 阴影线（hatch patterns）用于区分不同的token类型
    hatch_prompt = '///'      # prompt cache
    hatch_generated = '...'   # 生成的token
    hatch_padding = 'xxx'     # padding
    
    # 准备绘图数据
    num_samples = len(selected_samples)
    methods = ['SoT', 'PDOS', 'Ours']
    method_colors = [color_sot, color_pdos, color_ours]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    bar_width = 0.18  # 更窄的柱子
    group_gap = 0.3   # 每组之间的间隔
    x_base = np.arange(num_samples) * (3 * bar_width + group_gap)  # 每组的基准位置
    
    # 为每个方法准备数据
    all_data = {
        'SoT': {'prompt': [], 'generated': [], 'padding': []},
        'PDOS': {'prompt': [], 'generated': [], 'padding': []},
        'Ours': {'prompt': [], 'generated': [], 'padding': []}
    }
    
    for sample in selected_samples:
        branches = sample['branches']
        num_branches = len(branches)
        max_branch_len = max(branches)
        sum_branches = sum(branches)
        
        # SoT: 每个分支都有完整的prompt cache
        sot_prompt = prompt_len * num_branches
        sot_generated = sum_branches
        sot_padding = 0
        all_data['SoT']['prompt'].append(sot_prompt)
        all_data['SoT']['generated'].append(sot_generated)
        all_data['SoT']['padding'].append(sot_padding)
        
        # PDOS: prompt复用，但使用padding填充到最大长度
        pdos_prompt = prompt_len
        pdos_generated_actual = sum_branches
        pdos_padding = max_branch_len * num_branches - sum_branches
        all_data['PDOS']['prompt'].append(pdos_prompt)
        all_data['PDOS']['generated'].append(pdos_generated_actual)
        all_data['PDOS']['padding'].append(pdos_padding)
        
        # Ours: prompt复用 + 提前退出
        ours_prompt = prompt_len
        ours_generated = sum_branches
        ours_padding = 0
        all_data['Ours']['prompt'].append(ours_prompt)
        all_data['Ours']['generated'].append(ours_generated)
        all_data['Ours']['padding'].append(ours_padding)
    
    # 绘制分组柱状图
    for i, method in enumerate(methods):
        x_pos = x_base + i * bar_width  # 每个方法的位置
        
        prompt_data = all_data[method]['prompt']
        generated_data = all_data[method]['generated']
        padding_data = all_data[method]['padding']
        
        method_color = method_colors[i]
        
        # 绘制堆叠柱状图，使用不同的阴影线区分token类型
        ax.bar(x_pos, prompt_data, bar_width,
               color=method_color, edgecolor='black', linewidth=0.8,
               hatch=hatch_prompt, alpha=0.9)
        ax.bar(x_pos, generated_data, bar_width, bottom=prompt_data,
               color=method_color, edgecolor='black', linewidth=0.8,
               hatch=hatch_generated, alpha=0.9)
        ax.bar(x_pos, padding_data, bar_width,
               bottom=[prompt_data[j] + generated_data[j] for j in range(len(prompt_data))],
               color=method_color, edgecolor='black', linewidth=0.8,
               hatch=hatch_padding, alpha=0.9)
    
    # 设置x轴标签 - 标签位于每组的中心
    x_label_pos = x_base + bar_width  # 三个柱子的中心位置
    x_labels = [f'Sample {i+1}' for i, s in enumerate(selected_samples)]
    # x_labels = [f'{s["num_branches"]} branches\n{s["branches"]}' for s in selected_samples]
    ax.set_xticks(x_label_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    
    # 设置标签和标题
    ax.set_xlabel('Samples', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cached Token Count', fontsize=13, fontweight='bold')
    ax.set_title('Memory Usage Comparison Across Different Methods', 
                fontsize=14, pad=15, fontweight='bold')
    
    # 创建图例
    from matplotlib.patches import Patch
    
    # 方法图例（用颜色区分）
    method_legend = [
        Patch(facecolor=color_sot, edgecolor='black', label='SoT'),
        Patch(facecolor=color_pdos, edgecolor='black', label='PDOS'),
        Patch(facecolor=color_ours, edgecolor='black', label='Ours')
    ]
    
    # Token类型图例（用阴影线区分）
    token_legend = [
        Patch(facecolor='white', edgecolor='black', hatch=hatch_prompt, label='Prompt Cache'),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_generated, label='Generated Tokens'),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_padding, label='Padding Tokens')
    ]
    
    # 添加两个图例
    legend1 = ax.legend(handles=method_legend, loc='upper left', framealpha=0.95, 
                       fontsize=10, title='Method', title_fontsize=11)
    ax.add_artist(legend1)
    ax.legend(handles=token_legend, loc='upper right', framealpha=0.95, 
             fontsize=10, title='Token Type', title_fontsize=11)
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Y轴格式化
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, "plot_memory_usage_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Memory usage comparison plot saved to: {save_path}")
    plt.close()

def plot_throughput_with_std(df, output_dir):
    """
    绘制 Throughput vs Parallelism 图，展示均值线和标准差阴影
    类似于性能随时间变化的图表风格
    """
    print("\n[Processing] Generating throughput plot with std deviation...")
    
    # 检查列是否存在
    if 'throughput' not in df.columns:
        print("[Warning] 'throughput' column not found. Skipping throughput std plot.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    token_settings = sorted(df['total_token_setting'].unique())
    
    for idx, tt in enumerate(token_settings):
        subset = df[df['total_token_setting'] == tt]
        
        if subset.empty:
            continue
        
        # 按 num_para 分组，计算均值和标准差
        grouped = subset.groupby('num_para')['throughput'].agg(['mean', 'std', 'count']).reset_index()
        
        # 如果只有一个样本，std 为 NaN，设为 0
        grouped['std'] = grouped['std'].fillna(0)
        
        x = grouped['num_para'].values
        mean = grouped['mean'].values
        std = grouped['std'].values
        
        color = SCI_COLORS[idx % len(SCI_COLORS)]
        linestyle = LINESTYLES[idx % len(LINESTYLES)]
        
        # 绘制标准差阴影区域（±1标准差），不显示边界线
        plt.fill_between(x, mean - std, mean + std, 
                        color=color, alpha=0.2, linewidth=0, edgecolor='none')
        
        # 绘制均值线
        plt.plot(x, mean, color=color, linestyle=linestyle, linewidth=2, 
                label=f'Total Token: {tt}', marker='o', markersize=6)
    
    # 设置标签和标题
    plt.xlabel('Number of Parallel Paths (num_para)', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (tokens/s)', fontsize=12, fontweight='bold')
    plt.title('Throughput vs Parallelism (with Standard Deviation)', 
             fontsize=14, pad=15, fontweight='bold')
    
    # 图例放在下方
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=min(4, len(token_settings)), fontsize=10, title="EAGLE Total Tokens")
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, "plot_throughput_with_std.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Throughput with std plot saved to: {save_path}")
    plt.close()

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
    
    # 1. Throughput (原有的聚合数据图)
    plot_metric(grouped_df, 'throughput', "Throughput vs Parallelism", "Throughput (tokens/s)", "plot_throughput.png", args.output_dir)
    
    # 1b. Throughput with std (新增：使用原始数据)
    plot_throughput_with_std(raw_df, args.output_dir)
    
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

    # 9. Memory Usage Comparison (使用原始数据)
    plot_memory_usage_comparison(raw_df, args.output_dir)

    print("\n[Success] All plots and aggregated data generated successfully!")

if __name__ == "__main__":
    main()