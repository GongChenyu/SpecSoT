"""
性能分析脚本
用于分析和对比两种cache同步策略的性能
"""

import re
import os
from typing import Dict, List
import json

def parse_log_file(log_path: str) -> Dict:
    """解析单个日志文件，提取时间信息"""
    
    if not os.path.exists(log_path):
        return None
        
    with open(log_path, 'r') as f:
        content = f.read()
    
    stats = {
        'rank': None,
        'prefill_time': None,
        'cache_sync_time': None,
        'decode_time': None,
        'total_time': None,
        'strategy': None
    }
    
    # 提取rank
    rank_match = re.search(r'Rank (\d+)', content)
    if rank_match:
        stats['rank'] = int(rank_match.group(1))
    
    # 提取策略
    strategy_match = re.search(r'策略: (\w+)', content)
    if strategy_match:
        stats['strategy'] = strategy_match.group(1)
    
    # 提取时间统计
    time_patterns = {
        'prefill_time': r'Prefill 时间:\s+([\d.]+)s',
        'cache_sync_time': r'Cache 同步时间:\s+([\d.]+)s',
        'decode_time': r'Decode 时间:\s+([\d.]+)s',
        'total_time': r'总时间:\s+([\d.]+)s'
    }
    
    for key, pattern in time_patterns.items():
        match = re.search(pattern, content)
        if match:
            stats[key] = float(match.group(1))
    
    return stats

def analyze_strategy(log_dir: str, strategy: str) -> Dict:
    """分析某个策略的性能"""
    
    results = []
    
    for rank in range(3):
        log_file = os.path.join(log_dir, f'rank{rank}.log')
        stats = parse_log_file(log_file)
        
        if stats:
            results.append(stats)
    
    if not results:
        return None
    
    # 计算平均值和最大值
    analysis = {
        'strategy': strategy,
        'num_devices': len(results),
        'prefill': {
            'avg': sum(r['prefill_time'] for r in results if r['prefill_time']) / len(results),
            'max': max(r['prefill_time'] for r in results if r['prefill_time']),
            'min': min(r['prefill_time'] for r in results if r['prefill_time'])
        },
        'cache_sync': {
            'avg': sum(r['cache_sync_time'] for r in results if r['cache_sync_time']) / len(results),
            'max': max(r['cache_sync_time'] for r in results if r['cache_sync_time']),
            'min': min(r['cache_sync_time'] for r in results if r['cache_sync_time'])
        },
        'decode': {
            'avg': sum(r['decode_time'] for r in results if r['decode_time']) / len(results),
            'max': max(r['decode_time'] for r in results if r['decode_time']),
            'min': min(r['decode_time'] for r in results if r['decode_time'])
        },
        'total': {
            'avg': sum(r['total_time'] for r in results if r['total_time']) / len(results),
            'max': max(r['total_time'] for r in results if r['total_time']),
            'min': min(r['total_time'] for r in results if r['total_time'])
        }
    }
    
    return analysis

def print_analysis(analysis: Dict):
    """打印分析结果"""
    
    if not analysis:
        print("❌ 无法分析，数据不足")
        return
    
    print("=" * 80)
    print(f"策略: {analysis['strategy'].upper()}")
    print(f"设备数: {analysis['num_devices']}")
    print("=" * 80)
    
    phases = [
        ('Prefill阶段', 'prefill'),
        ('Cache同步', 'cache_sync'),
        ('Decode阶段', 'decode'),
        ('总时间', 'total')
    ]
    
    print(f"\n{'阶段':<15} {'平均':<12} {'最小':<12} {'最大':<12}")
    print("-" * 80)
    
    for name, key in phases:
        data = analysis[key]
        print(f"{name:<15} {data['avg']:>10.3f}s  {data['min']:>10.3f}s  {data['max']:>10.3f}s")
    
    print("=" * 80)

def compare_strategies(pairwise_analysis: Dict, ring_analysis: Dict):
    """对比两种策略"""
    
    if not pairwise_analysis or not ring_analysis:
        print("⚠️  两种策略的数据不完整，无法对比")
        return
    
    print("\n" + "=" * 80)
    print("策略对比")
    print("=" * 80)
    
    print(f"\n{'阶段':<15} {'Pairwise':<15} {'Ring':<15} {'差异':<15} {'更快':<10}")
    print("-" * 80)
    
    phases = [
        ('Prefill', 'prefill'),
        ('Cache同步', 'cache_sync'),
        ('Decode', 'decode'),
        ('总时间', 'total')
    ]
    
    for name, key in phases:
        pairwise_time = pairwise_analysis[key]['avg']
        ring_time = ring_analysis[key]['avg']
        diff = abs(pairwise_time - ring_time)
        diff_pct = (diff / min(pairwise_time, ring_time)) * 100
        faster = "Pairwise" if pairwise_time < ring_time else "Ring"
        
        print(f"{name:<15} {pairwise_time:>13.3f}s  {ring_time:>13.3f}s  {diff_pct:>12.1f}%  {faster:<10}")
    
    print("=" * 80)
    
    # 给出建议
    print("\n建议:")
    
    if pairwise_analysis['cache_sync']['avg'] < ring_analysis['cache_sync']['avg']:
        print("  • Cache同步: Pairwise策略更快")
        print("    适合高带宽网络环境")
    else:
        print("  • Cache同步: Ring策略更快")
        print("    适合带宽受限的环境")
    
    if pairwise_analysis['total']['avg'] < ring_analysis['total']['avg']:
        speedup = (ring_analysis['total']['avg'] - pairwise_analysis['total']['avg']) / ring_analysis['total']['avg'] * 100
        print(f"  • 总体性能: Pairwise策略快 {speedup:.1f}%")
    else:
        speedup = (pairwise_analysis['total']['avg'] - ring_analysis['total']['avg']) / pairwise_analysis['total']['avg'] * 100
        print(f"  • 总体性能: Ring策略快 {speedup:.1f}%")

def main():
    print("\n" + "=" * 80)
    print("SP+PP分布式推理性能分析")
    print("=" * 80)
    
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录 {log_dir} 不存在")
        return
    
    # 分析两种策略（这里假设日志文件名包含策略信息，或者分别运行两次）
    # 简化版本：直接分析当前logs目录
    
    print("\n分析日志文件...")
    
    pairwise_analysis = None
    ring_analysis = None
    
    # 检查是否有分开的日志目录
    if os.path.exists("logs/pairwise"):
        pairwise_analysis = analyze_strategy("logs/pairwise", "pairwise")
    
    if os.path.exists("logs/ring"):
        ring_analysis = analyze_strategy("logs/ring", "ring")
    
    # 如果没有分开的目录，分析当前目录
    if pairwise_analysis is None and ring_analysis is None:
        # 尝试从日志内容判断策略
        rank0_log = "logs/rank0.log"
        if os.path.exists(rank0_log):
            with open(rank0_log, 'r') as f:
                content = f.read()
                if 'pairwise' in content.lower():
                    pairwise_analysis = analyze_strategy("logs", "pairwise")
                elif 'ring' in content.lower():
                    ring_analysis = analyze_strategy("logs", "ring")
    
    # 打印结果
    if pairwise_analysis:
        print_analysis(pairwise_analysis)
    
    if ring_analysis:
        print_analysis(ring_analysis)
    
    # 对比
    if pairwise_analysis and ring_analysis:
        compare_strategies(pairwise_analysis, ring_analysis)
    
    # 导出JSON
    if pairwise_analysis or ring_analysis:
        output = {}
        if pairwise_analysis:
            output['pairwise'] = pairwise_analysis
        if ring_analysis:
            output['ring'] = ring_analysis
        
        output_file = "performance_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ 详细结果已保存到: {output_file}")
    
    print("\n提示:")
    print("  1. 运行两次测试以对比两种策略")
    print("  2. 将日志分别保存到 logs/pairwise/ 和 logs/ring/")
    print("  3. 或者使用 test_both_strategies.sh 自动测试")

if __name__ == "__main__":
    main()
