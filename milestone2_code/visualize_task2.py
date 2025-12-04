#!/usr/bin/env python3
"""
Visualize Task 2 Results: Cache Block Usage and Reuse Patterns
(Offline Mode: Reads results from task2_results.json)

根据project.pdf要求，可视化：
1. 每个请求的prefix sharing比例 (Hit Rate)
2. 每个cache block的命中次数 (Hits per block)
3. cache block重用的时间间隔 (Time gaps) - 包含 Single-turn 和 Multi-turn 对比
"""
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置非交互式后端
matplotlib.use('Agg')

# 路径设置
current_dir = Path(__file__).parent
results_path = current_dir / "task2_results.json"
output_dir = current_dir / "task2_visualizations"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Task 2: Visualizing Results (Offline Mode)")
print("=" * 80)

# 1. 读取数据
if not results_path.exists():
    print(f"❌ Error: Results file not found at {results_path}")
    print("Please run 'python compare_multi_vs_single_turn.py' first to generate the data.")
    sys.exit(1)

print(f"Reading data from: {results_path}")
with open(results_path, 'r') as f:
    data = json.load(f)

single_stats = data.get('single_turn', {})
multi_stats = data.get('multi_turn', {})
single_cache = single_stats.get('cache_stats', {})
multi_cache = multi_stats.get('cache_stats', {})

# 设置绘图风格
plt.style.use('default')
colors = {'single': '#ff7f0e', 'multi': '#2ca02c', 'avg': '#1f77b4', 'max': '#d62728'}

print("Generating charts...")

# ============================================================================
# Chart 1: Prefix Sharing Effectiveness (Hit Rate Comparison)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Correct Hit Rate\n(First Prefill Only)', 'vLLM GPU Hit Rate\n(All Accesses)']
s_vals = [single_stats.get('overall_hit_rate', 0) * 100, single_stats.get('gpu_hit_rate', 0) * 100]
m_vals = [multi_stats.get('overall_hit_rate', 0) * 100, multi_stats.get('gpu_hit_rate', 0) * 100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, s_vals, width, label='Single-turn', color=colors['single'], alpha=0.8)
bars2 = ax.bar(x + width/2, m_vals, width, label='Multi-turn', color=colors['multi'], alpha=0.8)

ax.set_ylabel('Hit Rate (%)', fontsize=12)
ax.set_title('Effectiveness of Prefix Sharing: Single vs Multi-turn', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 110)

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
save_path = output_dir / "1_prefix_sharing_effectiveness.png"
plt.savefig(save_path, dpi=300)
plt.close()

# ============================================================================
# Chart 2: Cache Block Access Intensity (Average Accesses per Block)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

labels = ['Single-turn', 'Multi-turn']
avg_data = [single_cache.get('avg_hits_per_block', 0), multi_cache.get('avg_hits_per_block', 0)]

x = np.arange(len(labels))

# 使用单柱状图显示平均访问次数
rects = ax.bar(x, avg_data, width=0.6, color=[colors['single'], colors['multi']], alpha=0.8)

ax.set_ylabel('Average Accesses per Block', fontsize=12)
ax.set_title('Cache Block Access Intensity (GPU-based)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + max(avg_data)*0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
save_path = output_dir / "2_avg_accesses_per_block.png"
plt.savefig(save_path, dpi=300)
print(f"✓ Saved: {save_path.name}")
plt.close()

# ============================================================================
# Chart 3: Reuse Time Gaps (Comparison Summary) - [MODIFIED]
# 现在同时显示 Single-turn 和 Multi-turn
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Min Gap', 'Avg Gap', 'Max Gap']

# 提取数据 (如果数据不存在则默认为 0)
s_gaps = [
    single_cache.get('min_reuse_gap_seconds', 0),
    single_cache.get('avg_reuse_gap_seconds', 0),
    single_cache.get('max_reuse_gap_seconds', 0)
]
m_gaps = [
    multi_cache.get('min_reuse_gap_seconds', 0),
    multi_cache.get('avg_reuse_gap_seconds', 0),
    multi_cache.get('max_reuse_gap_seconds', 0)
]

x = np.arange(len(metrics))
width = 0.35

# 绘制分组柱状图
bars_s = ax.bar(x - width/2, s_gaps, width, label='Single-turn', color=colors['single'], alpha=0.8)
bars_m = ax.bar(x + width/2, m_gaps, width, label='Multi-turn', color=colors['multi'], alpha=0.8)

ax.set_ylabel('Time (Seconds)', fontsize=12)
ax.set_title('Time Gap Between Block Reuses (Comparison)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 智能处理 Y 轴刻度 (如果差异太大使用对数坐标，但要小心 0 值)
all_vals = s_gaps + m_gaps
max_val = max(all_vals) if all_vals else 0
min_val = min([v for v in all_vals if v > 0]) if any(v > 0 for v in all_vals) else 0

if max_val > 0 and min_val > 0 and max_val > 100 * min_val:
    ax.set_yscale('symlog', linthresh=0.001) # 使用 symlog 以允许 0 值存在
    ax.set_ylabel('Time (Seconds) - Log Scale', fontsize=12)

# 标注数值
def add_gap_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # 如果高度为0，标注 "N/A" 或 "0"
        text = f'{height:.4f}s' if height > 0 else "0.0s"
        y_pos = height if height > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                text, ha='center', va='bottom', fontsize=9, rotation=0)

add_gap_labels(bars_s)
add_gap_labels(bars_m)

plt.tight_layout()
save_path = output_dir / "3_reuse_time_gaps_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"✓ Saved: {save_path.name}")
plt.close()

# ============================================================================
# Chart 4a: Single-turn Reuse Time Gap Distribution
# ============================================================================
s_gaps_all = single_cache.get('all_reuse_gaps', [])

if len(s_gaps_all) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 截断数据以便更好显示
    limit = 10.0
    s_plot = [g for g in s_gaps_all if g < limit]

    ax.hist(s_plot, bins=50, color=colors['single'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Reuse Time Gap (Seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Count)', fontsize=12)
    ax.set_title('Single-turn: Time Gap Distribution Between Block Reuses', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 添加统计信息
    avg_gap = single_cache.get('avg_reuse_gap_seconds', 0)
    ax.text(0.98, 0.97, f'Avg Gap: {avg_gap:.4f}s\nTotal Reuses: {len(s_gaps_all)}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = output_dir / "4a_single_turn_gap_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved: {save_path.name}")
    plt.close()
else:
    print("ℹ️ No single-turn reuse gap data, skipping single-turn histogram.")

# ============================================================================
# Chart 4b: Multi-turn Reuse Time Gap Distribution
# ============================================================================
m_gaps_all = multi_cache.get('all_reuse_gaps', [])

if len(m_gaps_all) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 截断数据以便更好显示
    limit = 10.0
    m_plot = [g for g in m_gaps_all if g < limit]

    ax.hist(m_plot, bins=50, color=colors['multi'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Reuse Time Gap (Seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Count)', fontsize=12)
    ax.set_title('Multi-turn: Time Gap Distribution Between Block Reuses', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 添加统计信息
    avg_gap = multi_cache.get('avg_reuse_gap_seconds', 0)
    ax.text(0.98, 0.97, f'Avg Gap: {avg_gap:.4f}s\nTotal Reuses: {len(m_gaps_all)}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = output_dir / "4b_multi_turn_gap_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved: {save_path.name}")
    plt.close()
else:
    print("ℹ️ No multi-turn reuse gap data, skipping multi-turn histogram.")

# ============================================================================
# Report Generation
# ============================================================================
report_path = output_dir / "analysis_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Task 2 Analysis Report\n" + "="*30 + "\n\n")
    f.write(f"Single-turn Hit Rate: {single_stats.get('overall_hit_rate', 0):.2%}\n")
    f.write(f"Multi-turn Hit Rate:  {multi_stats.get('overall_hit_rate', 0):.2%}\n\n")
    f.write(f"Single-turn Avg Gap:  {single_cache.get('avg_reuse_gap_seconds', 0):.4f}s\n")
    f.write(f"Multi-turn Avg Gap:   {multi_cache.get('avg_reuse_gap_seconds', 0):.4f}s\n")

print(f"\n📊 Analysis report saved to: {report_path}")
print("\nDone! Check the 'task2_visualizations' folder for images.")