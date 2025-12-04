#!/usr/bin/env python3
"""
Advanced Visualization for Milestone 2 Task 2

Generates three new visualizations:
1. CDF (Cumulative Distribution Function) of block hits
2. Distribution of block hits
3. Cache sharing rate over time
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'single': '#2E86AB',
    'multi': '#A23B72',
    'combined': '#F18F01'
}


def load_results(result_file):
    """加载实验结果"""
    with open(result_file, 'r') as f:
        return json.load(f)


def plot_cdf_block_hits(results, output_dir):
    """
    Chart 5: CDF of Block Hits

    Shows cumulative distribution of how many times each block is accessed.
    This helps understand the concentration of cache usage.
    """
    single_cache = results['single_turn']['cache_stats']
    multi_cache = results['multi_turn']['cache_stats']

    # Extract hit counts per block
    single_hits = list(single_cache.get('block_hit_counts', {}).values())
    multi_hits = list(multi_cache.get('block_hit_counts', {}).values())

    if not single_hits and not multi_hits:
        print("  ⚠ No block hit data available for CDF")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Single-turn CDF
    if single_hits:
        sorted_hits = np.sort(single_hits)
        cumulative = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)

        ax1.plot(sorted_hits, cumulative, color=colors['single'], linewidth=2, label='Single-turn')
        ax1.set_xlabel('Number of Hits per Block', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('Single-turn: CDF of Block Hit Counts', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        median_hits = np.median(sorted_hits)
        p90_hits = np.percentile(sorted_hits, 90)
        ax1.axvline(median_hits, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_hits:.1f}')
        ax1.axvline(p90_hits, color='orange', linestyle='--', alpha=0.7, label=f'90th %ile: {p90_hits:.1f}')
        ax1.legend()

        stats_text = f'Blocks: {len(single_hits)}\nMean: {np.mean(sorted_hits):.1f}\nStd: {np.std(sorted_hits):.1f}'
        ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Multi-turn CDF
    if multi_hits:
        sorted_hits = np.sort(multi_hits)
        cumulative = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)

        ax2.plot(sorted_hits, cumulative, color=colors['multi'], linewidth=2, label='Multi-turn')
        ax2.set_xlabel('Number of Hits per Block', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Multi-turn: CDF of Block Hit Counts', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        median_hits = np.median(sorted_hits)
        p90_hits = np.percentile(sorted_hits, 90)
        ax2.axvline(median_hits, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_hits:.1f}')
        ax2.axvline(p90_hits, color='orange', linestyle='--', alpha=0.7, label=f'90th %ile: {p90_hits:.1f}')
        ax2.legend()

        stats_text = f'Blocks: {len(multi_hits)}\nMean: {np.mean(sorted_hits):.1f}\nStd: {np.std(sorted_hits):.1f}'
        ax2.text(0.98, 0.05, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = output_dir / "5_cdf_block_hits.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Chart 5 saved: {save_path}")


def plot_block_hit_distribution(results, output_dir):
    """
    Chart 6: Distribution of Block Hits

    Shows histogram of how many blocks received N hits.
    This reveals patterns like: are most blocks accessed once, or multiple times?
    """
    single_cache = results['single_turn']['cache_stats']
    multi_cache = results['multi_turn']['cache_stats']

    # Extract hit counts per block
    single_hits = list(single_cache.get('block_hit_counts', {}).values())
    multi_hits = list(multi_cache.get('block_hit_counts', {}).values())

    if not single_hits and not multi_hits:
        print("  ⚠ No block hit data available for distribution")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Single-turn distribution
    if single_hits:
        # Count frequency of each hit count
        hit_counter = Counter(single_hits)
        hit_counts = sorted(hit_counter.keys())
        frequencies = [hit_counter[h] for h in hit_counts]

        ax1.bar(hit_counts, frequencies, color=colors['single'], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Number of Hits per Block', fontsize=12)
        ax1.set_ylabel('Number of Blocks', fontsize=12)
        ax1.set_title('Single-turn: Distribution of Block Hit Counts', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add statistics
        total_blocks = len(single_hits)
        max_hits = max(single_hits)
        avg_hits = np.mean(single_hits)

        stats_text = f'Total Blocks: {total_blocks}\nMax Hits: {max_hits}\nAvg Hits: {avg_hits:.1f}'
        ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Multi-turn distribution
    if multi_hits:
        hit_counter = Counter(multi_hits)
        hit_counts = sorted(hit_counter.keys())
        frequencies = [hit_counter[h] for h in hit_counts]

        ax2.bar(hit_counts, frequencies, color=colors['multi'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Hits per Block', fontsize=12)
        ax2.set_ylabel('Number of Blocks', fontsize=12)
        ax2.set_title('Multi-turn: Distribution of Block Hit Counts', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add statistics
        total_blocks = len(multi_hits)
        max_hits = max(multi_hits)
        avg_hits = np.mean(multi_hits)

        stats_text = f'Total Blocks: {total_blocks}\nMax Hits: {max_hits}\nAvg Hits: {avg_hits:.1f}'
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = output_dir / "6_block_hit_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Chart 6 saved: {save_path}")


def plot_sharing_rate_over_time(results, output_dir):
    """
    Chart 7: Cache Sharing Rate Over Time

    Shows how cache hit rate evolves as the experiment progresses.
    Uses sliding window to compute hit rate over time.
    """
    single_cache = results['single_turn']['cache_stats']
    multi_cache = results['multi_turn']['cache_stats']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Single-turn sharing rate over time
    if 'request_timeline' in single_cache and single_cache['request_timeline']:
        timeline = single_cache['request_timeline']

        # Extract timestamps and calculate cumulative hit rate
        timestamps = []
        cumulative_hit_rates = []

        total_hits = 0
        total_requests = 0

        for ts, req_id, num_hits, num_unique_blocks in timeline:
            total_requests += 1
            total_hits += num_hits

            # Cumulative hit rate = total hits / total requests
            hit_rate = (total_hits / total_requests) if total_requests > 0 else 0

            timestamps.append(ts - timeline[0][0])  # Relative time
            cumulative_hit_rates.append(hit_rate)

        ax1.plot(timestamps, cumulative_hit_rates, color=colors['single'], linewidth=2)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Cumulative Cache Hit Rate', fontsize=12)
        ax1.set_title('Single-turn: Cache Sharing Rate Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        final_rate = cumulative_hit_rates[-1] if cumulative_hit_rates else 0
        duration = timestamps[-1] if timestamps else 0

        stats_text = f'Duration: {duration:.2f}s\nFinal Rate: {final_rate:.2f}\nRequests: {len(timeline)}'
        ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Multi-turn sharing rate over time
    if 'request_timeline' in multi_cache and multi_cache['request_timeline']:
        timeline = multi_cache['request_timeline']

        timestamps = []
        cumulative_hit_rates = []

        total_hits = 0
        total_requests = 0

        for ts, req_id, num_hits, num_unique_blocks in timeline:
            total_requests += 1
            total_hits += num_hits

            hit_rate = (total_hits / total_requests) if total_requests > 0 else 0

            timestamps.append(ts - timeline[0][0])
            cumulative_hit_rates.append(hit_rate)

        ax2.plot(timestamps, cumulative_hit_rates, color=colors['multi'], linewidth=2)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Cumulative Cache Hit Rate', fontsize=12)
        ax2.set_title('Multi-turn: Cache Sharing Rate Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        final_rate = cumulative_hit_rates[-1] if cumulative_hit_rates else 0
        duration = timestamps[-1] if timestamps else 0

        stats_text = f'Duration: {duration:.2f}s\nFinal Rate: {final_rate:.2f}\nRequests: {len(timeline)}'
        ax2.text(0.98, 0.05, stats_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = output_dir / "7_sharing_rate_over_time.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Chart 7 saved: {save_path}")


def main():
    print("=" * 60)
    print("Advanced Visualization for Task 2")
    print("=" * 60)
    print()

    # 配置路径
    result_file = Path('milestone2_code/task2_results.json')
    output_dir = Path('milestone2_code/task2_visualizations')

    # 检查结果文件
    if not result_file.exists():
        print(f"✗ Error: Results file not found: {result_file}")
        print("  Please run: python milestone2_code/compare_multi_vs_single_turn.py")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载结果
    print("Loading results...")
    results = load_results(result_file)
    print(f"  ✓ Loaded from: {result_file}")
    print()

    # 生成三个新图表
    print("Generating advanced visualizations...")
    print()

    print("Chart 5: CDF of Block Hits")
    plot_cdf_block_hits(results, output_dir)
    print()

    print("Chart 6: Distribution of Block Hits")
    plot_block_hit_distribution(results, output_dir)
    print()

    print("Chart 7: Cache Sharing Rate Over Time")
    plot_sharing_rate_over_time(results, output_dir)
    print()

    print("=" * 60)
    print("✓ All advanced visualizations generated!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated charts:")
    print("  5. CDF of Block Hits (5_cdf_block_hits.png)")
    print("  6. Distribution of Block Hits (6_block_hit_distribution.png)")
    print("  7. Cache Sharing Rate Over Time (7_sharing_rate_over_time.png)")
    print()


if __name__ == "__main__":
    main()
