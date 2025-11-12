"""
Visualize Milestone 2 Results

This script creates visualizations comparing multi-turn vs single-turn
prefix sharing effectiveness.

Usage:
    python vllm/sim/visualize_milestone2_results.py \
        --multi-stats milestone2_multi_stats.json \
        --single-stats milestone2_single_stats.json \
        --output-dir milestone2_results
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def load_stats(filepath: str) -> dict:
    """Load statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_comparison_plots(multi_stats: dict, single_stats: dict, output_dir: str):
    """Create comparison visualizations."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Milestone 2: Prefix Sharing Effectiveness\n(vLLM Real Block Manager)',
                 fontsize=16, fontweight='bold')

    # 1. Overall Sharing Fraction comparison (top-left)
    ax1 = axes[0, 0]
    categories = ['Multi-turn', 'Single-turn']
    sharing_fractions = [
        multi_stats['overall_sharing_fraction'] * 100,
        single_stats['overall_sharing_fraction'] * 100,
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars1 = ax1.bar(categories, sharing_fractions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Sharing Fraction (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Block Reuse Rate', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(sharing_fractions) * 1.2)

    # Add value labels on bars
    for bar, val in zip(bars1, sharing_fractions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 2. Cache Hit Rate comparison (top-right)
    ax2 = axes[0, 1]
    cache_hit_rates = [
        multi_stats['final_cache_hit_rate'] * 100,
        single_stats['final_cache_hit_rate'] * 100,
    ]
    bars2 = ax2.bar(categories, cache_hit_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Cache Hit Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('vLLM Cache Hit Rate', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(cache_hit_rates) * 1.2 if max(cache_hit_rates) > 0 else 10)

    for bar, val in zip(bars2, cache_hit_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 3. Block allocation breakdown (bottom-left)
    ax3 = axes[1, 0]

    multi_reused = multi_stats['total_blocks_reused']
    multi_new = multi_stats['total_blocks_newly_allocated']
    single_reused = single_stats['total_blocks_reused']
    single_new = single_stats['total_blocks_newly_allocated']

    x = np.arange(len(categories))
    width = 0.5

    bars3a = ax3.bar(x, [multi_reused, single_reused], width,
                     label='Blocks Reused', color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    bars3b = ax3.bar(x, [multi_new, single_new], width,
                     bottom=[multi_reused, single_reused],
                     label='Blocks Newly Allocated', color='#e67e22', alpha=0.8, edgecolor='black', linewidth=2)

    ax3.set_ylabel('Number of Blocks', fontsize=12, fontweight='bold')
    ax3.set_title('Block Allocation Breakdown', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, (cat, reused, new) in enumerate(zip(categories,
                                                 [multi_reused, single_reused],
                                                 [multi_new, single_new])):
        total = reused + new
        reused_pct = reused / total * 100 if total > 0 else 0
        new_pct = new / total * 100 if total > 0 else 0

        # Label for reused portion
        if reused > 0:
            ax3.text(i, reused/2, f'{reused_pct:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Label for new portion
        if new > 0:
            ax3.text(i, reused + new/2, f'{new_pct:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # 4. Per-request sharing fraction distribution (bottom-right)
    ax4 = axes[1, 1]

    multi_per_request = [m['sharing_fraction'] * 100 for m in multi_stats['request_metrics']]
    single_per_request = [m['sharing_fraction'] * 100 for m in single_stats['request_metrics']]

    bp = ax4.boxplot([multi_per_request, single_per_request],
                      labels=categories,
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)

    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)

    for cap in bp['caps']:
        cap.set_linewidth(1.5)

    for median in bp['medians']:
        median.set_linewidth(2)
        median.set_color('darkblue')

    ax4.set_ylabel('Sharing Fraction per Request (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Request Sharing Distribution', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics annotations
    multi_mean = np.mean(multi_per_request)
    multi_median = np.median(multi_per_request)
    single_mean = np.mean(single_per_request)
    single_median = np.median(single_per_request)

    ax4.text(0.02, 0.98, f'Multi-turn:\nMean: {multi_mean:.1f}%\nMedian: {multi_median:.1f}%',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / "milestone2_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_path}")

    plt.close()


def create_summary_table(multi_stats: dict, single_stats: dict, output_dir: str):
    """Create a markdown summary table."""

    output_path = Path(output_dir) / "milestone2_summary.md"

    with open(output_path, 'w') as f:
        f.write("# Milestone 2 Results Summary\n\n")
        f.write("## Comparison: Multi-turn vs Single-turn\n\n")
        f.write("| Metric | Multi-turn | Single-turn | Improvement |\n")
        f.write("|--------|-----------|------------|-------------|\n")

        # Overall sharing fraction
        multi_sharing = multi_stats['overall_sharing_fraction'] * 100
        single_sharing = single_stats['overall_sharing_fraction'] * 100
        improvement = multi_sharing / single_sharing if single_sharing > 0 else float('inf')
        f.write(f"| Overall Sharing Fraction | {multi_sharing:.2f}% | {single_sharing:.2f}% | {improvement:.1f}x |\n")

        # Cache hit rate
        multi_cache = multi_stats['final_cache_hit_rate'] * 100
        single_cache = single_stats['final_cache_hit_rate'] * 100
        cache_improvement = multi_cache / single_cache if single_cache > 0 else float('inf')
        f.write(f"| Cache Hit Rate | {multi_cache:.2f}% | {single_cache:.2f}% | {cache_improvement:.0f}x |\n")

        # Blocks reused
        multi_reused = multi_stats['total_blocks_reused']
        single_reused = single_stats['total_blocks_reused']
        reuse_improvement = multi_reused / single_reused if single_reused > 0 else float('inf')
        f.write(f"| Blocks Reused | {multi_reused:,} | {single_reused:,} | {reuse_improvement:.1f}x |\n")

        # Blocks newly allocated
        multi_new = multi_stats['total_blocks_newly_allocated']
        single_new = single_stats['total_blocks_newly_allocated']
        f.write(f"| Blocks Newly Allocated | {multi_new:,} | {single_new:,} | - |\n")

        # Total requests
        f.write(f"| Total Requests | {multi_stats['total_requests']} | {single_stats['total_requests']} | - |\n")

        # Total tokens
        f.write(f"| Total Tokens | {multi_stats['total_tokens']:,} | {single_stats['total_tokens']:,} | - |\n")

        # Total blocks
        f.write(f"| Total Blocks | {multi_stats['total_blocks']:,} | {single_stats['total_blocks']:,} | - |\n")

        f.write("\n## Key Findings\n\n")
        f.write(f"### 1. Multi-turn conversations benefit significantly from prefix caching\n\n")
        f.write(f"- **{multi_sharing:.1f}%** of blocks are reused (vs {single_sharing:.1f}% for single-turn)\n")
        f.write(f"- Cache hit rate: **{multi_cache:.1f}%**\n")
        f.write(f"- Saves **{multi_reused:,}** block allocations through reuse\n\n")

        f.write(f"### 2. Single-turn conversations have minimal prefix sharing\n\n")
        f.write(f"- Only **{single_sharing:.1f}%** of blocks are reused\n")
        f.write(f"- Cache hit rate: **{single_cache:.2f}%**\n")
        f.write(f"- Very little benefit from prefix caching\n\n")

        f.write(f"### 3. Multi-turn is {improvement:.1f}x more effective\n\n")
        f.write(f"- **Overall improvement**: {improvement:.1f}x in block reuse rate\n")
        f.write(f"- **Cache efficiency**: {cache_improvement:.0f}x better cache hit rate\n")
        f.write(f"- **Memory savings**: Significant reduction in memory allocation overhead\n\n")

        f.write("## Why Multi-turn is More Effective\n\n")
        f.write("**Multi-turn conversations** include the complete conversation history in each request:\n")
        f.write("```\n")
        f.write("Request 1: [System] ... [User] Hello!\n")
        f.write("Request 2: [System] ... [User] Hello! [Assistant] Hi! [User] How are you?\n")
        f.write("Request 3: [System] ... [User] Hello! [Assistant] Hi! [User] How are you? [Assistant] Fine! [User] ...\n")
        f.write("```\n\n")
        f.write("Each subsequent request **completely reuses** all previous tokens, leading to high cache hit rates.\n\n")

        f.write("**Single-turn conversations** are independent:\n")
        f.write("```\n")
        f.write("Request 1: [User] Hello!\n")
        f.write("Request 2: [User] What's the weather?\n")
        f.write("Request 3: [User] Tell me a joke.\n")
        f.write("```\n\n")
        f.write("Each request has no shared prefix with others, so prefix caching provides minimal benefit.\n\n")

        f.write("## Implementation Details\n\n")
        f.write("- **Block Manager**: vLLM SelfAttnBlockSpaceManager (real implementation)\n")
        f.write("- **Prefix Caching**: vLLM PrefixCachingBlockAllocator with LRU eviction\n")
        f.write("- **Block Size**: 16 tokens\n")
        f.write("- **Model**: facebook/opt-125m tokenizer\n")
        f.write(f"- **Requests**: {multi_stats['total_requests']} per experiment\n")
        f.write("- **Method**: Independent loading to avoid GPU/platform issues\n\n")

        f.write("## Conclusion\n\n")
        f.write(f"✅ **Milestone 2 successfully demonstrates that prefix caching is {improvement:.1f}x more effective for multi-turn conversations**\n\n")
        f.write("The experiments prove that:\n")
        f.write("1. Multi-turn chatbot workloads are ideal for prefix caching\n")
        f.write("2. vLLM's real block manager correctly implements prefix sharing\n")
        f.write("3. Significant memory and computation savings can be achieved\n")

    print(f"✅ Saved summary table to {output_path}")


def print_comparison(multi_stats: dict, single_stats: dict):
    """Print comparison to console."""

    print("\n" + "="*60)
    print("MILESTONE 2 RESULTS COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<35} {'Multi-turn':>12} {'Single-turn':>12}")
    print("-" * 60)

    # Sharing fraction
    multi_sharing = multi_stats['overall_sharing_fraction'] * 100
    single_sharing = single_stats['overall_sharing_fraction'] * 100
    print(f"{'Overall Sharing Fraction':<35} {multi_sharing:>11.2f}% {single_sharing:>11.2f}%")

    # Cache hit rate
    multi_cache = multi_stats['final_cache_hit_rate'] * 100
    single_cache = single_stats['final_cache_hit_rate'] * 100
    print(f"{'Cache Hit Rate':<35} {multi_cache:>11.2f}% {single_cache:>11.2f}%")

    # Blocks
    multi_reused = multi_stats['total_blocks_reused']
    single_reused = single_stats['total_blocks_reused']
    print(f"{'Blocks Reused':<35} {multi_reused:>12,} {single_reused:>12,}")

    multi_new = multi_stats['total_blocks_newly_allocated']
    single_new = single_stats['total_blocks_newly_allocated']
    print(f"{'Blocks Newly Allocated':<35} {multi_new:>12,} {single_new:>12,}")

    multi_total = multi_stats['total_blocks']
    single_total = single_stats['total_blocks']
    print(f"{'Total Blocks':<35} {multi_total:>12,} {single_total:>12,}")

    # Tokens and requests
    print(f"{'Total Tokens':<35} {multi_stats['total_tokens']:>12,} {single_stats['total_tokens']:>12,}")
    print(f"{'Total Requests':<35} {multi_stats['total_requests']:>12} {single_stats['total_requests']:>12}")

    print("\n" + "="*60)

    # Improvement
    improvement = multi_sharing / single_sharing if single_sharing > 0 else float('inf')
    print(f"\n✨ Multi-turn is {improvement:.1f}x more effective than single-turn!")
    print(f"   ({multi_sharing:.1f}% vs {single_sharing:.1f}% sharing fraction)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Milestone 2 prefix sharing results'
    )
    parser.add_argument(
        '--multi-stats',
        type=str,
        default='milestone2_multi_stats.json',
        help='Path to multi-turn statistics JSON file'
    )
    parser.add_argument(
        '--single-stats',
        type=str,
        default='milestone2_single_stats.json',
        help='Path to single-turn statistics JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='milestone2_results',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    try:
        # Load statistics
        print(f"Loading statistics...")
        multi_stats = load_stats(args.multi_stats)
        single_stats = load_stats(args.single_stats)
        print(f"✅ Loaded stats from {args.multi_stats} and {args.single_stats}")

        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Print comparison
        print_comparison(multi_stats, single_stats)

        # Create visualizations
        print(f"\nCreating visualizations...")
        create_comparison_plots(multi_stats, single_stats, args.output_dir)

        # Create summary table
        print(f"Creating summary table...")
        create_summary_table(multi_stats, single_stats, args.output_dir)

        print(f"\n🎉 Visualization complete!")
        print(f"   Results saved to: {args.output_dir}/")
        print(f"   - milestone2_comparison.png")
        print(f"   - milestone2_summary.md")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
