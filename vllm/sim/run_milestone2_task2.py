"""
Milestone 2 Complete Pipeline

This script runs the complete Milestone 2 workflow:
1. Generate single-turn and multi-turn traces (Task 1)
2. Run experiments with vLLM real block manager (Task 2)
3. Collect metrics and visualize results

Usage:
    python vllm/sim/run_milestone2_task2.py \
        --data-path vllm/ShareGPTData.jsonl \
        --max-conversations 100 \
        --output-dir milestone2_results \
        --model facebook/opt-125m \
        --block-size 16
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add vllm to path
VLLM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(VLLM_ROOT))

from sim.client_simulator import (
    ShareGPTLoader,
    ChatTemplateFormatter,
    RequestGenerator,
    create_trace_file_for_simulator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_traces(
    data_path: str,
    output_dir: str,
    max_conversations: int,
    model_name: str,
    arrival_rate: float = 2.0,
) -> Dict[str, str]:
    """
    Generate single-turn and multi-turn traces.

    Returns:
        Dictionary with paths to generated trace files
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Generating Traces (Task 1)")
    logger.info("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Load ShareGPT data
    logger.info(f"Loading ShareGPT data from {data_path}...")
    loader = ShareGPTLoader(trace_path=data_path, max_conversations=max_conversations)
    conversations = loader.conversations
    logger.info(f"✅ Loaded {len(conversations)} conversations")

    # Format with chat template
    logger.info(f"Formatting with chat template for {model_name}...")

    # Load tokenizer for chat template
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        formatter = ChatTemplateFormatter(tokenizer=tokenizer, model_name=model_name)
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}, using simple format")
        formatter = ChatTemplateFormatter(model_name=model_name)

    # Format each conversation
    formatted_convs = []
    for conv in conversations:
        formatted_prompt = formatter.format_conversation(conv.turns)
        formatted_convs.append({
            'conversation_id': conv.conversation_id,
            'turns': conv.turns,
            'formatted_prompt': formatted_prompt,
        })

    logger.info(f"✅ Formatted {len(formatted_convs)} conversations")

    # Generate traces
    logger.info("Generating traces with Poisson arrival...")
    generator = RequestGenerator(
        conversations=conversations,
        arrival_rate=arrival_rate,
        use_poisson=True,
        seed=42
    )

    # Single-turn trace
    single_trace_path = os.path.join(output_dir, "single_turn_trace.jsonl")
    logger.info(f"Generating single-turn trace: {single_trace_path}")
    single_traces = generator.generate_single_turn_traces(formatter)
    create_trace_file_for_simulator(single_traces, single_trace_path)
    logger.info(f"✅ Generated {len(single_traces)} single-turn requests")

    # Multi-turn trace
    multi_trace_path = os.path.join(output_dir, "multi_turn_trace.jsonl")
    logger.info(f"Generating multi-turn trace: {multi_trace_path}")
    multi_traces = generator.generate_multi_turn_traces(formatter)
    create_trace_file_for_simulator(multi_traces, multi_trace_path)
    logger.info(f"✅ Generated {len(multi_traces)} multi-turn requests")

    logger.info("\n✅ Trace generation complete!")

    return {
        'single_turn': single_trace_path,
        'multi_turn': multi_trace_path,
    }


def run_experiment(
    trace_file: str,
    model_name: str,
    block_size: int,
    output_file: str,
    no_prefix_caching: bool = False,
) -> Dict[str, Any]:
    """
    Run experiment using run_milestone2_correct_approach.py

    Returns:
        Statistics dictionary
    """
    script_path = VLLM_ROOT / "sim" / "run_milestone2_correct_approach.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--trace-file", trace_file,
        "--model", model_name,
        "--block-size", str(block_size),
        "--output-file", output_file,
    ]

    if no_prefix_caching:
        cmd.append("--no-prefix-caching")

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Experiment failed!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError("Experiment failed")

    # Print the output (which contains the summary)
    print(result.stdout)

    # Load and return stats
    with open(output_file, 'r') as f:
        stats = json.load(f)

    return stats


def visualize_results(
    multi_stats: Dict[str, Any],
    single_stats: Dict[str, Any],
    output_dir: str,
):
    """
    Create visualization comparing multi-turn vs single-turn
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Visualizing Results")
    logger.info("="*60)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Milestone 2: Prefix Sharing Effectiveness\n(vLLM Real Block Manager)',
                 fontsize=16, fontweight='bold')

    # 1. Overall Sharing Fraction comparison
    ax1 = axes[0, 0]
    categories = ['Multi-turn', 'Single-turn']
    sharing_fractions = [
        multi_stats['overall_sharing_fraction'] * 100,
        single_stats['overall_sharing_fraction'] * 100,
    ]
    colors = ['#2ecc71', '#e74c3c']
    bars1 = ax1.bar(categories, sharing_fractions, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Sharing Fraction (%)', fontsize=12)
    ax1.set_title('Overall Block Reuse Rate', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(sharing_fractions) * 1.2)

    # Add value labels on bars
    for bar, val in zip(bars1, sharing_fractions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.grid(axis='y', alpha=0.3)

    # 2. Cache Hit Rate comparison
    ax2 = axes[0, 1]
    cache_hit_rates = [
        multi_stats['final_cache_hit_rate'] * 100,
        single_stats['final_cache_hit_rate'] * 100,
    ]
    bars2 = ax2.bar(categories, cache_hit_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax2.set_title('vLLM Cache Hit Rate', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(cache_hit_rates) * 1.2 if max(cache_hit_rates) > 0 else 10)

    for bar, val in zip(bars2, cache_hit_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.grid(axis='y', alpha=0.3)

    # 3. Block allocation breakdown
    ax3 = axes[1, 0]

    multi_reused = multi_stats['total_blocks_reused']
    multi_new = multi_stats['total_blocks_newly_allocated']
    single_reused = single_stats['total_blocks_reused']
    single_new = single_stats['total_blocks_newly_allocated']

    x = np.arange(len(categories))
    width = 0.35

    bars3a = ax3.bar(x, [multi_reused, single_reused], width,
                     label='Blocks Reused', color='#3498db', alpha=0.7, edgecolor='black')
    bars3b = ax3.bar(x, [multi_new, single_new], width,
                     bottom=[multi_reused, single_reused],
                     label='Blocks Newly Allocated', color='#e67e22', alpha=0.7, edgecolor='black')

    ax3.set_ylabel('Number of Blocks', fontsize=12)
    ax3.set_title('Block Allocation Breakdown', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (cat, reused, new) in enumerate(zip(categories,
                                                 [multi_reused, single_reused],
                                                 [multi_new, single_new])):
        total = reused + new
        reused_pct = reused / total * 100 if total > 0 else 0
        ax3.text(i, reused/2, f'{reused_pct:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # 4. Per-request sharing fraction distribution
    ax4 = axes[1, 1]

    multi_per_request = [m['sharing_fraction'] * 100 for m in multi_stats['request_metrics']]
    single_per_request = [m['sharing_fraction'] * 100 for m in single_stats['request_metrics']]

    bp = ax4.boxplot([multi_per_request, single_per_request],
                      labels=categories,
                      patch_artist=True,
                      showmeans=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Sharing Fraction per Request (%)', fontsize=12)
    ax4.set_title('Per-Request Sharing Distribution', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "milestone2_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved visualization to {output_path}")

    plt.close()

    # Create summary comparison table
    create_summary_table(multi_stats, single_stats, output_dir)


def create_summary_table(
    multi_stats: Dict[str, Any],
    single_stats: Dict[str, Any],
    output_dir: str,
):
    """
    Create a markdown summary table
    """
    output_path = os.path.join(output_dir, "milestone2_summary.md")

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
        improvement = multi_cache / single_cache if single_cache > 0 else float('inf')
        f.write(f"| Cache Hit Rate | {multi_cache:.2f}% | {single_cache:.2f}% | {improvement:.0f}x |\n")

        # Blocks reused
        multi_reused = multi_stats['total_blocks_reused']
        single_reused = single_stats['total_blocks_reused']
        improvement = multi_reused / single_reused if single_reused > 0 else float('inf')
        f.write(f"| Blocks Reused | {multi_reused:,} | {single_reused:,} | {improvement:.1f}x |\n")

        # Blocks newly allocated
        multi_new = multi_stats['total_blocks_newly_allocated']
        single_new = single_stats['total_blocks_newly_allocated']
        f.write(f"| Blocks Newly Allocated | {multi_new:,} | {single_new:,} | - |\n")

        # Total requests
        f.write(f"| Total Requests | {multi_stats['total_requests']} | {single_stats['total_requests']} | - |\n")

        # Total tokens
        f.write(f"| Total Tokens | {multi_stats['total_tokens']:,} | {single_stats['total_tokens']:,} | - |\n")

        f.write("\n## Key Findings\n\n")
        f.write(f"1. **Multi-turn conversations benefit significantly from prefix caching**\n")
        f.write(f"   - {multi_sharing:.1f}% of blocks are reused (vs {single_sharing:.1f}% for single-turn)\n")
        f.write(f"   - Cache hit rate: {multi_cache:.1f}%\n\n")

        f.write(f"2. **Single-turn conversations have minimal prefix sharing**\n")
        f.write(f"   - Only {single_sharing:.1f}% of blocks are reused\n")
        f.write(f"   - Cache hit rate: {single_cache:.2f}%\n\n")

        f.write(f"3. **Multi-turn is {improvement:.0f}x more effective**\n")
        f.write(f"   - Saves {multi_reused:,} block allocations through reuse\n")
        f.write(f"   - Reduces memory allocation overhead significantly\n\n")

        f.write("## Implementation Details\n\n")
        f.write("- **Block Manager**: vLLM SelfAttnBlockSpaceManager (real implementation)\n")
        f.write("- **Prefix Caching**: vLLM PrefixCachingBlockAllocator with LRU eviction\n")
        f.write(f"- **Block Size**: 16 tokens\n")
        f.write(f"- **Model**: facebook/opt-125m tokenizer\n")
        f.write(f"- **Requests**: {multi_stats['total_requests']} per experiment\n")

    logger.info(f"✅ Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Milestone 2 Complete Pipeline: Generate traces, run experiments, visualize results'
    )

    # Data parameters
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to ShareGPT JSONL file'
    )
    parser.add_argument(
        '--max-conversations',
        type=int,
        default=100,
        help='Maximum number of conversations to process (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='milestone2_results',
        help='Output directory for results (default: milestone2_results)'
    )

    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-125m',
        help='Model name for tokenizer (default: facebook/opt-125m)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=16,
        help='Block size in tokens (default: 16)'
    )

    # Trace generation parameters
    parser.add_argument(
        '--arrival-rate',
        type=float,
        default=2.0,
        help='Request arrival rate (req/s) for Poisson process (default: 2.0)'
    )

    # Control flags
    parser.add_argument(
        '--skip-trace-generation',
        action='store_true',
        help='Skip trace generation (use existing traces)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )

    args = parser.parse_args()

    try:
        logger.info("\n" + "="*60)
        logger.info("MILESTONE 2: COMPLETE PIPELINE")
        logger.info("="*60)
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Block size: {args.block_size}")
        logger.info(f"Max conversations: {args.max_conversations}")

        # Step 1: Generate traces (if not skipped)
        if args.skip_trace_generation:
            logger.info("\n⏭️  Skipping trace generation (using existing traces)")
            trace_files = {
                'single_turn': os.path.join(args.output_dir, "single_turn_trace.jsonl"),
                'multi_turn': os.path.join(args.output_dir, "multi_turn_trace.jsonl"),
            }
        else:
            trace_files = generate_traces(
                data_path=args.data_path,
                output_dir=args.output_dir,
                max_conversations=args.max_conversations,
                model_name=args.model,
                arrival_rate=args.arrival_rate,
            )

        # Step 2: Run experiments
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Running Experiments (Task 2)")
        logger.info("="*60)

        # Multi-turn experiment
        logger.info("\n📊 Running multi-turn experiment...")
        multi_output = os.path.join(args.output_dir, "milestone2_multi_stats.json")
        multi_stats = run_experiment(
            trace_file=trace_files['multi_turn'],
            model_name=args.model,
            block_size=args.block_size,
            output_file=multi_output,
            no_prefix_caching=False,
        )
        logger.info("✅ Multi-turn experiment complete")

        # Single-turn experiment
        logger.info("\n📊 Running single-turn experiment...")
        single_output = os.path.join(args.output_dir, "milestone2_single_stats.json")
        single_stats = run_experiment(
            trace_file=trace_files['single_turn'],
            model_name=args.model,
            block_size=args.block_size,
            output_file=single_output,
            no_prefix_caching=False,
        )
        logger.info("✅ Single-turn experiment complete")

        # Step 3: Visualize results
        if not args.skip_visualization:
            visualize_results(multi_stats, single_stats, args.output_dir)
        else:
            logger.info("\n⏭️  Skipping visualization")

        # Final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 MILESTONE 2 COMPLETE!")
        logger.info("="*60)
        logger.info(f"\n📁 Results saved to: {args.output_dir}/")
        logger.info(f"  - Traces: {trace_files['multi_turn']}, {trace_files['single_turn']}")
        logger.info(f"  - Stats: {multi_output}, {single_output}")
        if not args.skip_visualization:
            logger.info(f"  - Visualization: {args.output_dir}/milestone2_comparison.png")
            logger.info(f"  - Summary: {args.output_dir}/milestone2_summary.md")

        logger.info(f"\n✨ Key Result: Multi-turn achieves {multi_stats['overall_sharing_fraction']*100:.1f}% sharing fraction")
        logger.info(f"   vs Single-turn {single_stats['overall_sharing_fraction']*100:.1f}% sharing fraction")
        logger.info(f"   ({multi_stats['overall_sharing_fraction']/single_stats['overall_sharing_fraction']:.1f}x improvement!)")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
