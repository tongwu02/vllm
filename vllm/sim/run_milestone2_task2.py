"""
Milestone 2 Task 2: Replay ShareGPT trace and measure prefix sharing

This script:
1. Loads ShareGPT data
2. Generates single-turn and multi-turn traces
3. Simulates the serving process with prefix sharing
4. Collects and visualizes metrics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add vllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client_simulator import (
    ShareGPTLoader,
    ChatTemplateFormatter,
    RequestGenerator,
    create_trace_file_for_simulator
)
from prefix_sharing_metrics import (
    PrefixSharingMetricsCollector,
    MockBlockManagerMetricsIntegration
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToyTokenizer:
    """Simple tokenizer for testing."""
    def encode(self, text: str, add_special_tokens: bool = False):
        # Simple character-based tokenization
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True):
        return "".join(chr(i) if i < 128 else '?' for i in ids)


def run_simulation(
    traces,
    mode: str,
    output_dir: str,
    block_size: int = 16
):
    """
    Run the simulation with prefix sharing metrics collection.

    Args:
        traces: List of RequestTrace objects
        mode: 'single' or 'multi' turn
        output_dir: Directory to save results
        block_size: Block size for KV cache
    """
    logger.info(f"Running {mode}-turn simulation with {len(traces)} requests")

    # Initialize metrics collector
    metrics_collector = PrefixSharingMetricsCollector()
    block_manager = MockBlockManagerMetricsIntegration(
        metrics_collector,
        block_size=block_size
    )

    # Simple tokenizer
    tokenizer = ToyTokenizer()

    # Simulate each request
    for i, trace in enumerate(traces):
        if i % 100 == 0:
            logger.info(f"Processing request {i}/{len(traces)}")

        # Tokenize the prompt
        prompt_tokens = tokenizer.encode(trace.prompt)

        # Allocate blocks (this will detect sharing)
        total_blocks, shared_blocks, shared_tokens = \
            block_manager.allocate_blocks_for_request(
                trace.request_id,
                prompt_tokens
            )

        # Record completion metrics
        metrics_collector.on_request_complete(
            trace.request_id,
            total_prompt_tokens=len(prompt_tokens),
            shared_prefix_tokens=shared_tokens
        )

    # Get statistics
    stats = metrics_collector.get_statistics()

    # Print summary
    metrics_collector.print_summary()

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON stats
    stats_file = os.path.join(output_dir, f"{mode}_turn_stats.json")
    metrics_collector.save_to_json(stats_file)

    # Save detailed CSV for analysis
    save_detailed_metrics(metrics_collector, mode, output_dir)

    return stats


def save_detailed_metrics(
    metrics_collector: PrefixSharingMetricsCollector,
    mode: str,
    output_dir: str
):
    """Save detailed metrics to CSV files for further analysis."""
    import csv

    # Save per-request metrics
    request_csv = os.path.join(output_dir, f"{mode}_turn_request_metrics.csv")
    with open(request_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'request_id',
            'total_tokens',
            'shared_tokens',
            'sharing_fraction',
            'blocks_accessed',
            'blocks_reused'
        ])

        for req_id, metrics in metrics_collector.request_metrics.items():
            sharing_fraction = (
                metrics.shared_tokens / metrics.total_tokens
                if metrics.total_tokens > 0 else 0
            )
            writer.writerow([
                req_id,
                metrics.total_tokens,
                metrics.shared_tokens,
                sharing_fraction,
                metrics.num_blocks_accessed,
                metrics.num_blocks_reused
            ])

    # Save block metrics
    block_csv = os.path.join(output_dir, f"{mode}_turn_block_metrics.csv")
    with open(block_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'block_id',
            'hit_count',
            'first_access_time',
            'last_access_time',
            'avg_reuse_gap'
        ])

        for block_id, block_info in metrics_collector.block_info.items():
            avg_gap = (
                sum(block_info.reuse_gaps) / len(block_info.reuse_gaps)
                if block_info.reuse_gaps else 0
            )
            writer.writerow([
                block_id,
                block_info.access_count,
                block_info.first_access_time,
                block_info.last_access_time,
                avg_gap
            ])

    logger.info(f"Saved detailed metrics to {output_dir}")


def visualize_results(stats_single, stats_multi, output_dir: str):
    """
    Create visualizations of the results.

    Args:
        stats_single: Statistics from single-turn run
        stats_multi: Statistics from multi-turn run
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. CDF of sharing fractions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (stats, label) in enumerate([
        (stats_single, 'Single-turn'),
        (stats_multi, 'Multi-turn')
    ]):
        fractions = sorted(stats['sharing_fraction']['distribution'])
        if not fractions:
            continue

        y = np.arange(1, len(fractions) + 1) / len(fractions)
        axes[idx].plot(fractions, y, linewidth=2)
        axes[idx].set_xlabel('Sharing Fraction')
        axes[idx].set_ylabel('CDF')
        axes[idx].set_title(f'{label}: Sharing Fraction CDF')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sharing_fraction_cdf.png'), dpi=150)
    plt.close()

    # 2. Block hit count distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (stats, label) in enumerate([
        (stats_single, 'Single-turn'),
        (stats_multi, 'Multi-turn')
    ]):
        hit_counts = stats['block_hits']['distribution']
        if not hit_counts:
            continue

        axes[idx].hist(hit_counts, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Hit Count')
        axes[idx].set_ylabel('Number of Blocks')
        axes[idx].set_title(f'{label}: Block Hit Count Distribution')
        axes[idx].set_yscale('log')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'block_hit_distribution.png'), dpi=150)
    plt.close()

    # 3. Reuse gap distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (stats, label) in enumerate([
        (stats_single, 'Single-turn'),
        (stats_multi, 'Multi-turn')
    ]):
        gaps = sorted(stats['reuse_gaps']['distribution'])
        if not gaps:
            continue

        y = np.arange(1, len(gaps) + 1) / len(gaps)
        axes[idx].plot(gaps, y, linewidth=2)
        axes[idx].set_xlabel('Reuse Gap (seconds)')
        axes[idx].set_ylabel('CDF')
        axes[idx].set_title(f'{label}: Reuse Gap CDF')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reuse_gap_cdf.png'), dpi=150)
    plt.close()

    # 4. Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Mean Sharing\nFraction', 'Median Block\nHit Count', 'Block Reuse\nRate']
    single_vals = [
        stats_single['sharing_fraction']['mean'],
        stats_single['block_hits']['median'],
        stats_single['total_blocks_reused'] /
            (stats_single['total_blocks_allocated'] + stats_single['total_blocks_reused'])
            if stats_single['total_blocks_allocated'] > 0 else 0
    ]
    multi_vals = [
        stats_multi['sharing_fraction']['mean'],
        stats_multi['block_hits']['median'],
        stats_multi['total_blocks_reused'] /
            (stats_multi['total_blocks_allocated'] + stats_multi['total_blocks_reused'])
            if stats_multi['total_blocks_allocated'] > 0 else 0
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, single_vals, width, label='Single-turn', alpha=0.8)
    ax.bar(x + width/2, multi_vals, width, label='Multi-turn', alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title('Single-turn vs Multi-turn Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run Milestone 2 Task 2')
    parser.add_argument(
        '--data-path',
        type=str,
        default='vllm/ShareGPTData.jsonl',
        help='Path to ShareGPT data file'
    )
    parser.add_argument(
        '--max-conversations',
        type=int,
        default=1000,
        help='Maximum number of conversations to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='milestone2_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=16,
        help='Block size for KV cache'
    )
    parser.add_argument(
        '--arrival-rate',
        type=float,
        default=2.0,
        help='Request arrival rate (requests/second)'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization (if matplotlib not available)'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Milestone 2 Task 2: ShareGPT Replay with Prefix Sharing")
    logger.info("="*60)

    # Load ShareGPT data
    logger.info(f"Loading ShareGPT data from {args.data_path}")
    loader = ShareGPTLoader(args.data_path, max_conversations=args.max_conversations)
    conversations = loader.get_conversations()
    logger.info(f"Loaded {len(conversations)} conversations")

    # Create formatter
    formatter = ChatTemplateFormatter()

    # Create request generator
    generator = RequestGenerator(
        conversations,
        arrival_rate=args.arrival_rate,
        use_poisson=True
    )

    # Generate single-turn traces
    logger.info("\n" + "="*60)
    logger.info("Generating single-turn traces...")
    logger.info("="*60)
    single_turn_traces = generator.generate_single_turn_traces(formatter)

    # Generate multi-turn traces
    logger.info("\n" + "="*60)
    logger.info("Generating multi-turn traces...")
    logger.info("="*60)
    multi_turn_traces = generator.generate_multi_turn_traces(formatter, turn_delay=1.0)

    # Run simulations
    logger.info("\n" + "="*60)
    logger.info("Running single-turn simulation...")
    logger.info("="*60)
    stats_single = run_simulation(
        single_turn_traces,
        'single',
        args.output_dir,
        block_size=args.block_size
    )

    logger.info("\n" + "="*60)
    logger.info("Running multi-turn simulation...")
    logger.info("="*60)
    stats_multi = run_simulation(
        multi_turn_traces,
        'multi',
        args.output_dir,
        block_size=args.block_size
    )

    # Create visualizations
    if not args.skip_visualization:
        logger.info("\n" + "="*60)
        logger.info("Creating visualizations...")
        logger.info("="*60)
        visualize_results(stats_single, stats_multi, args.output_dir)

    # Create trace files for use with vLLM simulator
    logger.info("\n" + "="*60)
    logger.info("Creating trace files for vLLM simulator...")
    logger.info("="*60)
    create_trace_file_for_simulator(
        single_turn_traces,
        os.path.join(args.output_dir, 'single_turn_trace.jsonl')
    )
    create_trace_file_for_simulator(
        multi_turn_traces,
        os.path.join(args.output_dir, 'multi_turn_trace.jsonl')
    )

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
