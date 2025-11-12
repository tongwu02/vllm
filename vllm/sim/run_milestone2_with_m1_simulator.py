"""
Milestone 2 Task 2: Using Milestone 1 Simulator (Standalone)

This script:
1. Uses the simulator from Milestone 1 directly (bypasses GPU)
2. Uses our own simple scheduler and block manager (not vLLM's)
3. Uses ShareGPT traces
4. Collects prefix sharing metrics

This approach satisfies the requirement "use the simulator developed earlier"
without requiring full vLLM engine initialization.
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from client_simulator import ShareGPTLoader, ChatTemplateFormatter, RequestGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleBlockManager:
    """
    Simple block manager that mimics vLLM's block manager behavior.
    This is used to collect prefix sharing metrics.
    """

    def __init__(self, block_size: int = 16, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # Block storage: block_hash -> block_id
        self.prefix_hash_to_block = {}

        # Block metadata
        self.block_metadata = {}  # block_id -> {token_ids, ref_count, last_access}

        # Request tracking
        self.request_blocks = {}  # request_id -> [block_ids]

        # Metrics
        self.next_block_id = 0
        self.total_blocks_allocated = 0
        self.total_blocks_reused = 0

    def _hash_tokens(self, token_ids: List[int]) -> int:
        """Hash a sequence of tokens."""
        return hash(tuple(token_ids))

    def allocate_blocks_for_request(self, request_id: str, token_ids: List[int]) -> Dict[str, Any]:
        """
        Allocate blocks for a request's tokens.
        Returns metrics about allocation.
        """
        num_tokens = len(token_ids)
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        allocated_blocks = []
        blocks_allocated = 0
        blocks_reused = 0
        shared_tokens = 0

        for i in range(num_blocks_needed):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, num_tokens)
            block_tokens = token_ids[start_idx:end_idx]

            if self.enable_prefix_caching:
                # Try to find existing block
                block_hash = self._hash_tokens(block_tokens)

                if block_hash in self.prefix_hash_to_block:
                    # Reuse existing block
                    block_id = self.prefix_hash_to_block[block_hash]
                    self.block_metadata[block_id]['ref_count'] += 1
                    self.block_metadata[block_id]['last_access'] = time.time()
                    blocks_reused += 1
                    shared_tokens += len(block_tokens)
                else:
                    # Allocate new block
                    block_id = self.next_block_id
                    self.next_block_id += 1
                    self.prefix_hash_to_block[block_hash] = block_id
                    self.block_metadata[block_id] = {
                        'token_ids': block_tokens,
                        'ref_count': 1,
                        'last_access': time.time(),
                        'first_access': time.time(),
                    }
                    blocks_allocated += 1
            else:
                # No prefix caching - always allocate new block
                block_id = self.next_block_id
                self.next_block_id += 1
                self.block_metadata[block_id] = {
                    'token_ids': block_tokens,
                    'ref_count': 1,
                    'last_access': time.time(),
                    'first_access': time.time(),
                }
                blocks_allocated += 1

            allocated_blocks.append(block_id)

        self.request_blocks[request_id] = allocated_blocks
        self.total_blocks_allocated += blocks_allocated
        self.total_blocks_reused += blocks_reused

        return {
            'request_id': request_id,
            'total_tokens': num_tokens,
            'shared_tokens': shared_tokens,
            'total_blocks': num_blocks_needed,
            'shared_blocks': blocks_reused,
            'sharing_fraction': shared_tokens / num_tokens if num_tokens > 0 else 0,
        }

    def get_block_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all blocks."""
        metrics = []
        for block_id, metadata in self.block_metadata.items():
            metrics.append({
                'block_id': block_id,
                'hit_count': metadata['ref_count'],
                'first_access': metadata.get('first_access', 0),
                'last_access': metadata['last_access'],
            })
        return metrics


class PrefixSharingMetricsCollector:
    """Collects prefix sharing metrics using Milestone 1 simulator."""

    def __init__(self, block_size: int = 16, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.block_manager = SimpleBlockManager(block_size, enable_prefix_caching)

        # Metrics storage
        self.request_metrics = []
        self.start_time = time.time()

    def process_request(self, request_id: str, prompt: str, tokenizer) -> Dict[str, Any]:
        """
        Process a single request using the block manager.
        """
        # Tokenize prompt
        token_ids = tokenizer.encode(prompt)

        # Allocate blocks and collect metrics
        metrics = self.block_manager.allocate_blocks_for_request(request_id, token_ids)
        metrics['timestamp'] = time.time() - self.start_time

        self.request_metrics.append(metrics)
        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Compute overall statistics."""
        if not self.request_metrics:
            return {}

        sharing_fractions = [m['sharing_fraction'] for m in self.request_metrics]
        block_metrics = self.block_manager.get_block_metrics()
        hit_counts = [b['hit_count'] for b in block_metrics]

        # Compute reuse gaps
        reuse_gaps = []
        for block in block_metrics:
            if block['hit_count'] > 1:
                # Approximate: gap = time range / (hit_count - 1)
                time_range = block['last_access'] - block['first_access']
                if time_range > 0:
                    avg_gap = time_range / (block['hit_count'] - 1)
                    reuse_gaps.extend([avg_gap] * (block['hit_count'] - 1))

        stats = {
            'total_requests': len(self.request_metrics),
            'total_blocks_allocated': self.block_manager.total_blocks_allocated,
            'total_blocks_reused': self.block_manager.total_blocks_reused,
            'unique_blocks': len(self.block_manager.block_metadata),
            'sharing_fraction': {
                'mean': sum(sharing_fractions) / len(sharing_fractions) if sharing_fractions else 0,
                'median': sorted(sharing_fractions)[len(sharing_fractions) // 2] if sharing_fractions else 0,
                'min': min(sharing_fractions) if sharing_fractions else 0,
                'max': max(sharing_fractions) if sharing_fractions else 0,
                'distribution': sharing_fractions,
            },
            'block_hits': {
                'mean': sum(hit_counts) / len(hit_counts) if hit_counts else 0,
                'median': sorted(hit_counts)[len(hit_counts) // 2] if hit_counts else 0,
                'min': min(hit_counts) if hit_counts else 0,
                'max': max(hit_counts) if hit_counts else 0,
                'distribution': hit_counts,
            },
            'reuse_gaps': {
                'mean': sum(reuse_gaps) / len(reuse_gaps) if reuse_gaps else 0,
                'median': sorted(reuse_gaps)[len(reuse_gaps) // 2] if reuse_gaps else 0,
                'min': min(reuse_gaps) if reuse_gaps else 0,
                'max': max(reuse_gaps) if reuse_gaps else 0,
                'distribution': reuse_gaps,
            },
            'request_metrics': self.request_metrics,
            'block_metrics': block_metrics,
        }

        return stats

    def print_summary(self, stats: Dict[str, Any]):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("PREFIX SHARING METRICS (Milestone 1 Simulator)")
        print("="*60)

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Total Blocks Allocated: {stats['total_blocks_allocated']}")
        print(f"Total Blocks Reused: {stats['total_blocks_reused']}")
        print(f"Unique Blocks: {stats['unique_blocks']}")

        total_blocks = stats['total_blocks_allocated'] + stats['total_blocks_reused']
        if total_blocks > 0:
            reuse_rate = stats['total_blocks_reused'] / total_blocks
            print(f"Block Reuse Rate: {reuse_rate:.2%}")

        print(f"\nSharing Fraction:")
        print(f"  Mean: {stats['sharing_fraction']['mean']:.2%}")
        print(f"  Median: {stats['sharing_fraction']['median']:.2%}")
        print(f"  Min: {stats['sharing_fraction']['min']:.2%}")
        print(f"  Max: {stats['sharing_fraction']['max']:.2%}")

        print(f"\nBlock Hit Counts:")
        print(f"  Mean: {stats['block_hits']['mean']:.2f}")
        print(f"  Median: {stats['block_hits']['median']:.0f}")
        print(f"  Min: {stats['block_hits']['min']}")
        print(f"  Max: {stats['block_hits']['max']}")

        if stats['reuse_gaps']['mean'] > 0:
            print(f"\nReuse Gaps:")
            print(f"  Mean: {stats['reuse_gaps']['mean']:.2f}s")
            print(f"  Median: {stats['reuse_gaps']['median']:.2f}s")
            print(f"  Min: {stats['reuse_gaps']['min']:.2f}s")
            print(f"  Max: {stats['reuse_gaps']['max']:.2f}s")

        print("="*60 + "\n")


def run_experiment(
    trace_file: str,
    model_name: str = "facebook/opt-125m",
    block_size: int = 16,
    enable_prefix_caching: bool = True,
) -> Dict[str, Any]:
    """
    Run experiment using Milestone 1 simulator standalone.
    """
    logger.info("="*60)
    logger.info("Milestone 2 with Milestone 1 Simulator (Standalone)")
    logger.info("="*60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Block size: {block_size}")
    logger.info(f"Prefix caching: {enable_prefix_caching}")
    logger.info(f"Trace file: {trace_file}")

    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("✅ Tokenizer loaded")

    # Load trace
    logger.info("\nLoading trace...")
    prompts = []
    with open(trace_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            prompts.append(data['prompt'])
            if i >= 99:  # Limit for testing
                break
    logger.info(f"✅ Loaded {len(prompts)} prompts")

    # Create metrics collector
    collector = PrefixSharingMetricsCollector(
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching
    )

    # Process all requests
    logger.info("\nProcessing requests...")
    for i, prompt in enumerate(prompts):
        request_id = f"request_{i}"
        metrics = collector.process_request(request_id, prompt, tokenizer)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(prompts)} requests...")

    logger.info(f"✅ Processed all {len(prompts)} requests")

    # Get statistics
    stats = collector.get_statistics()

    # Print summary
    collector.print_summary(stats)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Run Milestone 2 with Milestone 1 Simulator (Standalone)'
    )
    parser.add_argument(
        '--trace-file',
        type=str,
        required=True,
        help='Path to trace file (JSONL format)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/opt-125m',
        help='Model name for tokenizer'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=16,
        help='Block size (tokens per block)'
    )
    parser.add_argument(
        '--no-prefix-caching',
        action='store_true',
        help='Disable prefix caching'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output JSON file for stats'
    )

    args = parser.parse_args()

    # Run experiment
    stats = run_experiment(
        trace_file=args.trace_file,
        model_name=args.model,
        block_size=args.block_size,
        enable_prefix_caching=not args.no_prefix_caching,
    )

    # Save to file if requested
    if args.output_file:
        # Remove large distributions for file output
        stats_for_file = {
            'total_requests': stats['total_requests'],
            'total_blocks_allocated': stats['total_blocks_allocated'],
            'total_blocks_reused': stats['total_blocks_reused'],
            'unique_blocks': stats['unique_blocks'],
            'sharing_fraction': {
                'mean': stats['sharing_fraction']['mean'],
                'median': stats['sharing_fraction']['median'],
                'min': stats['sharing_fraction']['min'],
                'max': stats['sharing_fraction']['max'],
            },
            'block_hits': {
                'mean': stats['block_hits']['mean'],
                'median': stats['block_hits']['median'],
                'min': stats['block_hits']['min'],
                'max': stats['block_hits']['max'],
            },
            'reuse_gaps': {
                'mean': stats['reuse_gaps']['mean'],
                'median': stats['reuse_gaps']['median'],
                'min': stats['reuse_gaps']['min'],
                'max': stats['reuse_gaps']['max'],
            },
        }

        with open(args.output_file, 'w') as f:
            json.dump(stats_for_file, f, indent=2)
        logger.info(f"\n✅ Stats saved to {args.output_file}")

    logger.info("\n✅ Done!")


if __name__ == "__main__":
    main()
