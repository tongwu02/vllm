"""
Prefix Sharing Metrics Collection

This module collects and analyzes prefix sharing statistics:
1. Fraction of each request that benefits from prefix sharing
2. Number of hits per cache block
3. Time gap between reuses of each cache block
4. Additional metrics for analyzing prefix sharing effectiveness
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BlockAccessInfo:
    """Information about a cache block access."""
    block_id: int
    access_count: int = 0
    first_access_time: Optional[float] = None
    last_access_time: Optional[float] = None
    access_times: List[float] = field(default_factory=list)
    reuse_gaps: List[float] = field(default_factory=list)  # Time between consecutive accesses


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    total_tokens: int = 0  # Total prompt tokens
    shared_tokens: int = 0  # Tokens that benefited from prefix sharing
    num_blocks_accessed: int = 0
    num_blocks_reused: int = 0  # Blocks that were already in cache
    creation_time: float = 0.0


class PrefixSharingMetricsCollector:
    """
    Collects metrics about prefix sharing effectiveness.

    This integrates with vLLM's block manager to track:
    - Which blocks are reused
    - How many times each block is accessed
    - Time gaps between reuses
    """

    def __init__(self):
        self.block_info: Dict[int, BlockAccessInfo] = {}
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.start_time = time.time()

        # Additional statistics
        self.total_blocks_allocated = 0
        self.total_blocks_reused = 0
        self.total_requests = 0

    def on_block_access(
        self,
        block_id: int,
        request_id: str,
        is_new_allocation: bool = True
    ):
        """
        Called when a block is accessed.

        Args:
            block_id: The ID of the block being accessed
            request_id: The request accessing this block
            is_new_allocation: True if this is a new allocation, False if reusing existing block
        """
        current_time = time.time() - self.start_time

        # Update block info
        if block_id not in self.block_info:
            self.block_info[block_id] = BlockAccessInfo(
                block_id=block_id,
                first_access_time=current_time
            )

        block = self.block_info[block_id]
        block.access_count += 1

        # Calculate reuse gap if this is a reuse
        if block.last_access_time is not None:
            gap = current_time - block.last_access_time
            block.reuse_gaps.append(gap)

        block.last_access_time = current_time
        block.access_times.append(current_time)

        # Update request metrics
        if request_id not in self.request_metrics:
            self.request_metrics[request_id] = RequestMetrics(
                request_id=request_id,
                creation_time=current_time
            )

        req_metrics = self.request_metrics[request_id]
        req_metrics.num_blocks_accessed += 1

        if not is_new_allocation:
            req_metrics.num_blocks_reused += 1
            self.total_blocks_reused += 1
        else:
            self.total_blocks_allocated += 1

    def on_request_complete(
        self,
        request_id: str,
        total_prompt_tokens: int,
        shared_prefix_tokens: int
    ):
        """
        Called when a request completes.

        Args:
            request_id: The completed request ID
            total_prompt_tokens: Total number of tokens in the prompt
            shared_prefix_tokens: Number of tokens that were shared (reused from cache)
        """
        if request_id not in self.request_metrics:
            self.request_metrics[request_id] = RequestMetrics(request_id=request_id)

        req_metrics = self.request_metrics[request_id]
        req_metrics.total_tokens = total_prompt_tokens
        req_metrics.shared_tokens = shared_prefix_tokens
        self.total_requests += 1

    def get_request_sharing_fraction(self, request_id: str) -> float:
        """Get the fraction of tokens that benefited from sharing for a request."""
        if request_id not in self.request_metrics:
            return 0.0

        metrics = self.request_metrics[request_id]
        if metrics.total_tokens == 0:
            return 0.0

        return metrics.shared_tokens / metrics.total_tokens

    def get_all_sharing_fractions(self) -> List[float]:
        """Get sharing fractions for all requests."""
        fractions = []
        for req_id in self.request_metrics:
            fraction = self.get_request_sharing_fraction(req_id)
            fractions.append(fraction)
        return fractions

    def get_block_hit_counts(self) -> List[int]:
        """Get hit counts for all blocks."""
        return [block.access_count for block in self.block_info.values()]

    def get_reuse_gaps(self) -> List[float]:
        """Get all reuse gaps (time between consecutive accesses)."""
        all_gaps = []
        for block in self.block_info.values():
            all_gaps.extend(block.reuse_gaps)
        return all_gaps

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about prefix sharing."""
        sharing_fractions = self.get_all_sharing_fractions()
        hit_counts = self.get_block_hit_counts()
        reuse_gaps = self.get_reuse_gaps()

        stats = {
            'total_requests': self.total_requests,
            'total_blocks_allocated': self.total_blocks_allocated,
            'total_blocks_reused': self.total_blocks_reused,
            'unique_blocks': len(self.block_info),
            'sharing_fraction': {
                'mean': sum(sharing_fractions) / len(sharing_fractions) if sharing_fractions else 0,
                'median': sorted(sharing_fractions)[len(sharing_fractions) // 2] if sharing_fractions else 0,
                'min': min(sharing_fractions) if sharing_fractions else 0,
                'max': max(sharing_fractions) if sharing_fractions else 0,
                'distribution': sharing_fractions
            },
            'block_hits': {
                'mean': sum(hit_counts) / len(hit_counts) if hit_counts else 0,
                'median': sorted(hit_counts)[len(hit_counts) // 2] if hit_counts else 0,
                'min': min(hit_counts) if hit_counts else 0,
                'max': max(hit_counts) if hit_counts else 0,
                'distribution': hit_counts
            },
            'reuse_gaps': {
                'mean': sum(reuse_gaps) / len(reuse_gaps) if reuse_gaps else 0,
                'median': sorted(reuse_gaps)[len(reuse_gaps) // 2] if reuse_gaps else 0,
                'min': min(reuse_gaps) if reuse_gaps else 0,
                'max': max(reuse_gaps) if reuse_gaps else 0,
                'distribution': reuse_gaps
            }
        }

        return stats

    def print_summary(self):
        """Print a summary of the collected metrics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("PREFIX SHARING METRICS SUMMARY")
        print("="*60)

        print(f"\nTotal Requests: {stats['total_requests']}")
        print(f"Total Blocks Allocated: {stats['total_blocks_allocated']}")
        print(f"Total Blocks Reused: {stats['total_blocks_reused']}")
        print(f"Unique Blocks: {stats['unique_blocks']}")

        if stats['total_blocks_allocated'] > 0:
            reuse_rate = stats['total_blocks_reused'] / (stats['total_blocks_allocated'] + stats['total_blocks_reused'])
            print(f"Block Reuse Rate: {reuse_rate:.2%}")

        print(f"\nSharing Fraction (per request):")
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
            print(f"\nReuse Gaps (seconds):")
            print(f"  Mean: {stats['reuse_gaps']['mean']:.2f}s")
            print(f"  Median: {stats['reuse_gaps']['median']:.2f}s")
            print(f"  Min: {stats['reuse_gaps']['min']:.2f}s")
            print(f"  Max: {stats['reuse_gaps']['max']:.2f}s")

        print("="*60 + "\n")

    def save_to_json(self, output_path: str):
        """Save statistics to a JSON file."""
        import json
        stats = self.get_statistics()

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")


class MockBlockManagerMetricsIntegration:
    """
    Mock integration with vLLM's block manager for testing.

    In production, this would hook into vLLM's actual block manager.
    """

    def __init__(self, metrics_collector: PrefixSharingMetricsCollector, block_size: int = 16):
        self.metrics_collector = metrics_collector
        self.block_size = block_size
        self.token_to_block_cache: Dict[str, List[int]] = {}  # token_hash -> block_ids
        self.next_block_id = 0

    def allocate_blocks_for_request(
        self,
        request_id: str,
        prompt_tokens: List[int]
    ) -> tuple:
        """
        Simulate block allocation for a request.

        Returns:
            (total_blocks, shared_blocks, shared_tokens)
        """
        # Create a hash of the prompt prefix for sharing detection
        num_blocks_needed = (len(prompt_tokens) + self.block_size - 1) // self.block_size

        shared_blocks = 0
        shared_tokens = 0
        allocated_blocks = []

        for block_idx in range(num_blocks_needed):
            start_token = block_idx * self.block_size
            end_token = min(start_token + self.block_size, len(prompt_tokens))
            block_tokens = tuple(prompt_tokens[start_token:end_token])

            # Check if this block exists in cache
            block_hash = hash(block_tokens)
            cache_key = f"{block_hash}"

            if cache_key in self.token_to_block_cache:
                # Block found in cache - reuse it
                block_id = self.token_to_block_cache[cache_key][0]
                allocated_blocks.append(block_id)
                shared_blocks += 1
                shared_tokens += len(block_tokens)
                self.metrics_collector.on_block_access(block_id, request_id, is_new_allocation=False)
            else:
                # Allocate new block
                block_id = self.next_block_id
                self.next_block_id += 1
                self.token_to_block_cache[cache_key] = [block_id]
                allocated_blocks.append(block_id)
                self.metrics_collector.on_block_access(block_id, request_id, is_new_allocation=True)

        return num_blocks_needed, shared_blocks, shared_tokens
