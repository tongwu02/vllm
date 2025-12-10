#!/usr/bin/env python3
"""
Cache Block Tracker for Milestone 2 Task 2

Tracks the following per project.pdf requirements:
- The number of hits per cache block
- The time gap between reuses of each cache block
- Additional interesting metrics
"""
import time
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CacheBlockTracker:
    """Tracks usage of cache blocks"""

    def __init__(self):
        # block_id -> list of (timestamp, request_id)
        self.block_access_history: Dict[int, List[Tuple[float, str]]] = defaultdict(list)

        # block_id -> number of hits
        self.block_hit_count: Dict[int, int] = defaultdict(int)

        # block_id -> list of reuse time gaps (in seconds)
        self.block_reuse_gaps: Dict[int, List[float]] = defaultdict(list)

        # request_id -> list of block_ids that were cache hits
        self.request_hit_blocks: Dict[str, List[int]] = defaultdict(list)

        # Temporal data for sharing rate over time
        # List of (timestamp, is_hit) for each cache access
        self.temporal_access_log: List[Tuple[float, bool]] = []

        # Per-request statistics for temporal analysis
        # request_id -> (timestamp, num_hits, num_blocks)
        self.request_timeline: List[Tuple[float, str, int, int]] = []

        self.start_time = time.time()

    def record_cache_hit(self, request_id: str, block_ids: List[int]):
        """
        Record cache hits (block reuse) for a request

        Args:
            request_id: Request ID
            block_ids: List of hit block IDs (every cache hit counts as a block reuse)
        """
        current_time = time.time()

        for block_id in block_ids:
            # Record access history
            self.block_access_history[block_id].append((current_time, request_id))

            # Increment hit count (every hit counts as a usage)
            self.block_hit_count[block_id] += 1

            # Calculate reuse time gap (recorded starting from the first access)
            access_history = self.block_access_history[block_id]
            if len(access_history) >= 2:
                # If it is the 2nd access or later, record the time gap from the previous access
                prev_time, _ = access_history[-2]
                gap = current_time - prev_time
                self.block_reuse_gaps[block_id].append(gap)

            # Record hit blocks for the request
            self.request_hit_blocks[request_id].append(block_id)

            # Record temporal access log (once per block access)
            self.temporal_access_log.append((current_time, True))

        # Record request timeline
        if block_ids:
            self.request_timeline.append((current_time, request_id, len(block_ids), len(set(block_ids))))

    def get_stats(self) -> Dict:
        """Get statistics"""

        # Calculate hits per block statistics
        hit_counts = list(self.block_hit_count.values())
        if hit_counts:
            avg_hits_per_block = sum(hit_counts) / len(hit_counts)
            max_hits = max(hit_counts)
            min_hits = min(hit_counts)
        else:
            avg_hits_per_block = 0
            max_hits = 0
            min_hits = 0

        # Calculate reuse time gaps statistics
        all_gaps = []
        for gaps in self.block_reuse_gaps.values():
            all_gaps.extend(gaps)

        if all_gaps:
            avg_reuse_gap = sum(all_gaps) / len(all_gaps)
            max_reuse_gap = max(all_gaps)
            min_reuse_gap = min(all_gaps)
        else:
            avg_reuse_gap = 0
            max_reuse_gap = 0
            min_reuse_gap = 0

        # Build distribution data (for visualization)
        hit_count_distribution = defaultdict(int)
        for count in hit_counts:
            hit_count_distribution[count] += 1

        # Reuse gap distribution (by bucket)
        gap_buckets = [0.001, 0.01, 0.1, 1.0, 10.0, float('inf')]
        gap_distribution = defaultdict(int)
        for gap in all_gaps:
            for i, threshold in enumerate(gap_buckets):
                if gap < threshold:
                    bucket_name = f"<{threshold}s" if i == 0 else f"{gap_buckets[i-1]}-{threshold}s"
                    gap_distribution[bucket_name] += 1
                    break

        return {
            # Cache block usage statistics
            'total_cached_blocks': len(self.block_hit_count),
            'total_block_accesses': sum(hit_counts),  # Total cache hit count (including first and repeated)
            'avg_hits_per_block': avg_hits_per_block,
            'max_hits_per_block': max_hits,
            'min_hits_per_block': min_hits,
            'hit_count_distribution': dict(hit_count_distribution),

            # Block reuse time gap statistics (gaps exist only for 2nd access onwards)
            'total_reuses': len(all_gaps),  # Number of reuses with time gaps (2nd access onwards)
            'avg_reuse_gap_seconds': avg_reuse_gap,
            'max_reuse_gap_seconds': max_reuse_gap,
            'min_reuse_gap_seconds': min_reuse_gap,
            'reuse_gap_distribution': dict(gap_distribution),

            # Raw data (for detailed analysis and visualization)
            'block_hit_counts': dict(self.block_hit_count),
            'reuse_gaps_by_block': {k: list(v) for k, v in self.block_reuse_gaps.items()},
            'all_reuse_gaps': all_gaps,

            # Added: Temporal data (for sharing rate over time)
            'temporal_access_log': self.temporal_access_log,
            'request_timeline': self.request_timeline,
            'experiment_duration': time.time() - self.start_time,
        }

    def reset(self):
        """Reset all statistics"""
        self.block_access_history.clear()
        self.block_hit_count.clear()
        self.block_reuse_gaps.clear()
        self.request_hit_blocks.clear()
        self.temporal_access_log.clear()
        self.request_timeline.clear()
        self.start_time = time.time()


# Global tracker instance
global_cache_block_tracker = CacheBlockTracker()