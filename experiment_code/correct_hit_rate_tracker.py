#!/usr/bin/env python3
"""
Correct Hit Rate Tracker

Correctly track the hit rate of each request according to the requirements on page 5 of 25TheFutureCloud.pdf.
Only count the hit rate of the first prefill call for each request, ignoring subsequent chunked prefills.
"""
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class CorrectHitRateTracker:
    """
    Correct Hit Rate Tracker

    Only count the hit rate at the first prefill of each request,
    avoiding pollution from repeated calls by chunked prefill.
    """

    def __init__(self):
        # Track which requests have already been counted
        self.counted_requests: Set[str] = set()

        # Statistics
        self.total_requests = 0
        self.total_blocks = 0
        self.hit_blocks = 0

        # Per-request detailed statistics
        self.request_stats: Dict[str, Dict] = {}

    def record_first_prefill(self, request_id: str, hit_blocks_count: int, total_blocks_count: int):
        """
        Record the hit rate of the first prefill for a request

        Args:
            request_id: Request ID
            hit_blocks_count: Number of hit blocks
            total_blocks_count: Total number of blocks
        """
        # Record only once
        if request_id in self.counted_requests:
            logger.debug(f"Request {request_id} already counted, skipping")
            return

        self.counted_requests.add(request_id)
        self.total_requests += 1
        self.total_blocks += total_blocks_count
        self.hit_blocks += hit_blocks_count

        # Record detailed information
        hit_rate = hit_blocks_count / total_blocks_count if total_blocks_count > 0 else 0.0
        self.request_stats[request_id] = {
            'hit_blocks': hit_blocks_count,
            'total_blocks': total_blocks_count,
            'hit_rate': hit_rate,
        }

        logger.debug(f"[HIT_RATE_TRACKER] request_id={request_id}, "
                    f"hit={hit_blocks_count}/{total_blocks_count}, "
                    f"hit_rate={hit_rate:.4f}")

    def get_overall_hit_rate(self) -> float:
        """
        Get overall hit rate (block-based)

        Returns:
            Hit rate (0.0 - 1.0)
        """
        if self.total_blocks == 0:
            return 0.0
        return self.hit_blocks / self.total_blocks

    def get_stats(self) -> Dict:
        """
        Get statistics

        Returns:
            Statistics dictionary
        """
        return {
            'total_requests': self.total_requests,
            'total_blocks': self.total_blocks,
            'hit_blocks': self.hit_blocks,
            'overall_hit_rate': self.get_overall_hit_rate(),
            'requests_with_hits': sum(1 for s in self.request_stats.values() if s['hit_blocks'] > 0),
            'requests_with_zero_hits': sum(1 for s in self.request_stats.values() if s['hit_blocks'] == 0),
            'request_stats': self.request_stats,  # Add per-request statistics
        }

    def reset(self):
        """Reset statistics"""
        self.counted_requests.clear()
        self.total_requests = 0
        self.total_blocks = 0
        self.hit_blocks = 0
        self.request_stats.clear()


# Global tracker instance (will be accessed from scheduler)
global_hit_rate_tracker = CorrectHitRateTracker()