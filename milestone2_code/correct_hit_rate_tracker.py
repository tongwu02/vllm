#!/usr/bin/env python3
"""
Correct Hit Rate Tracker

根据25TheFutureCloud.pdf第5页的要求,正确追踪每个request的hit rate。
只统计每个request第一次prefill调用的hit rate,忽略后续的chunked prefill。
"""
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class CorrectHitRateTracker:
    """
    正确的Hit Rate追踪器

    只统计每个request第一次prefill时的hit rate,
    避免被chunked prefill的重复调用污染。
    """

    def __init__(self):
        # 追踪哪些requests已经统计过
        self.counted_requests: Set[str] = set()

        # 统计数据
        self.total_requests = 0
        self.total_blocks = 0
        self.hit_blocks = 0

        # Per-request详细数据
        self.request_stats: Dict[str, Dict] = {}

    def record_first_prefill(self, request_id: str, hit_blocks_count: int, total_blocks_count: int):
        """
        记录某个request第一次prefill的hit rate

        Args:
            request_id: Request ID
            hit_blocks_count: 命中的blocks数量
            total_blocks_count: 总blocks数量
        """
        # 只记录一次
        if request_id in self.counted_requests:
            logger.debug(f"Request {request_id} already counted, skipping")
            return

        self.counted_requests.add(request_id)
        self.total_requests += 1
        self.total_blocks += total_blocks_count
        self.hit_blocks += hit_blocks_count

        # 记录详细信息
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
        获取整体hit rate (基于blocks)

        Returns:
            Hit rate (0.0 - 1.0)
        """
        if self.total_blocks == 0:
            return 0.0
        return self.hit_blocks / self.total_blocks

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计字典
        """
        return {
            'total_requests': self.total_requests,
            'total_blocks': self.total_blocks,
            'hit_blocks': self.hit_blocks,
            'overall_hit_rate': self.get_overall_hit_rate(),
            'requests_with_hits': sum(1 for s in self.request_stats.values() if s['hit_blocks'] > 0),
            'requests_with_zero_hits': sum(1 for s in self.request_stats.values() if s['hit_blocks'] == 0),
            'request_stats': self.request_stats,  # 添加per-request统计
        }

    def reset(self):
        """重置统计"""
        self.counted_requests.clear()
        self.total_requests = 0
        self.total_blocks = 0
        self.hit_blocks = 0
        self.request_stats.clear()


# Global tracker instance (will be accessed from scheduler)
global_hit_rate_tracker = CorrectHitRateTracker()
