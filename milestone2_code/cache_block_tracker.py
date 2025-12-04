#!/usr/bin/env python3
"""
Cache Block Tracker for Milestone 2 Task 2

根据project.pdf要求，追踪：
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
    """追踪cache block的使用情况"""

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
        记录一个request的cache hits（block reuse）

        Args:
            request_id: 请求ID
            block_ids: 命中的block IDs列表（每次cache hit都算作一次block reuse）
        """
        current_time = time.time()

        for block_id in block_ids:
            # 记录访问历史
            self.block_access_history[block_id].append((current_time, request_id))

            # 增加命中计数（每次hit都算作一次使用）
            self.block_hit_count[block_id] += 1

            # 计算reuse time gap（从第一次访问开始就记录）
            access_history = self.block_access_history[block_id]
            if len(access_history) >= 2:
                # 如果是第2次及以上访问，记录与前一次的时间间隔
                prev_time, _ = access_history[-2]
                gap = current_time - prev_time
                self.block_reuse_gaps[block_id].append(gap)

            # 记录request的hit blocks
            self.request_hit_blocks[request_id].append(block_id)

            # 记录temporal access log（每个block access记录一次）
            self.temporal_access_log.append((current_time, True))

        # 记录request timeline
        if block_ids:
            self.request_timeline.append((current_time, request_id, len(block_ids), len(set(block_ids))))

    def get_stats(self) -> Dict:
        """获取统计信息"""

        # 计算hits per block统计
        hit_counts = list(self.block_hit_count.values())
        if hit_counts:
            avg_hits_per_block = sum(hit_counts) / len(hit_counts)
            max_hits = max(hit_counts)
            min_hits = min(hit_counts)
        else:
            avg_hits_per_block = 0
            max_hits = 0
            min_hits = 0

        # 计算reuse time gaps统计
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

        # 构建分布数据（用于可视化）
        hit_count_distribution = defaultdict(int)
        for count in hit_counts:
            hit_count_distribution[count] += 1

        # Reuse gap分布（按bucket）
        gap_buckets = [0.001, 0.01, 0.1, 1.0, 10.0, float('inf')]
        gap_distribution = defaultdict(int)
        for gap in all_gaps:
            for i, threshold in enumerate(gap_buckets):
                if gap < threshold:
                    bucket_name = f"<{threshold}s" if i == 0 else f"{gap_buckets[i-1]}-{threshold}s"
                    gap_distribution[bucket_name] += 1
                    break

        return {
            # Cache block使用统计
            'total_cached_blocks': len(self.block_hit_count),
            'total_block_accesses': sum(hit_counts),  # 总的cache hit次数（包括首次和重复）
            'avg_hits_per_block': avg_hits_per_block,
            'max_hits_per_block': max_hits,
            'min_hits_per_block': min_hits,
            'hit_count_distribution': dict(hit_count_distribution),

            # Block重用时间间隔统计（第2次及以上访问才有间隔）
            'total_reuses': len(all_gaps),  # 有时间间隔的reuse次数（第2次及以上）
            'avg_reuse_gap_seconds': avg_reuse_gap,
            'max_reuse_gap_seconds': max_reuse_gap,
            'min_reuse_gap_seconds': min_reuse_gap,
            'reuse_gap_distribution': dict(gap_distribution),

            # 原始数据（用于详细分析和可视化）
            'block_hit_counts': dict(self.block_hit_count),
            'reuse_gaps_by_block': {k: list(v) for k, v in self.block_reuse_gaps.items()},
            'all_reuse_gaps': all_gaps,

            # 新增：时序数据（用于sharing rate over time）
            'temporal_access_log': self.temporal_access_log,
            'request_timeline': self.request_timeline,
            'experiment_duration': time.time() - self.start_time,
        }

    def reset(self):
        """重置所有统计"""
        self.block_access_history.clear()
        self.block_hit_count.clear()
        self.block_reuse_gaps.clear()
        self.request_hit_blocks.clear()
        self.temporal_access_log.clear()
        self.request_timeline.clear()
        self.start_time = time.time()


# Global tracker instance
global_cache_block_tracker = CacheBlockTracker()
