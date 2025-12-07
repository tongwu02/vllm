import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from collections import deque

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy."""
    LRU = enum.auto()
    LFU = enum.auto()
    FIFO = enum.auto()


class Evictor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        pass

    @abstractmethod
    def remove(self, block_id: int):
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order."""
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            last_accessed, _, block_id, _ = heapq.heappop(self.priority_queue)
            
            # Check if block is valid and timestamp matches
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                
                # [DEFENSIVE FIX] Get content_hash from free_table, not heap
                meta = self.free_table.pop(block_id)
                return block_id, meta.content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float):
        # Update metadata
        if block_id in self.free_table:
            self.free_table[block_id].last_accessed = last_accessed
            block = self.free_table[block_id]
            # Push new state to heap
            heapq.heappush(
                self.priority_queue,
                (last_accessed, -block.num_hashed_tokens, block_id, block.content_hash)
            )
            self._cleanup_if_necessary()

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue = []
        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)
        self.priority_queue = new_priority_queue

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class LFUEvictor(Evictor):
    """Evicts in a Least Frequently Used order."""
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.freq: Dict[int, int] = {}
        self.heap = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def add(self, block_id, content_hash, num_hashed_tokens, last_accessed):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        self.freq[block_id] = 1
        heapq.heappush(self.heap, (1, last_accessed, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id, last_accessed):
        if block_id in self.free_table:
            # Update metadata
            self.free_table[block_id].last_accessed = last_accessed
            
            # Increment frequency
            new_freq = self.freq.get(block_id, 0) + 1
            self.freq[block_id] = new_freq
            
            # Push new state to heap (Lazy Update)
            # Use current content_hash from table
            current_hash = self.free_table[block_id].content_hash
            heapq.heappush(self.heap, 
                           (new_freq, last_accessed, block_id, current_hash))
            self._cleanup_if_necessary()

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")

        while self.heap:
            freq, last_accessed, block_id, _ = heapq.heappop(self.heap)
            
            # Validity check:
            # 1. Block must still be in free_table
            # 2. Frequency in heap must match current frequency (filter stale updates)
            if (block_id in self.free_table and 
                self.freq.get(block_id) == freq):
                
                # [DEFENSIVE FIX] Use meta from free_table as source of truth
                meta = self.free_table.pop(block_id)
                self.freq.pop(block_id, None)
                
                # Verify consistency (Optional, helps debug)
                # assert meta.content_hash == hash_from_heap
                
                return block_id, meta.content_hash

        # If heap is exhausted but table is not (should happen rarely due to lazy delete), cleanup
        if self.free_table:
            self._cleanup()
            return self.evict()

        raise ValueError("No usable cache memory left")

    def remove(self, block_id):
        if block_id not in self.free_table:
            # Defensive return instead of raising Error, 
            # in case vLLM calls remove idempotently
            return 
        self.free_table.pop(block_id)
        self.freq.pop(block_id, None)

    def _cleanup_if_necessary(self):
        if len(self.heap) > LFUEvictor.CLEANUP_THRESHOLD * len(self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_heap = []
        for block_id, block in self.free_table.items():
            f = self.freq.get(block_id, 1)
            new_heap.append((f, block.last_accessed, block_id, block.content_hash))
        heapq.heapify(new_heap)
        self.heap = new_heap

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class FIFOEvictor(Evictor):
    """Evicts blocks in first-in-first-out order."""
    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.queue = deque()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")

        while self.queue:
            block_id = self.queue.popleft()
            
            if block_id in self.free_table:
                # [DEFENSIVE FIX] Use meta from free_table
                meta = self.free_table.pop(block_id)
                return block_id, meta.content_hash
            
            # If not in free_table, it was removed lazily, skip it.

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        if block_id in self.free_table:
            # Update meta, keep position in queue
            meta = self.free_table[block_id]
            meta.content_hash = content_hash
            meta.num_hashed_tokens = num_hashed_tokens
            meta.last_accessed = last_accessed
            return

        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed)
        self.queue.append(block_id)

    def update(self, block_id: int, last_accessed: float):
        if block_id in self.free_table:
            self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id in self.free_table:
            self.free_table.pop(block_id)
        # ID remains in queue, will be skipped by evict()

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")