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
    PROTECTED_LRU = enum.auto()  # protected LRU

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

class ProtectedLRUEvictor(Evictor):
    """
    Hard-Partitioned SLRU (Segmented LRU).
    Enforces a strict limit on how much space 'One-hit Wonders' (Noise) can occupy.
    """
    
    # 设定试用区最大占比 (例如 20%)
    # 这意味着 80% 的显存是专门留给 Shared Prompt 等回头客的
    PROBATION_RATIO = 0.20 

    def __init__(self):
        # 1. 试用区 (Noise 集中营)
        self.probation_table: Dict[int, BlockMetaData] = {}
        self.probation_queue = [] 

        # 2. 保护区 (VIP 俱乐部)
        self.protected_table: Dict[int, BlockMetaData] = {}
        self.protected_queue = [] 

    def __contains__(self, block_id: int) -> bool:
        return (block_id in self.probation_table) or (block_id in self.protected_table)

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """新来的 Block 只能进试用区"""
        meta = BlockMetaData(content_hash, num_hashed_tokens, last_accessed)
        self.probation_table[block_id] = meta
        heapq.heappush(self.probation_queue, (last_accessed, block_id, content_hash))

    def update(self, block_id: int, last_accessed: float):
        """复用发生了！立刻晋升！"""
        
        # A. 在试用区 -> 晋升到保护区
        if block_id in self.probation_table:
            meta = self.probation_table.pop(block_id)
            meta.last_accessed = last_accessed
            
            self.protected_table[block_id] = meta
            heapq.heappush(self.protected_queue, (last_accessed, block_id, meta.content_hash))
            
        # B. 已经在保护区 -> 更新时间
        elif block_id in self.protected_table:
            meta = self.protected_table[block_id]
            meta.last_accessed = last_accessed
            heapq.heappush(self.protected_queue, (last_accessed, block_id, meta.content_hash))

    def evict(self) -> Tuple[int, int]:
        """
        屠杀逻辑 (关键修改)：
        我们根据当前各区的占用情况，决定杀谁。
        """
        total_blocks = len(self.probation_table) + len(self.protected_table)
        if total_blocks == 0:
            raise ValueError("Cache empty")

        probation_size = len(self.probation_table)
        
        # === 核心逻辑 ===
        # 如果试用区占用的空间超过了限制 (20%)，或者保护区是空的，
        # 就必须杀试用区的人。这保证了 Noise 永远无法挤占 >20% 的空间。
        should_evict_probation = (probation_size > total_blocks * self.PROBATION_RATIO) or (not self.protected_table)

        if should_evict_probation and self.probation_table:
            block_id, content_hash = self._pop_valid_lru(self.probation_queue, self.probation_table)
            if block_id is not None:
                return block_id, content_hash

        # 否则 (试用区很空，或者是保护区满了)，才动保护区
        if self.protected_table:
            block_id, content_hash = self._pop_valid_lru(self.protected_queue, self.protected_table)
            if block_id is not None:
                return block_id, content_hash
        
        # 兜底：如果上面都没返回 (极罕见情况)，再试一次试用区
        if self.probation_table:
             block_id, content_hash = self._pop_valid_lru(self.probation_queue, self.probation_table)
             if block_id is not None:
                 return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def remove(self, block_id: int):
        if block_id in self.probation_table:
            self.probation_table.pop(block_id)
        elif block_id in self.protected_table:
            self.protected_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.probation_table) + len(self.protected_table)

    def _pop_valid_lru(self, queue, table) -> Tuple[Optional[int], int]:
        while queue:
            last_accessed, block_id, _ = heapq.heappop(queue)
            if block_id in table:
                meta = table[block_id]
                if meta.last_accessed == last_accessed:
                    del table[block_id]
                    return block_id, meta.content_hash
        return None, 0

# 3. 修改 Factory
def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    elif eviction_policy == EvictionPolicy.LFU:
        return LFUEvictor()
    elif eviction_policy == EvictionPolicy.FIFO:
        return FIFOEvictor()
    elif eviction_policy == EvictionPolicy.PROTECTED_LRU: # <--- 注册
        return ProtectedLRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")