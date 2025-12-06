import enum
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    LFU = enum.auto()
    FIFO = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                self.free_table.pop(block_id)
                return block_id, content_hash

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
        self.free_table[block_id].last_accessed = last_accessed

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class LFUEvictor(Evictor):
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
        self.freq[block_id] = self.freq.get(block_id, 0)
        heapq.heappush(self.heap,
                       (self.freq[block_id], last_accessed, block_id,
                        content_hash))

    def update(self, block_id, last_accessed):
        # 访问一次就 freq++，懒更新，真正 evict 时再过滤旧条目
        if block_id in self.free_table:
            self.freq[block_id] = self.freq.get(block_id, 0) + 1
            # 不直接改 heap，交给 evict 时做过滤即可

    def evict(self):
        while self.heap:
            freq, last_accessed, block_id, content_hash = heapq.heappop(
                self.heap)
            if (block_id in self.free_table
                    and self.freq.get(block_id, 0) == freq):
                self.free_table.pop(block_id)
                self.freq.pop(block_id, None)
                return block_id, content_hash
        raise ValueError("No usable cache memory left")

    def remove(self, block_id):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in evictor")
        self.free_table.pop(block_id)
        self.freq.pop(block_id, None)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

class FIFOEvictor(Evictor):
    """Evicts blocks in first-in-first-out order.

    语义：谁先变成“可被回收的 free block”，谁就先被 evict。
    不看 last_accessed，也不看 num_hashed_tokens。
    """

    def __init__(self):
        # block_id -> BlockMetaData
        self.free_table: Dict[int, BlockMetaData] = {}
        # 记录 block_id 进入 evictor 的顺序
        self.queue = deque()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        """按加入顺序，从队头开始找第一个还在 free_table 里的 block。"""
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.queue:
            block_id = self.queue.popleft()
            meta = self.free_table.get(block_id, None)
            if meta is not None:
                # 真正从候选里移除这个 block
                self.free_table.pop(block_id)
                return block_id, meta.content_hash

        # 理论上不太会走到这里，除非状态不一致
        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """把 block 标记成可被 evict 的候选（按 FIFO 顺序排队）。"""
        if block_id in self.free_table:
            # 已经在 evictor 里了，更新 meta，不重复入队
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
        """FIFO 不依赖 last_accessed，但为了接口一致，更新一下 metadata。"""
        if block_id in self.free_table:
            self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        """从 evictor 中删除一个 block。

        注意：我们不从 queue 里 O(n) 地删，lazy 删除，evict 时会自动跳过。
        """
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

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
