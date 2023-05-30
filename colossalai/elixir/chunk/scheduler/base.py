from abc import ABC, abstractmethod
from typing import Optional, Set

from colossalai.elixir.chunk.core import Chunk


class ChunkScheduler(ABC):
    """The base class of all chunk schedulers.
    A chunk scherduler stores all releasable chunks.
    It provides APIs to add, remove, display releasable chunks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.releasable_set: Optional[Set[Chunk]] = None
        self.current_step = -1

    @abstractmethod
    def reset(self) -> None:
        self.releasable_set = set()
        self.current_step = -1

    @abstractmethod
    def clear(self) -> None:
        # asure the set is empty now
        assert not bool(self.releasable_set)

    @abstractmethod
    def top(self) -> Optional[Chunk]:
        # return None if the releasable set is empty
        if not self.releasable_set:
            return False
        return True

    @abstractmethod
    def add(self, chunk: Chunk) -> bool:
        if chunk in self.releasable_set:
            return False
        self.releasable_set.add(chunk)
        return True

    @abstractmethod
    def remove(self, chunk: Chunk) -> bool:
        if chunk not in self.releasable_set:
            return False
        self.releasable_set.remove(chunk)
        return True

    def step(self, *args, **kwags):
        self.current_step += 1
