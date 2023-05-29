from collections import defaultdict
from typing import Iterable, List, Optional

import torch
from sortedcontainers import SortedSet

from .base import Chunk, ChunkScheduler


class PrefetchScheduler(ChunkScheduler):
    """The prefetch chunk scheduler.
    Its top functions gives the furthest used chunk.
    """

    def __init__(self, chunk_called_per_step: List[Iterable[Chunk]]) -> None:
        super().__init__()
        self.chunk_mapping = None
        self.evict_set = None
        self.search_step = -1

        self.chunks_per_step = chunk_called_per_step
        self.total_steps = len(chunk_called_per_step)
        self.next_step_dict = defaultdict(list)
        # initialize the next_step dictionary
        for i, c_list in enumerate(chunk_called_per_step):
            for c in c_list:
                self.next_step_dict[c].append(i)

    def _get_next_step(self, chunk: Chunk):
        step_list = self.next_step_dict[chunk]
        for i in step_list:
            if i > self.current_step:
                return i
        return self.total_steps

    def reset(self) -> None:
        super().reset()
        self.chunk_mapping = dict()
        self.evict_set = SortedSet()
        self.search_step = -1

    def clear(self) -> None:
        super().clear()
        if torch.is_grad_enabled():
            assert self.current_step == self.total_steps - 1
        self.chunk_mapping = None
        self.evict_set = None
        self.search_step = -1

    def top(self) -> Optional[Chunk]:
        if not super().top():
            return None
        next_step, chunk = self.evict_set[-1]
        return chunk

    def add(self, chunk: Chunk) -> bool:
        if not super().add(chunk):
            return False
        value = (self._get_next_step(chunk), chunk)
        self.chunk_mapping[chunk] = value
        self.evict_set.add(value)
        return True

    def remove(self, chunk: Chunk) -> bool:
        if not super().remove(chunk):
            return False
        value = self.chunk_mapping[chunk]
        self.evict_set.remove(value)
        self.chunk_mapping.pop(chunk)
        return True

    def step(self, *args, **kwags):
        super().step(*args, **kwags)
        if self.current_step >= self.total_steps:
            raise RuntimeError('exceed simulated steps, please modify your profiling `step_fn`')

    def get_next_chunk(self, chunks: List[Chunk]):
        self.search_step = max(self.search_step, self.current_step + 1)
        while self.search_step < self.total_steps:
            c_list = self.chunks_per_step[self.search_step]
            for c in c_list:
                if c not in chunks:
                    return c
            self.search_step += 1
        return None
