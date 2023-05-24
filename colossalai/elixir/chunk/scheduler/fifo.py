from typing import Optional

from .base import Chunk, ChunkScheduler


class FIFOScheduler(ChunkScheduler):

    def __init__(self) -> None:
        super().__init__()
        self.fifo_dict: Optional[dict] = None

    def reset(self) -> None:
        super().reset()
        self.fifo_dict = dict()

    def clear(self) -> None:
        super().clear()
        self.fifo_dict = None

    def top(self) -> Optional[Chunk]:
        if not super().top():
            return None
        dict_iter = iter(self.fifo_dict)
        ret = next(dict_iter)
        return ret

    def add(self, chunk: Chunk) -> bool:
        if not super().add(chunk):
            return False
        self.fifo_dict[chunk] = True
        return True

    def remove(self, chunk: Chunk) -> bool:
        if not super().remove(chunk):
            return False
        self.fifo_dict.pop(chunk)
        return True
