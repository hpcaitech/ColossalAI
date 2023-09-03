from typing import Any

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader


class CycledDataLoader:
    """
    Why do we need this class?
    In version 4da324cd60, "prompts = next(iter(self.prompt_dataloader))" is used to sample a batch of prompts/pretrain.
    However, this may be inefficient due to frequent re-initialization of the dataloader. (re-initialize workers...)
    NOTE: next(iter(dataloader)) is not equivalent to for batch in dataloader: break, it causes slightly different behavior.
    """

    def __init__(
        self,
        dataloader: DataLoader,
    ) -> None:
        self.dataloader = dataloader

        self.count = 0
        self.dataloader_iter = None

    def next(self):
        # defer initialization
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)

        self.count += 1
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.count = 0
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


from transformers.tokenization_utils_base import BatchEncoding

def to_device(x: Any, device: torch.device) -> Any:
    
    def _to(t: Any):
        if isinstance(t, BatchEncoding):
            for k in t.keys():
                t[k] = t[k].to(device)
            return t
        elif isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)