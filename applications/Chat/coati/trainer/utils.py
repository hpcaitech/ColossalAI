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


def to_device(x: Any, device: torch.device) -> Any:
    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    return tensor


def prepare_data_for_inference(
    chosen_input_ids, reject_input_ids, chosen_attention_mask, reject_attention_mask, tokenizer
):
    # This function will truncate each vector in the batch after the first zero
    assert tokenizer.bos_token_id != tokenizer.pad_token_id, "This bos token should not be the same as the pad token"
    return None, None
