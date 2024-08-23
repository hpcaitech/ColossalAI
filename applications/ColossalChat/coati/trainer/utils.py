"""
Training utilities for Coati.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from colossalai.booster import Plugin


class CycledDataLoader:
    """
    A data loader that cycles through the data when it reaches the end.

    Args:
        dataloader (DataLoader): The original data loader.

    Attributes:
        dataloader (DataLoader): The original data loader.
        count (int): The number of times the data loader has been cycled.
        dataloader_iter (iterable): The iterator for the data loader.

    Methods:
        next(): Returns the next batch of data from the data loader, cycling through the data if necessary.
    """

    def __init__(
        self,
        dataloader: DataLoader,
    ) -> None:
        self.dataloader = dataloader

        self.count = 0
        self.dataloader_iter = None

    def next(self):
        """
        Returns the next batch of data from the data loader, cycling through the data if necessary.

        Returns:
            Any: The next batch of data from the data loader.
        """
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
    """
    Check if the current process is the rank 0 process in a distributed training setup.

    Returns:
        bool: True if the current process is the rank 0 process, False otherwise.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def to_device(x: Any, device: torch.device) -> Any:
    """
    Move the input tensor or nested structure of tensors to the specified device.

    Args:
        x (Any): The input tensor or nested structure of tensors.
        device (torch.device): The target device to move the tensors to.

    Returns:
        Any: The tensor or nested structure of tensors moved to the target device.
    """

    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)


def all_reduce_mean(tensor: torch.Tensor, plugin: Plugin = None) -> torch.Tensor:
    """
    Perform all-reduce operation on the given tensor and compute the mean across all processes.

    Args:
        tensor (torch.Tensor): The input tensor to be reduced.

    Returns:
        torch.Tensor: The reduced tensor with mean computed across all processes.
    """
    # All reduce mean across DP group
    if plugin is not None:
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=plugin.dp_group)
        tensor.div_(plugin.dp_size)
    else:
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an all-reduce operation to sum the values of the given tensor across all processes.

    Args:
        tensor (torch.Tensor): The input tensor to be reduced.

    Returns:
        torch.Tensor: The reduced tensor with the sum of values across all processes.
    """
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    return tensor
