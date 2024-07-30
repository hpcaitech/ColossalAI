from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .moe_info import MoeParallelInfo


def is_moe_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a moe tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a moe tensor.
    """
    return hasattr(tensor, "ep_group")


def set_moe_tensor_ep_group(tensor: torch.Tensor, ep_group: ProcessGroup) -> None:
    """
    Set moe info for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be set.
        moe_info (dict): The moe info to be set.

    """
    tensor.__setattr__("ep_group", ep_group)


def get_moe_info(ep_size: int, dp_size: int, pp_size: int, ep_inside: bool) -> MoeParallelInfo:
    """
    Get moe info for the given tensor.

    Args:
        ep_size (int): The expert parallel size.
        dp_size (int): The data parallel size.
        pp_size (int): The pipeline parallel size.
        ep_inside (bool, optional): Use ep inside dp if True, dp inside ep if False.

    Returns:
        dict: The moe info of the given tensor.
    """
    return MoeParallelInfo(ep_inside, ep_size, dp_size, pp_size)


def get_ep_group(tensor: torch.Tensor) -> ProcessGroup:
    """
    Get the expert parallel group of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        torch.distributed.ProcessGroup: The expert parallel group of the given tensor.
    """
    return tensor.ep_group


def get_ep_size(tensor: torch.Tensor) -> int:
    """
    Get the expert parallel size of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The expert parallel size of the given tensor.
    """
    assert getattr(tensor, "ep_group") is not None, "The tensor does not have expert parallel group."
    return dist.get_world_size(tensor.ep_group)


def get_dp_size(tensor: torch.Tensor) -> int:
    """
    Get the data parallel size of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The data parallel size of the given tensor.
    """
    return tensor.moe_info.dp_size


def get_dp_group(tensor: torch.Tensor) -> ProcessGroup:
    """
    Get the data parallel group of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        torch.distributed.ProcessGroup: The data parallel group of the given tensor.
    """
    return tensor.moe_info.dp_group


def get_ep_rank(tensor: torch.Tensor) -> int:
    """
    Get the expert parallel rank of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The expert parallel rank of the given tensor.
    """
    return dist.get_rank(get_ep_group(tensor))


def get_dp_rank(tensor: torch.Tensor) -> int:
    """
    Get the data parallel rank of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The data parallel rank of the given tensor.
    """
    return dist.get_rank(get_dp_group(tensor))


def get_ep_group_ranks(tensor: torch.Tensor) -> List[int]:
    """
    Get the expert parallel group ranks of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The expert parallel group ranks of the given tensor.
    """
    return tensor.moe_info.ep_group_ranks


def get_dp_group_ranks(tensor: torch.Tensor) -> List[int]:
    """
    Get the data parallel group ranks of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The data parallel group ranks of the given tensor.
    """
    return tensor.moe_info.dp_group_ranks
