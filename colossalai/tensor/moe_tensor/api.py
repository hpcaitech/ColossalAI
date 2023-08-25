import torch
import torch.distributed as dist

from colossalai.tensor import ProcessGroup

from .moe_info import MoeParallelInfo


def is_moe_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a moe tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a moe tensor.
    """
    return hasattr(tensor, "moe_info")


def set_moe_tensor_info(tensor: torch.Tensor, moe_info: MoeParallelInfo) -> None:
    """
    Set moe info for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be set.
        moe_info (dict): The moe info to be set.

    """
    tensor.__setattr__('moe_info', moe_info)


def get_moe_info(ep_size: int, dp_size: int) -> MoeParallelInfo:
    """
    Get moe info for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        dict: The moe info of the given tensor.
    """
    return MoeParallelInfo(ep_size, dp_size)


def get_ep_group(tensor: torch.Tensor) -> ProcessGroup:
    """
    Get the expert parallel group of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        torch.distributed.ProcessGroup: The expert parallel group of the given tensor.
    """
    return tensor.moe_info.ep_group


def get_ep_size(tensor: torch.Tensor) -> int:
    """
    Get the expert parallel size of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        int: The expert parallel size of the given tensor.
    """
    return tensor.moe_info.ep_size


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
