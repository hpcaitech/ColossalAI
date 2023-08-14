import torch


def is_moe_param(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a moe param.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a moe param.
    """
    return hasattr(tensor, "moe_info")


def set_moe_param_info(tensor: torch.Tensor, moe_info: dict) -> None:
    """
    Set moe info for the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be set.
        moe_info (dict): The moe info to be set.

    """
    tensor.__setattr__('moe_info', moe_info)
