"""
Utils for Colossal-LLaMA
"""

import torch
import torch.distributed as dist


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor = tensor.data
    tensor.div_(dist.get_world_size())
    return tensor


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"
