from typing import List, Optional, Tuple
import torch
from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.nn.functional.instance_norm)
def torch_nn_func_instancenorm(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.group_norm)
def torch_nn_func_groupnorm(input: torch.Tensor,
                            num_groups: int,
                            weight: Optional[torch.Tensor] = None,
                            bias: Optional[torch.Tensor] = None,
                            eps: float = 1e-5) -> Tuple[int, int]:
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.layer_norm)
def torch_nn_func_layernorm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None
    flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.nn.functional.batch_norm)
def torch_nn_func_batchnorm(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None
    if training:
        flops = input.numel() * (2 if has_affine else 1)
    else:
        flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs
