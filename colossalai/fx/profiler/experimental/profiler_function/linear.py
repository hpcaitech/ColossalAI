from typing import Tuple

import torch

from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.nn.functional.linear)
def torch_nn_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> Tuple[int, int]:
    out_features = weight.shape[0]
    macs = torch.numel(input) * out_features
    flops = 2 * macs
    if bias is not None:
        flops += bias.numel()
    return flops, macs
