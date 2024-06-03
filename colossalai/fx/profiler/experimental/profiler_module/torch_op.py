from typing import Tuple

import torch

from ..registry import meta_profiler_module


@meta_profiler_module.register(torch.nn.Flatten)
def torch_nn_flatten(self: torch.nn.Flatten, input: torch.Tensor) -> Tuple[int, int]:
    flops = 0
    macs = 0
    return flops, macs
