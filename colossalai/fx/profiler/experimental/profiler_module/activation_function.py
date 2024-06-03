from typing import Tuple

import torch

from ..registry import meta_profiler_module

# TODO: different activation has different FLOPs count, currently unused.
_multiplier = {
    torch.nn.ReLU: 1,
    torch.nn.PReLU: 4,
    torch.nn.Sigmoid: 4,
    torch.nn.Tanh: 5,
    torch.nn.LeakyReLU: 3,
    torch.nn.ELU: 4,
    torch.nn.ReLU6: 2,
    torch.nn.GELU: 9,
    torch.nn.Hardswish: 5,
    torch.nn.Hardsigmoid: 4,
}


@meta_profiler_module.register(torch.nn.ELU)
@meta_profiler_module.register(torch.nn.LeakyReLU)
@meta_profiler_module.register(torch.nn.ReLU)
@meta_profiler_module.register(torch.nn.GELU)
@meta_profiler_module.register(torch.nn.Sigmoid)
@meta_profiler_module.register(torch.nn.Tanh)
@meta_profiler_module.register(torch.nn.ReLU6)
@meta_profiler_module.register(torch.nn.PReLU)
@meta_profiler_module.register(torch.nn.Hardswish)
@meta_profiler_module.register(torch.nn.Hardsigmoid)
def torch_nn_non_linear_act(self: torch.nn.Module, input: torch.Tensor) -> Tuple[int, int]:
    flops = input.numel()
    macs = 0
    return flops, macs
