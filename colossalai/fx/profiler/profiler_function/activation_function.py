from typing import Tuple
import torch
from ..registry import meta_profiler_function

# TODO: different activation has different FLOPs count, currently unused.
_multiplier = {
    torch.nn.functional.relu: 1,
    torch.nn.functional.prelu: 4,
    torch.nn.functional.sigmoid: 4,
    torch.nn.functional.tanh: 5,
    torch.nn.functional.leaky_relu: 3,
    torch.nn.functional.elu: 4,
    torch.nn.functional.relu6: 2,
    torch.nn.functional.gelu: 9,
    torch.nn.functional.hardswish: 5,
    torch.nn.functional.hardsigmoid: 4,
}


@meta_profiler_function.register(torch.nn.functional.leaky_relu)
@meta_profiler_function.register(torch.nn.functional.elu)
@meta_profiler_function.register(torch.nn.functional.gelu)
@meta_profiler_function.register(torch.nn.functional.relu6)
@meta_profiler_function.register(torch.nn.functional.prelu)
@meta_profiler_function.register(torch.nn.functional.relu)
@meta_profiler_function.register(torch.nn.functional.sigmoid)
@meta_profiler_function.register(torch.nn.functional.tanh)
@meta_profiler_function.register(torch.nn.functional.hardswish)
@meta_profiler_function.register(torch.nn.functional.hardsigmoid)
def torch_nn_func_non_linear_act(input: torch.Tensor, inplace: bool = False) -> Tuple[int, int]:
    flops = input.numel()
    macs = 0
    return flops, macs
