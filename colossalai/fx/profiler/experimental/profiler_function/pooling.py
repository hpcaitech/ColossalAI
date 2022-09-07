from typing import Tuple, Union
import torch
from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.nn.functional.avg_pool1d)
@meta_profiler_function.register(torch.nn.functional.avg_pool2d)
@meta_profiler_function.register(torch.nn.functional.avg_pool3d)
@meta_profiler_function.register(torch.nn.functional.max_pool1d)
@meta_profiler_function.register(torch.nn.functional.max_pool2d)
@meta_profiler_function.register(torch.nn.functional.max_pool3d)
@meta_profiler_function.register(torch.nn.functional.adaptive_avg_pool1d)
@meta_profiler_function.register(torch.nn.functional.adaptive_avg_pool2d)
@meta_profiler_function.register(torch.nn.functional.adaptive_avg_pool3d)
@meta_profiler_function.register(torch.nn.functional.adaptive_max_pool1d)
@meta_profiler_function.register(torch.nn.functional.adaptive_max_pool2d)
@meta_profiler_function.register(torch.nn.functional.adaptive_max_pool3d)
def torch_nn_func_pooling(input: torch.Tensor, *args, **kwargs) -> Tuple[int, int]:
    # all pooling could be considered as going over each input element only once (https://stackoverflow.com/a/67301217)
    flops = input.numel()
    macs = 0
    return flops, macs
