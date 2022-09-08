from typing import Tuple
import torch
from ..registry import meta_profiler_module


@meta_profiler_module.register(torch.nn.AvgPool1d)
@meta_profiler_module.register(torch.nn.AvgPool2d)
@meta_profiler_module.register(torch.nn.AvgPool3d)
@meta_profiler_module.register(torch.nn.MaxPool1d)
@meta_profiler_module.register(torch.nn.MaxPool2d)
@meta_profiler_module.register(torch.nn.MaxPool3d)
@meta_profiler_module.register(torch.nn.AdaptiveAvgPool1d)
@meta_profiler_module.register(torch.nn.AdaptiveMaxPool1d)
@meta_profiler_module.register(torch.nn.AdaptiveAvgPool2d)
@meta_profiler_module.register(torch.nn.AdaptiveMaxPool2d)
@meta_profiler_module.register(torch.nn.AdaptiveAvgPool3d)
@meta_profiler_module.register(torch.nn.AdaptiveMaxPool3d)
def torch_nn_pooling(self: torch.nn.Module, input: torch.Tensor) -> Tuple[int, int]:
    # all pooling could be considered as going over each input element only once (https://stackoverflow.com/a/67301217)
    flops = input.numel()
    macs = 0
    return flops, macs
