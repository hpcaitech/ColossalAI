import torch
from ..registry import meta_profiler_module
from typing import Optional, Tuple


# TODO: calculate rnn FLOPs
@meta_profiler_module.register(torch.nn.GRU)
@meta_profiler_module.register(torch.nn.RNN)
def torch_nn_rnn(self: torch.nn.Module, input: torch.Tensor, hx: torch.Tensor) -> Tuple[int, int]:
    raise NotImplementedError
    flops = 0
    macs = 0
    return flops, macs
