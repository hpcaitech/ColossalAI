import operator
from functools import reduce
from typing import Any, Optional, Tuple

import torch

from ..registry import meta_profiler_function


@meta_profiler_function.register(torch.arange)
@meta_profiler_function.register(torch.finfo)
@meta_profiler_function.register(torch.permute)
@meta_profiler_function.register(torch.Tensor.permute)
@meta_profiler_function.register(torch.Tensor.repeat)
@meta_profiler_function.register(torch.index_select)
@meta_profiler_function.register(torch.Tensor.index_select)
@meta_profiler_function.register(torch.squeeze)
@meta_profiler_function.register(torch.Tensor.squeeze)
@meta_profiler_function.register(torch.unsqueeze)
@meta_profiler_function.register(torch.Tensor.unsqueeze)
@meta_profiler_function.register(torch.cat)
@meta_profiler_function.register(torch.concat)
@meta_profiler_function.register(torch.repeat_interleave)
@meta_profiler_function.register(torch.Tensor.repeat_interleave)
@meta_profiler_function.register(torch.flatten)
@meta_profiler_function.register(torch.Tensor.flatten)
@meta_profiler_function.register(torch.roll)
@meta_profiler_function.register(torch.full)
@meta_profiler_function.register(torch.Tensor.cpu)
@meta_profiler_function.register(torch.Tensor.cuda)
@meta_profiler_function.register(torch._assert)
def torch_zero_flops_op(*args, **kwargs) -> Tuple[int, int]:
    flops = 0
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.where)
def torch_where(condition: torch.Tensor, x: Any, y: Any) -> Tuple[int, int]:
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    flops = condition.numel()
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.max)
def torch_max(
    input: torch.Tensor, dim: int = None, keepdim: bool = False, *, out: Optional[torch.Tensor] = None
) -> Tuple[int, int]:
    macs = 0
    assert out is None, "assigning value to out is not supported yet"
    if dim is not None:
        shape = list(input.shape)
        shape.pop(int(dim))
        flops = reduce(operator.mul, shape), macs
        return flops, macs
    else:
        flops = input.numel()
        return flops, macs
