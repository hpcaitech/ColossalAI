import operator
from functools import reduce
from typing import Any, Optional, Tuple, Union
import torch
from ..registry import meta_profiler_function


def _elementwise_flops_compute(input, other):
    # copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py#L763
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return reduce(operator.mul, other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return reduce(operator.mul, input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = reduce(operator.mul, final_shape)
        return flops, 0


@meta_profiler_function.register(torch.add)
@meta_profiler_function.register(torch.eq)
@meta_profiler_function.register(torch.sub)
@meta_profiler_function.register(torch.mul)
@meta_profiler_function.register(torch.floor_divide)
@meta_profiler_function.register('add')    # for built-in op +
@meta_profiler_function.register('iadd')    # for built-in op +=
@meta_profiler_function.register('eq')    # for built-in op =
@meta_profiler_function.register('sub')    # for built-in op -
@meta_profiler_function.register('isub')    # for built-in op -=
@meta_profiler_function.register('mul')    # for built-in op *
@meta_profiler_function.register('imul')    # for built-in op *=
@meta_profiler_function.register('floordiv')    # for built-in op //
@meta_profiler_function.register('ifloordiv')    # for built-in op //=
def torch_add_like_ops(input: Any, other: Any, *, out: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    return _elementwise_flops_compute(input, other)


@meta_profiler_function.register(torch.abs)
def torch_elementwise_op(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    flops = input.numel()
    macs = 0
    return flops, macs


@meta_profiler_function.register(torch.matmul)
@meta_profiler_function.register('matmul')    # for built-in op @
@meta_profiler_function.register(torch.Tensor.matmul)
def torch_matmul(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    macs = reduce(operator.mul, input.shape) * other.shape[-1]
    flops = 2 * macs
    return flops, macs


@meta_profiler_function.register(torch.bmm)
def torch_bmm(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    macs = reduce(operator.mul, input.shape) * other.shape[-1]
    flops = 2 * macs
    return flops, macs


@meta_profiler_function.register(torch.var_mean)
def torch_var_mean(input: torch.Tensor,
                   dim: Union[int, Tuple[int, ...]],
                   unbiased: Optional[bool] = True,
                   keepdim: Optional[bool] = False,
                   *,
                   out: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    assert out is None, 'saving to out is not supported yet'
    flops = input.numel() * 3
    macs = 0
    return flops, macs
