from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch
from torch.fx.node import Argument, Target

from ..._compatibility import compatibility
from ..memory_utils import activation_size
from .constants import INPLACE_METHOD, INPLACE_OPS, NON_INPLACE_METHOD
from .registry import meta_profiler_function, meta_profiler_module

__all__ = ['profile_function', 'profile_module', 'profile_method']


# this is for compatibility use
@compatibility(is_backward_compatible=True)
@dataclass
class GraphInfo:
    """
    GraphInfo is a dataclass for MetaInfo, which measures
    the execution memory cost and FLOPs with `MetaTensor`.
    The dataflow analysis is conducted on a single node of the FX graph.
    ============================================================================
                            -------------------------------
                            |            Node             |
    [fwd_in] are       ---> | [fwd_in]          [bwd_out] |    <----- [bwd_out] is marks the memory for `grad_out`
    placeholders saved for  |     | \__________     |     |
    backward.               |     |            \    |     |
                            | [fwd_tmp] ------> [bwd_tmp] |    <-----
                            |     |  \_________     |     |    [bwd_tmp] marks the peak memory
                            |    / \           \    |     |    in backward pass.
    [x] is not counted ---> | [x]  [fwd_tmp] -> [bwd_tmp] |    <-----
    in [fwd_tmp] because    |  |       |  \_____    |     |
    it is not saved for     |  |       |        \   |     |
    backward.               -------------------------------
    ============================================================================
    Attributes:
        fwd_flop (int): The forward FLOPs of a certain node
        bwd_flop (int): The backward FLOPs of a certain node.
        fwd_mem_in (int): See the above illustration.
        fwd_mem_tmp (int): See the above illustration.
        bwd_mem_tmp (int): See the above illustration.
        bwd_mem_out (int): See the above illustration.
    """
    fwd_flop: int = 0
    bwd_flop: int = 0
    fwd_mem_in: int = 0
    fwd_mem_tmp: int = 0
    bwd_mem_tmp: int = 0
    bwd_mem_out: int = 0


CALL_FUNCTION_MSG = \
"""
Colossal-AI hasn't supported profiling for {}, you might manually patch it with the following code.\n
from colossalai.fx.profiler.experimental import meta_profiler_function
@meta_profiler_function.register(YOUR_FUNCTION)
def profile_YOUR_FUNCTION(input: torch.Tensor, *args) -> Tuple[int, int]:
    flops = ...
    macs = ...
    return flops, macs
"""
CALL_METHOD_MSG = 'Please check if {} is an inplace method. If so, add target to INPLACE_METHOD={}. Otherwise, add target to NON_INPLACE_METHOD={}'
CALL_MODULE_MSG = \
"""
Colossal-AI hasn't supported profiling for {}, you might manually patch it with the following code.\n
from colossalai.fx.profiler.experimental import meta_profiler_module
@meta_profiler_module.register(YOUR_MODULE)
def profile_YOUR_MODULE(self: torch.nn.Module, input: torch.Tensor) -> Tuple[int, int]:
    flops = ...
    macs = ...
    return flops, macs
"""


@compatibility(is_backward_compatible=True)
def profile_function(target: 'Target') -> Callable:
    """
    Wrap a `call_function` node or `torch.nn.functional` in order to
    record the memory cost and FLOPs of the execution.
    Unfortunately, backward memory cost and FLOPs are estimated results.

    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn.functional` are available.

    Examples:
        >>> input = torch.rand(100, 100, 100, 100, device='meta')
        >>> func = torch.nn.functional.relu
        >>> output, (fwd_flop, bwd_flop), (fwd_tmp, fwd_out, bwd_tmp, bwd_out) = profile_function(func)(input, inplace=False)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        assert meta_profiler_function.has(target) or meta_profiler_function.has(
            target.__name__), CALL_FUNCTION_MSG.format(target)

        fwd_tmp = 0
        fwd_out = 0
        out = func(*args, **kwargs)
        if target not in INPLACE_OPS and not kwargs.get('inplace', False):
            fwd_out = activation_size(out)
        if meta_profiler_function.has(target):
            profiler = meta_profiler_function.get(target)
        else:
            profiler = meta_profiler_function.get(target.__name__)
        fwd_flop, _ = profiler(*args, **kwargs)
        return out, GraphInfo(fwd_flop, fwd_flop * 2, fwd_tmp, fwd_out, fwd_tmp + fwd_out, 0)

    f.__name__ = target.__name__
    func = target
    return f


@compatibility(is_backward_compatible=True)
def profile_method(target: 'Target') -> Callable:
    """
    Wrap a `call_method` node
    record the memory cost and FLOPs of the execution.

    Warnings:
        This is not fully implemented and you may follow the error message to debug.
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'

        out = getattr(self_obj, target)(*args_tail, **kwargs)
        assert target in INPLACE_METHOD + NON_INPLACE_METHOD, CALL_METHOD_MSG.format(
            target, INPLACE_METHOD, NON_INPLACE_METHOD)
        # call_method has no parameters and are MOSTLY(?) inplace, and has no FLOPs or MACs.
        fwd_tmp = 0 if target in INPLACE_METHOD else activation_size(out)
        fwd_out = 0 if target not in INPLACE_METHOD else activation_size(out)
        return out, GraphInfo(0, 0, fwd_tmp, fwd_out, fwd_tmp + fwd_out, 0)

    return f


@compatibility(is_backward_compatible=True)
def profile_module(module: torch.nn.Module) -> Callable:
    """
    Wrap a `call_module` node or `torch.nn` in order to
    record the memory cost and FLOPs of the execution.

    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn` are available.

    Example:
        >>> input = torch.rand(4, 3, 224, 224, device='meta')
        >>> mod = torch.nn.Conv2d(3, 128, 3)
        >>> output, (fwd_flop, bwd_flop), (fwd_tmp, fwd_out, bwd_tmp, bwd_out) = profile_module(mod)(input)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        assert meta_profiler_module.has(type(module)), CALL_MODULE_MSG.format(type(module))

        fwd_tmp = 0
        fwd_out = 0
        out = func(*args, **kwargs)
        if getattr(module, 'inplace', False):
            fwd_out = activation_size(out)
        profiler = meta_profiler_module.get(type(module))
        fwd_flop, _ = profiler(module, *args, **kwargs)
        return out, GraphInfo(fwd_flop, fwd_flop * 2, fwd_tmp, fwd_out, fwd_tmp + fwd_out, 0)

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
