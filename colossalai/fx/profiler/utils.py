from functools import partial
from operator import add, floordiv, getitem, mul, neg, setitem, sub, pos
from typing import Callable, NamedTuple, Any, Dict, Tuple
import torch
from torch.fx.node import Argument, Target
from torch.fx._compatibility import compatibility
from colossalai.fx.tracer.meta_patch import meta_patched_function, meta_patched_module
from . import meta_profiler_function, meta_profiler_module

__all__ = [
    'MetaProfile', 'profile_function', 'profile_module', 'profile_method', 'calculate_activation_size',
    'calculate_param_size'
]

# TODO fill out the inplace ops
INPLACE_OPS = [
    add,
    sub,
    mul,
    floordiv,
    neg,
    pos,
    getitem,
    setitem,
    torch.Tensor.cpu,
]

# TODO check that call_methods are indeed inplace
INPLACE_METHOD = [
    'transpose',
    'permute',
]


@compatibility(is_backward_compatible=True)
class MetaProfile(NamedTuple):
    # MetaProfile is a structure containing pertinent information
    # about a node within a torch.fx GraphModule.

    param: int
    activation: int
    flops: int
    macs: int


def calculate_activation_size(activation: any) -> int:
    """
    Calculate activation size of a node.
    """
    activation_size = 0
    if isinstance(activation, torch.Tensor):
        activation_size += activation.numel() * torch.tensor([], dtype=activation.dtype).element_size()
    elif isinstance(activation, dict):
        value_list = [v for _, v in activation.items()]
        activation_size += calculate_activation_size(value_list)
    else:
        for element in activation:
            activation_size += calculate_activation_size(element)
    return activation_size


def calculate_param_size(mod: torch.nn.Module) -> int:
    """
    Calculate param size of a node.
    """
    param_size = 0
    for param in mod.parameters():
        param_size += param.numel() * torch.tensor([], dtype=param.dtype).element_size()
    return param_size


def profile_function(target: 'Target') -> Callable:
    """
    Wrap a `call_function` node or `torch.nn.functional` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn.functional` are available.
    
    Usage:
        input = torch.rand(100, 100, 100, 100, device='meta')
        func = torch.nn.functional.relu
        output, profile = profile_function(func)(input, inplace=False)
        print(f"Profiling function {func},")
        print(f"Param size: {profile.param / 1024**2:.3f} MB, Activation size: {profile.activation / 1024**2:.3f} MB, {profile.flops} FLOPs, {profile.macs} MACs")
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        assert meta_profiler_function.has(target) or meta_profiler_function.has(
            target.__name__), f"Colossal-AI hasn't supported profiling for {target}, you might manually patch it."

        # call_function has no parameters
        param_size = 0
        activation_size = 0
        result = func(*args, **kwargs)
        if target not in INPLACE_OPS and not kwargs.get('inplace', False):
            activation_size += calculate_activation_size(result)
        if meta_profiler_function.has(target):
            profiler = meta_profiler_function.get(target)
        else:
            profiler = meta_profiler_function.get(target.__name__)
        flops, macs = profiler(*args, **kwargs)
        return result, MetaProfile(param_size, activation_size, flops, macs)

    f.__name__ = target.__name__
    # fetch patched function
    if meta_patched_function.has(target):
        func = meta_patched_function.get(target)
    elif meta_patched_function.has(target.__name__):
        func = meta_patched_function.get(target.__name__)
    else:
        func = target
    return f


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

        # Execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'
        result = getattr(self_obj, target)(*args_tail, **kwargs)
        assert target in INPLACE_METHOD, f'Please check {target} is an inplace method. If so, add target to INPLACE_METHOD={INPLACE_METHOD}.'

        # call_method has no parameters and are MOSTLY(?) inplace, and has no FLOPs or MACs.
        param_size = 0
        activation_size = 0
        flops = 0
        macs = 0
        return result, MetaProfile(param_size, activation_size, flops, macs)

    return f


def profile_module(module: torch.nn.Module) -> Callable:
    """
    Wrap a `call_module` node or `torch.nn` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn` are available.
    
    Usage:
        input = torch.rand(4, 3, 224, 224, device='meta')
        mod = torch.nn.Conv2d(3, 128, 3)
        output, profile = profile_module(mod)(input)
        print(f"Profiling function {mod},")
        print(f"Param size: {profile.param / 1024**2:.3f} MB, Activation size: {profile.activation / 1024**2:.3f} MB, {profile.flops} FLOPs, {profile.macs} MACs")
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        assert meta_profiler_module.has(
            type(module)), f"Colossal-AI hasn't supported profiling for {module}, you might manually patch it."
        param_size = calculate_param_size(module)
        activation_size = 0
        result = func(*args, **kwargs)
        if not getattr(module, 'inplace', False):
            activation_size += calculate_activation_size(result)
        profiler = meta_profiler_module.get(type(module))
        flops, macs = profiler(module, *args, **kwargs)
        return result, MetaProfile(param_size, activation_size, flops, macs)

    f.__name__ = module.__class__.__name__
    # fetch patched module
    if meta_patched_module.has(type(module)):
        func = partial(meta_patched_module.get(type(module)), module)
    else:
        func = module.forward
    return f
