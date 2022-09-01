from operator import add, floordiv, getitem, mul, neg, setitem, sub, pos
from typing import Callable, List, NamedTuple, Any, Dict, Tuple, Union
import torch
from torch.fx.node import Argument, Target
from torch.fx._compatibility import compatibility
from torch.utils._pytree import tree_map, tree_flatten
from .tensor import MetaTensor
from .opcount import flop_mapping
from .memory import activation_size

__all__ = ['profile_function', 'profile_module', 'profile_method', '_profile']


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _profile(target: Callable, args, kwargs) -> Tuple[Any, ...]:
    """Profile a Callable function with args and kwargs.

    Args:
        target (Callable): A Callable function
        args (Any): Argument
        kwargs (Any): Argument

    Returns:
        out (Tuple[Any, ...]): The argument value that was retrieved
        flop_count (Tuple[int, ...]): The flop count for (fwd_flop, bwd_flop).
        mem_stat (Tuple[int, ...]): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
    """

    flop_count = {
        'f': 0,
        'l': 0,
        'b': 0,
    }
    temp = {
        'f': [],
        'l': [],
        'b': [],
    }
    stage = 'f'

    class FlopTensor(MetaTensor):

        def __repr__(self):
            if self.grad_fn:
                return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)}, grad_fn={self.grad_fn})"
            return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def unwrap(x):
                if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor'):
                    x = FlopTensor(x.to('meta'))
                return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

            def to_meta(x):
                return x.to('meta')

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)

            # TODO: this will be, but we should examine all aten ops first
            # if func in flop_mapping:
            #     flop_count[stage] += flop_mapping[func](args, normalize_tuple(out))
            flop_count[stage] += flop_mapping[func](args, normalize_tuple(out))
            temp[stage].append(tree_map(to_meta, normalize_tuple(out)))

            def wrap(x):
                return FlopTensor(x.to('meta')) if isinstance(x, torch.Tensor) else x

            return tree_map(wrap, out)

    def wrap(x):
        return FlopTensor(
            x.detach().requires_grad_(True)) if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor') else x

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    out = target(*args, **kwargs)
    stage = 'l'
    loss = out.sum()
    stage = 'b'
    loss.backward()

    fwd_flop = flop_count['f']
    bwd_flop = flop_count['b']

    fwd_tmp = activation_size(temp['f'][:-1])
    fwd_out = activation_size(temp['f'][-1]) if len(temp['f']) else 0
    bwd_tmp = activation_size(temp['b'])

    def unwrap(x):
        return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

    return tree_map(unwrap, out), (fwd_flop, bwd_flop), (fwd_tmp, fwd_out, bwd_tmp, 0)


def profile_function(target: 'Target') -> Callable:
    """
    Wrap a `call_function` node or `torch.nn.functional` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn.functional` are available.
    
    Examples:
        >> input = torch.rand(100, 100, 100, 100, device='meta')
        >> func = torch.nn.functional.relu
        >> output, (fwd_flop, bwd_flop), (fwd_tmp, fwd_out, bwd_tmp, bwd_out) = profile_function(func)(input, inplace=False)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        out, flop_count, mem_stat = _profile(func, args, kwargs)
        return out, flop_count, mem_stat

    f.__name__ = target.__name__
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

        # execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'

        out = getattr(self_obj, target)(args_tail, kwargs)
        return out, (0, 0), (0, activation_size(out), activation_size(out), 0)

    return f


def profile_module(module: torch.nn.Module) -> Callable:
    """
    Wrap a `call_module` node or `torch.nn` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn` are available.
    
    Example:
        >> input = torch.rand(4, 3, 224, 224, device='meta')
        >> mod = torch.nn.Conv2d(3, 128, 3)
        >> output, (fwd_flop, bwd_flop), (fwd_tmp, fwd_out, bwd_tmp, bwd_out) = profile_module(mod)(input)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        out, flop_count, mem_stat = _profile(func, args, kwargs)
        return out, flop_count, mem_stat

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
