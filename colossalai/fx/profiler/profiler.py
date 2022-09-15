from typing import Callable, Any, Dict, Tuple
import torch
from torch.fx import Graph, Node
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_map
from .dataflow import GraphInfo, autograd_graph_analysis, Phase
from .memory import WEIRD_OPS
from .tensor import MetaTensor
from .opcount import flop_mapping

__all__ = ['profile_function', 'profile_module', 'profile_method']


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def is_autogradable(x):
    return isinstance(x, torch.Tensor) and x.is_floating_point()


def _profile(target: Callable, *args, **kwargs) -> Tuple[Any, ...]:
    """
    Profile a Callable function with args and kwargs.

    Args:
        target (Callable): A Callable function
        args (Any): Argument
        kwargs (Any): Argument

    Returns:
        out (Tuple[Any, ...]): The argument value that was retrieved.
        meta_info (GraphInfo): The memory cost and FLOPs estimated with `MetaTensor`.
    """
    # This subgraph traces aten level ops inside one node.
    subgraph = Graph()

    # `flop_count`` serves as a global dictionary to store results.
    flop_count = {
        Phase.FORWARD: 0,
        Phase.BACKWARD: 0,
    }

    # FlopTensor not only get the flop statistics of a single node,
    # it also build a full autograd graph for this node.
    # This makes sure we can analyze the dependencies of memory, and
    # decide which forward intermediate results should be kept until
    # backward is executed.
    # Hopefully, this attempt will provide a better estimation of memory.
    class FlopTensor(MetaTensor):

        _node: Node

        def __repr__(self):
            if self.grad_fn:
                return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)}, grad_fn={self.grad_fn})"
            return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def get_node(x):
                return None if not hasattr(x, '_node') else x._node

            args_node = tree_map(get_node, args)
            kwargs_node = tree_map(get_node, kwargs)
            node = subgraph.create_node('call_function', func, args_node, kwargs_node)

            # do not allocate on `cpu`
            if 'device' in kwargs:
                kwargs['device'] = 'meta'

            def unwrap(x):
                # if x is a `nn.Parameter`, we can first wrap it with `FlopTensor`
                if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor'):
                    x = FlopTensor(x.to('meta'))
                return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)
            flop_count[phase] += flop_mapping[func](args, normalize_tuple(out))
            node.meta['out'] = normalize_tuple(out)
            node.meta['phase'] = phase

            def wrap(x):
                return FlopTensor(x.to('meta')) if isinstance(x, torch.Tensor) else x

            def set_node(x):
                x._node = node

            out = tree_map(wrap, out)
            tree_map(set_node, out)
            return out

    # `WEIRD_OPS` are tough to handle because they don't accept autograd
    #  on meta tensor.
    if target not in WEIRD_OPS:

        def wrap(x):
            return FlopTensor(
                x.detach().requires_grad_(True)) if is_autogradable(x) and not hasattr(x, '_tensor') else x
    else:

        def wrap(x):
            return FlopTensor(
                x.detach().requires_grad_(False)) if is_autogradable(x) and not hasattr(x, '_tensor') else x

    # Basically, we need to detach the args and kwargs from the outer graph.
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    def set_placeholder(x):
        if isinstance(x, FlopTensor):
            x._node = subgraph.create_node('placeholder',
                                           'placeholder', (subgraph._root,),
                                           name=subgraph._graph_namespace.create_name('input', x._tensor))
            x._node.meta['phase'] = Phase.PLACEHOLDER
            x._node.meta['out'] = (x._tensor,)

    tree_map(set_placeholder, args)
    tree_map(set_placeholder, kwargs)

    def pack(x):
        if isinstance(x, FlopTensor) and not isinstance(x, torch.nn.Parameter):
            x._node.meta['saved'] = True
        return x

    def unpack(x):
        return x

    # `phase` will mark the phase of autograd from outside scope.
    phase = Phase.FORWARD
    # mark saved tensors with saved_tensors_hooks
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        if isinstance(target, str):
            # args[0] is the `self` object for this method call
            self_obj, *args_tail = args
            out = getattr(self_obj, target)(*args_tail, **kwargs)
        else:
            out = target(*args, **kwargs)

        # If the output is not a floating point `torch.Tensor` or it does not
        # requires grad, then we should not run backward for this node.
        if is_autogradable(out) and out.requires_grad:
            phase = Phase.BACKWARD
            if isinstance(out, FlopTensor):
                out._node.meta['save'] = False
            grad = torch.empty_like(out._tensor, device='meta') if isinstance(out, FlopTensor) else torch.empty_like(
                out, device='meta')
            torch.autograd.backward(out, FlopTensor(grad))

    graph_info = autograd_graph_analysis(subgraph)
    graph_info.fwd_flop, graph_info.bwd_flop = flop_count[Phase.FORWARD], flop_count[Phase.BACKWARD]

    def unwrap(x):
        return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

    return tree_map(unwrap, out), graph_info


def profile_function(target: 'Target') -> Callable:
    """
    Wrap a `call_function` node or `torch.nn.functional` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn.functional` are available.
    
    Examples:
        >>> input = torch.rand(100, 100, 100, 100, device='meta')
        >>> func = torch.nn.functional.relu
        >>> output, meta_info = profile_function(func)(input)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:

        # If there is an argument that this `call_function` is inplace, we should
        # skip the autograd profiling.
        if kwargs.get('inplace', False):
            args = tree_map(lambda x: x.to('meta') if isinstance(x, torch.Tensor) else x, args)
            kwargs = tree_map(lambda x: x.to('meta') if isinstance(x, torch.Tensor) else x, kwargs)
            out = func(*args, **kwargs)
            return out, GraphInfo(out.numel(), out.numel(), 0, 0, 0, 0)
        out, meta = _profile(func, *args, **kwargs)
        return out, meta

    f.__name__ = target.__name__
    func = target
    return f


def profile_method(target: 'Target') -> Callable:
    """
    Wrap a `call_method` node
    record the memory cost and FLOPs of the execution. 
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        # execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'
        out, meta = _profile(target, *args, **kwargs)
        return out, meta

    return f


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
        >>> output, meta_info = profile_module(mod)(input)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:

        # If there is an argument that this `call_module` is inplace, we should
        # skip the autograd profiling.
        if getattr(module, 'inplace', False):
            args = tree_map(lambda x: x.to('meta'), args)
            kwargs = tree_map(lambda x: x.to('meta'), kwargs)
            out = func(*args, **kwargs)
            return out, GraphInfo(out.numel(), out.numel(), 0, 0, 0, 0)
        out, meta = _profile(func, *args, **kwargs)
        return out, meta

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
