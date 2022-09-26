from functools import partial
from typing import Callable, Any, Dict, Tuple
import torch
from torch.fx import Graph, Node
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_map
from .dataflow import autograd_graph_analysis, is_phase, Phase, GraphInfo
from .memory import activation_size
from .constant import ALIAS_ATEN
from .tensor import MetaTensor
from .opcount import flop_mapping

__all__ = ['profile_function', 'profile_module', 'profile_method']

# super-dainiu: this cache should be global, otherwise it cannot
# track duplicated tensors between nodes
cache = set()


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def is_autogradable(x):
    return isinstance(x, torch.Tensor) and x.is_floating_point()


# super-dainiu:
# x.detach() will change the unique identifier of data_ptr
# we need to handle this in a stupid way
def detach(x):
    if isinstance(x, torch.Tensor):
        requires_grad = x.requires_grad
        x.requires_grad_(False)
        x.requires_grad_(requires_grad)


def _profile_concrete(target: Callable, *args, **kwargs) -> Tuple[Tuple[Any, ...], GraphInfo]:
    """
    Profile a Callable function with args and kwargs on concrete devices.

    Args:
        target (Callable): A Callable function
        args (Any): Argument
        kwargs (Any): Argument

    Raises:
        NotImplementedError: TODO(yby)

    Returns:
        out (Tuple[Any, ...]): The argument value that was retrieved.
        meta_info (GraphInfo): The memory cost and FLOPs estimated with `MetaTensor`.
    """
    raise NotImplementedError


def _profile_meta(target: Callable, *args, **kwargs) -> Tuple[Tuple[Any, ...], GraphInfo]:
    """
    Profile a Callable function with args and kwargs on meta devices.

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

        _node: Node = None

        def __repr__(self):
            if self.grad_fn:
                return f"FlopTensor({self._tensor}, fake_device='{self.device}', size={tuple(self.shape)}, grad_fn={self.grad_fn})"
            return f"FlopTensor({self._tensor}, fake_device='{self.device}', size={tuple(self.shape)}, requires_grad={self.requires_grad})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            args_node = tree_map(lambda x: x._node if isinstance(x, FlopTensor) else None, args)
            kwargs_node = tree_map(lambda x: x._node if isinstance(x, FlopTensor) else None, kwargs)
            node = subgraph.create_node('call_function', func, args_node, kwargs_node)

            out = super().__torch_dispatch__(func, types, args, kwargs)

            flop_count[phase] += flop_mapping[func](args, normalize_tuple(out))
            node.meta['phase'] = phase

            # super-dainiu: in `nn.MultiheadAttention` this weird thing occurs,
            # i.e. `Phase.PLACEHOLDER` tensors are aliased and saved during
            # `Phase.FORWARD`
            if phase == Phase.FORWARD:
                if all(map(partial(is_phase, phase=Phase.PLACEHOLDER), node.all_input_nodes)) and func in ALIAS_ATEN:
                    node.meta['phase'] = Phase.PLACEHOLDER

            # TODO(yby): specify `saved_tensors` for backward memory estimation
            node.meta['saved_tensor'] = []
            if phase == Phase.BACKWARD:
                node.meta['saved_tensor'] = normalize_tuple(out)

            def wrap(x):
                if isinstance(x, MetaTensor):
                    x = FlopTensor(x)
                    x._node = node
                return x

            out = tree_map(wrap, out)
            return out

    def wrap(x):
        if isinstance(x, torch.Tensor):
            x = FlopTensor(x)
            if is_autogradable(x):
                x.requires_grad_(True)
            x._node = subgraph.create_node('placeholder',
                                           'placeholder', (subgraph._root,),
                                           name=subgraph._graph_namespace.create_name('input', x._tensor))
            x._node.meta['phase'] = Phase.PLACEHOLDER
            x._node.meta['saved_tensor'] = []
            detach(x)
        return x

    # Basically, we need to detach the args and kwargs from the outer graph.
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    def pack(x):
        global cache
        if isinstance(x, FlopTensor) and not x._tensor.data_ptr in cache:
            x._node.meta['saved_tensor'] += [x]
            cache.add(x._tensor.data_ptr)
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
        for tensor in normalize_tuple(out):
            if is_autogradable(tensor) and tensor.requires_grad:
                phase = Phase.BACKWARD
                grad = torch.empty_like(tensor._tensor, device=torch.device('meta')) if isinstance(
                    tensor, FlopTensor) else torch.empty_like(tensor, device=torch.device('meta'))
                torch.autograd.backward(tensor, FlopTensor(grad, fake_device=tensor.device), retain_graph=True)

    graph_info = autograd_graph_analysis(subgraph)
    graph_info.fwd_flop, graph_info.bwd_flop = flop_count[Phase.FORWARD], flop_count[Phase.BACKWARD]
    graph_info.fwd_mem_out = activation_size(out)

    def unwrap(x):
        return MetaTensor(x) if isinstance(x, torch.Tensor) else x

    return tree_map(unwrap, out), graph_info


def profile_function(target: 'Target', device: str = 'meta') -> Callable:
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
        # still run the profiling but discard some results regarding `target`
        inplace = kwargs.get('inplace', False)
        if inplace:
            kwargs['inplace'] = False
        if device == 'meta':
            out, meta = _profile_meta(func, *args, **kwargs)
        else:
            out, meta = _profile_concrete(func, *args, **kwargs)
        if inplace:
            if target in [torch.nn.functional.relu]:
                meta.save_fwd_in = False
                meta.bwd_mem_out = 0
        return out, meta

    f.__name__ = target.__name__
    func = target
    return f


def profile_method(target: 'Target', device: str = 'meta') -> Callable:
    """
    Wrap a `call_method` node
    record the memory cost and FLOPs of the execution. 
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        # execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'
        if device == 'meta':
            out, meta = _profile_meta(target, *args, **kwargs)
        else:
            out, meta = _profile_concrete(target, *args, **kwargs)
        return out, meta

    return f


def profile_module(module: torch.nn.Module, device: str = 'meta') -> Callable:
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
        # still run the profiling but discard some results regarding `module`.
        inplace = getattr(module, 'inplace', False)
        if inplace:
            module.inplace = False
        if device == 'meta':
            out, meta = _profile_meta(func, *args, **kwargs)
        else:
            out, meta = _profile_concrete(func, *args, **kwargs)
        if inplace:
            # super-dainiu: experiments on mobilenet_v2 shows that `torch.nn.ReLU`
            # is the only inplace activation function that discard its input.
            if type(module) in [torch.nn.ReLU]:
                meta.save_fwd_in = False
                meta.bwd_mem_out = 0
        return out, meta

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
