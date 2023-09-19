import time
from functools import partial
from typing import Any, Callable, Dict, Tuple

import torch
from torch.fx import Graph, Node
from torch.fx.node import Argument, Target
from torch.nn.parameter import Parameter
from torch.utils._pytree import tree_map

from .._compatibility import compatibility
from .constants import ALIAS_ATEN, OUTPUT_SAVED_MOD, OUTPUT_SAVED_OPS
from .dataflow import GraphInfo, Phase, autograd_graph_analysis, is_phase
from .memory_utils import activation_size, parameter_size
from .opcount import flop_mapping
from .tensor import MetaTensor

__all__ = ["profile_function", "profile_module", "profile_method"]

# super-dainiu: this cache should be global, otherwise it cannot
# track duplicated tensors between nodes
cache = set()

# a global identifier for inplace ops
do_not_cache = False


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def is_autogradable(x):
    return isinstance(x, torch.Tensor) and x.is_floating_point()


def detach_variables(x):
    if isinstance(x, torch.Tensor):
        requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad = requires_grad

    return x


@compatibility(is_backward_compatible=True)
def _profile_concrete(target: Callable, *args, **kwargs) -> Tuple[Tuple[Any, ...], GraphInfo]:
    """Profile a Callable function with args and kwargs on concrete devices by https://github.com/Cypher30
    To profile the actual forward memory, we first run target in the context torch.no_grad() to get
    the fwd_mem_out, then we run target with grad enable to found the extra memory stored in the memory
    by memory allocated minus the fwd_mem_out.
    To profile the actual backward memory, we first make dummy gradient for torch.autograd.backward, then
    find the bwd_mem_tmp with memory peak during the process minus bwd_mem_out(it is actually equal to size
    of args and kwargs).
    We also add time stamps to profile the real forward and backward time.

    Args:
        target (Callable): A Callable function
        args (Any): Arguments
        kwargs (Any): Arguments

    Returns:
        Tuple[Tuple[Any, ...], GraphInfo]: Output for next node & memory cost and real forward and backward
        time.
    """

    graphinfo = GraphInfo()

    # detach input from the graph
    args = tree_map(detach_variables, args)
    kwargs = tree_map(detach_variables, kwargs)
    if isinstance(target, str):
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # calculate fwd_mem_out
        mem_stamp0 = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = getattr(self_obj, target)(*args_tail, **kwargs)
        mem_stamp1 = torch.cuda.memory_allocated()
        graphinfo.fwd_mem_out = mem_stamp1 - mem_stamp0
        del out

        # calculate fwd_mem_tmp & fwd_time
        mem_stamp0 = torch.cuda.memory_allocated()
        fwd_time0 = time.time()
        out = getattr(self_obj, target)(*args_tail, **kwargs)
        fwd_time1 = time.time()
        graphinfo.fwd_time = fwd_time1 - fwd_time0
        mem_stamp1 = torch.cuda.memory_allocated()
        graphinfo.fwd_mem_tmp = mem_stamp1 - mem_stamp0 - graphinfo.fwd_mem_out

        # calculate bwd_mem_tmp & bwd_time
        grad_tensors = tree_map(lambda x: torch.ones_like(x) if isinstance(x, torch.Tensor) else None, out)
        torch.cuda.reset_peak_memory_stats()
        mem_stamp0 = torch.cuda.memory_allocated()
        bwd_time0 = time.time()
        torch.autograd.backward(out, grad_tensors=grad_tensors)
        bwd_time1 = time.time()
        graphinfo.bwd_time = bwd_time1 - bwd_time0
        mem_stamp1 = torch.cuda.max_memory_allocated()

        # calculate bwd memory stats
        # NOTE: the module should add param to bwd_mem_out for bwd_mem_tmp calculation
        graphinfo.bwd_mem_out = activation_size(args) + activation_size(kwargs)
        graphinfo.bwd_mem_out += parameter_size(target.__self__) if hasattr(target.__self__, "parameters") else 0
        graphinfo.bwd_mem_tmp = mem_stamp1 - mem_stamp0 - graphinfo.bwd_mem_out

    else:
        # calculate fwd_mem_out
        mem_stamp0 = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = target(*args, **kwargs)
        mem_stamp1 = torch.cuda.memory_allocated()
        graphinfo.fwd_mem_out = mem_stamp1 - mem_stamp0
        del out

        # calculate fwd_mem_tmp & fwd_time
        mem_stamp0 = torch.cuda.memory_allocated()
        fwd_time0 = time.time()
        out = target(*args, **kwargs)
        fwd_time1 = time.time()
        graphinfo.fwd_time = fwd_time1 - fwd_time0
        mem_stamp1 = torch.cuda.memory_allocated()
        graphinfo.fwd_mem_tmp = mem_stamp1 - mem_stamp0 - graphinfo.fwd_mem_out

        # calculate bwd_mem_tmp & bwd_time
        grad_tensors = tree_map(lambda x: torch.ones_like(x) if isinstance(x, torch.Tensor) else None, out)
        torch.cuda.reset_peak_memory_stats()
        mem_stamp0 = torch.cuda.memory_allocated()
        bwd_time0 = time.time()
        torch.autograd.backward(out, grad_tensors=grad_tensors)
        bwd_time1 = time.time()
        graphinfo.bwd_time = bwd_time1 - bwd_time0
        mem_stamp1 = torch.cuda.max_memory_allocated()

        # calculate bwd memory stats
        # NOTE: the module should add param to bwd_mem_out for bwd_mem_tmp calculation
        graphinfo.bwd_mem_out = activation_size(args) + activation_size(kwargs)
        graphinfo.bwd_mem_out += parameter_size(target.__self__) if hasattr(target.__self__, "parameters") else 0
        graphinfo.bwd_mem_tmp = mem_stamp1 - mem_stamp0 - graphinfo.bwd_mem_out

    return tree_map(detach_variables, out), graphinfo


@compatibility(is_backward_compatible=False)
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
            node = subgraph.create_node("call_function", func, args_node, kwargs_node)

            out = super().__torch_dispatch__(func, types, args, kwargs)

            flop_count[phase] += flop_mapping[func](args, normalize_tuple(out))
            node.meta["phase"] = phase

            # super-dainiu: in `nn.MultiheadAttention` this weird thing occurs,
            # i.e. `Phase.PLACEHOLDER` tensors are aliased and saved during
            # `Phase.FORWARD`
            if phase == Phase.FORWARD:
                if all(map(partial(is_phase, phase=Phase.PLACEHOLDER), node.all_input_nodes)) and func in ALIAS_ATEN:
                    node.meta["phase"] = Phase.PLACEHOLDER

            # TODO(yby): specify `saved_tensors` for backward memory estimation
            node.meta["saved_tensor"] = []
            if phase == Phase.BACKWARD:
                node.meta["saved_tensor"] = normalize_tuple(out)

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
            x._node = subgraph.create_node(
                "placeholder",
                "placeholder",
                (subgraph._root,),
                name=subgraph._graph_namespace.create_name("input", x._tensor),
            )
            x._node.meta["phase"] = Phase.PLACEHOLDER
            x._node.meta["saved_tensor"] = []
        return x

    # Basically, we need to detach the args and kwargs from the outer graph.
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    def pack(x):
        global cache, do_not_cache
        if isinstance(x, FlopTensor) and not x._tensor.data_ptr() in cache:
            tensor = x._tensor.detach()
            tensor.data_ptr = x._tensor.data_ptr
            x._node.meta["saved_tensor"] += [tensor]
            if not do_not_cache:
                cache.add(x._tensor.data_ptr())
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
        if all(map(lambda x: is_autogradable(x) and x.requires_grad, normalize_tuple(out))):
            grad_out = [torch.zeros_like(t) for t in normalize_tuple(out)]
            phase = Phase.BACKWARD
            torch.autograd.backward(
                out,
                grad_out,
            )

    graph_info = autograd_graph_analysis(subgraph)
    graph_info.fwd_flop, graph_info.bwd_flop = flop_count[Phase.FORWARD], flop_count[Phase.BACKWARD]

    def extract_tensor(x: Any):
        if isinstance(x, MetaTensor):
            tensor = x._tensor.detach()
            tensor.data_ptr = x._tensor.data_ptr
            return tensor
        if not isinstance(x, torch.finfo):
            return x

    graph_info.fwd_out = list(map(extract_tensor, normalize_tuple(out)))

    def unwrap(x):
        return MetaTensor(x) if isinstance(x, torch.Tensor) else x

    return tree_map(unwrap, out), graph_info


@compatibility(is_backward_compatible=True)
def profile_function(target: "Target", device: str = "meta") -> Callable:
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
        # find the grad for parameter in args and kwargs
        param_size = 0

        def get_param_size(x):
            nonlocal param_size
            if isinstance(x, Parameter):
                param_size += activation_size(x)

        tree_map(get_param_size, args)
        tree_map(get_param_size, kwargs)

        # If there is an argument that this `call_function` is inplace, we should
        # still run the profiling but discard some results regarding `target`
        global do_not_cache

        inplace = kwargs.get("inplace", False)
        if target in OUTPUT_SAVED_OPS:
            do_not_cache = True
        if inplace:
            do_not_cache = True
            kwargs["inplace"] = False
        if device == "meta":
            out, meta = _profile_meta(func, *args, **kwargs)
        else:
            out, meta = _profile_concrete(func, *args, **kwargs)
        if inplace:
            kwargs["inplace"] = True
            meta.bwd_mem_tmp = 0
            meta.bwd_mem_out = 0
        do_not_cache = False

        meta.bwd_mem_out -= param_size
        return out, meta

    f.__name__ = target.__name__
    func = target
    return f


@compatibility(is_backward_compatible=True)
def profile_method(target: "Target", device: str = "meta") -> Callable:
    """
    Wrap a `call_method` node
    record the memory cost and FLOPs of the execution.
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        # execute the method and return the result
        assert isinstance(target, str), f"{target} instance is not str."
        if device == "meta":
            out, meta = _profile_meta(target, *args, **kwargs)
        else:
            out, meta = _profile_concrete(target, *args, **kwargs)
        return out, meta

    return f


@compatibility(is_backward_compatible=True)
def profile_module(module: torch.nn.Module, device: str = "meta") -> Callable:
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
        # calculate parameter size
        param_size = parameter_size(module)

        # If there is an argument that this `call_module` is inplace, we should
        # still run the profiling but discard some results regarding `module`.
        global do_not_cache

        inplace = getattr(module, "inplace", False)
        if type(module) in OUTPUT_SAVED_MOD:
            do_not_cache = True
        if inplace:
            do_not_cache = True
            module.inplace = False
        if device == "meta":
            out, meta = _profile_meta(func, *args, **kwargs)
        else:
            out, meta = _profile_concrete(func, *args, **kwargs)
        if inplace:
            module.inplace = True
            meta.bwd_mem_tmp = 0
            meta.bwd_mem_out = 0
        do_not_cache = False

        # grad for param will not be counted
        meta.bwd_mem_out -= param_size
        return out, meta

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
