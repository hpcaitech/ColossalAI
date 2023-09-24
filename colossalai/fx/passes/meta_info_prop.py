from dataclasses import asdict
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
import torch.fx
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_map

from colossalai.fx._compatibility import compatibility, is_compatible_with_meta
from colossalai.fx.profiler import (
    GraphInfo,
    activation_size,
    calculate_fwd_in,
    calculate_fwd_out,
    calculate_fwd_tmp,
    profile_function,
    profile_method,
    profile_module,
)


@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int]
    numel: int
    is_tensor: bool
    # TODO: we can add a list of sharding spec here, and record the sharding
    # behavior by appending sharding spec into list.


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()
    numel = result.numel()
    is_tensor = True

    return TensorMetadata(shape, dtype, requires_grad, stride, numel, is_tensor)


@compatibility(is_backward_compatible=True)
class MetaInfoProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node with meta tensor and
    record the memory usage, FLOPs, and type of the result
    into the corresponding node.

    Usage:
        BATCH_SIZE = 2
        DIM_IN = 4
        DIM_HIDDEN = 16
        DIM_OUT = 16
        model = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, DIM_HIDDEN),
            torch.nn.Linear(DIM_HIDDEN, DIM_OUT),
            )
        input_sample = torch.rand(BATCH_SIZE, DIM_IN)
        gm = symbolic_trace(model)
        interp = MetaInfoProp(gm)
        interp.run(input_sample)
        print(interp.summary(format='kb'))    # don't panic if some statistics are 0.00 MB


        # output of above code is
            Op type       Op    Forward FLOPs    Backward FLOPs    FWD_OUT    FWD_TMP    BWD_OUT    BWD_TMP
        -----------  -------  ---------------  ----------------  ---------  ---------  ---------  ---------
        placeholder  input_1          0 FLOPs           0 FLOPs    0.00 KB    0.00 KB    0.00 KB    0.00 KB
        call_module       _0        128 FLOPs         288 FLOPs    0.12 KB    0.00 KB    0.34 KB    0.00 KB
        call_module       _1        512 FLOPs       1,056 FLOPs    0.12 KB    0.00 KB    1.19 KB    0.00 KB
             output   output          0 FLOPs           0 FLOPs    0.00 KB    0.00 KB    0.00 KB    0.00 KB
    Args:
         module (GraphModule): The module to be executed

    """

    _is_proped: bool = False

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        self._is_proped = True
        result, meta_info = super().run_node(n)

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                return _extract_tensor_metadata(obj)
            else:
                return TensorMetadata(None, None, False, None, 0, False)

        tensor_meta = tree_map(extract_tensor_meta, result)
        n.meta["tensor_meta"] = tensor_meta
        n.meta = {**n.meta, **asdict(meta_info)}  # extend MetaInfo to `n.meta`
        # TODO: the attribute node_size should be removed in the future
        setattr(n, "node_size", activation_size(n.meta.get("fwd_out", 0)) + activation_size(n.meta.get("fwd_tmp", 0)))
        setattr(n, "fwd_flop", n.meta.get("fwd_flop", 0))
        setattr(n, "bwd_flop", n.meta.get("bwd_flop", 0))
        n.meta["type"] = type(result)

        # retain the autograd graph
        for param in self.module.parameters():
            param.grad = None

        return result

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        return super().placeholder(target, args, kwargs), GraphInfo()

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        return super().get_attr(target, args, kwargs), GraphInfo()

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        assert not isinstance(target, str)
        return profile_function(target)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        return profile_method(target)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        # Retrieve executed args and kwargs values from the environment
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return profile_module(submod)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        if hasattr(args[0], "_tensor"):
            return args[0], GraphInfo(fwd_in=[args[0]._tensor])
        return args[0], GraphInfo(save_fwd_in=True)

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)

    def summary(self, unit: str = "MB") -> str:
        """
        Summarizes the memory and FLOPs statistics of the `GraphModule` in
        tabular format. Note that this API requires the ``tabulate`` module
        to be installed.
        """
        # https://github.com/pytorch/pytorch/blob/master/torch/fx/graph.py
        try:
            from tabulate import tabulate
        except ImportError:
            print(
                "`summary` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library."
            )

        assert self._is_proped, "Please call `interp.run(input)` before calling `interp.summary()`."

        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []

        def mem_repr(mem: int) -> str:
            unit_divisor_map = {
                "kb": 1024,
                "mb": 1024**2,
                "gb": 1024**3,
                "tb": 1024**4,
            }
            return f"{mem / unit_divisor_map[unit.lower()]:.2f} {unit.upper()}"

        def flops_repr(flop: int) -> str:
            return f"{flop:,} FLOPs"

        accumulate_size = 0
        for node in self.module.graph.nodes:
            node: Node
            accumulate_size += calculate_fwd_out(node) + calculate_fwd_tmp(node)
            node_summaries.append(
                [
                    node.op,
                    str(node),
                    flops_repr(node.meta["fwd_flop"]),
                    flops_repr(node.meta["bwd_flop"]),
                    mem_repr(accumulate_size),
                    mem_repr(calculate_fwd_in(node)),
                    mem_repr(calculate_fwd_out(node)),
                    mem_repr(calculate_fwd_tmp(node)),
                    mem_repr(node.meta["bwd_mem_out"]),
                    mem_repr(node.meta["bwd_mem_tmp"]),
                ]
            )

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = [
            "Op type",
            "Op",
            "Forward FLOPs",
            "Backward FLOPs",
            "Accumulated Memory",
            "FWD_IN",
            "FWD_OUT",
            "FWD_TMP",
            "BWD_OUT",
            "BWD_TMP",
        ]

        return tabulate(node_summaries, headers=headers, stralign="right")


def metainfo_trace(gm: torch.fx.GraphModule, *args, verbose: bool = False, unit: str = "MB", **kwargs) -> None:
    """
    MetaInfo tracing API

    Given a ``GraphModule`` and a sample input, this API will trace the MetaInfo of a single training cycle,
    and annotate them on ``gm.graph``.

    Uses:
        >>> model = ...
        >>> gm = symbolic_trace(model)
        >>> args = ...  # sample input to the ``GraphModule``
        >>> metainfo_trace(gm, *args)

    Args:
        gm (torch.fx.GraphModule): The ``GraphModule`` to be annotated with MetaInfo.
        verbose (bool, optional): Whether to show ``MetaInfoProp.summary()`. Defaults to False.
        unit (str, optional): The unit of memory. Defaults to "MB".

    Returns:
        torch.fx.GraphModule: The ``GraphModule`` annotated with MetaInfo.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    interp = MetaInfoProp(gm.to(device))
    if is_compatible_with_meta():
        from colossalai.fx.profiler import MetaTensor

        args = tree_map(lambda x: MetaTensor(x, fake_device=device), args)
        kwargs = tree_map(lambda x: MetaTensor(x, fake_device=device), kwargs)
    interp.propagate(*args, **kwargs)
    if verbose:
        interp.summary(unit)
    gm.to("cpu")
    del interp
    return gm
