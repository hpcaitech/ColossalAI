import torch
import torch.fx
from torch.fx.node import Node, Argument, Target
from torch.utils._pytree import tree_map
from typing import Any, Tuple, NamedTuple, Dict
from torch.fx._compatibility import compatibility
from colossalai.fx.profiler import profile_function, profile_module, profile_method, activation_size, parameter_size


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
    # behaviour by appending sharding spec into list.


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
    record the shape, FLOPs, MACs and type of the result
    into the corresponding node.

    Usage:
        BATCH_SIZE = 2
        DIM_IN = 4
        DIM_OUT = 16
        model = torch.nn.Linear(DIM_IN, DIM_OUT)
        input_sample = torch.rand(BATCH_SIZE, DIM_IN)
        orig_output = model(input_sample)
        gm = symbolic_trace(model)
        MetaInfoProp(gm).run(input_sample)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape, node.meta['tensor_meta'].numel)
        
        # output of above code is 
        # input_1 torch.float32 torch.Size([2, 4]) 8
        # weight torch.float32 torch.Size([16, 4]) 64
        # bias torch.float32 torch.Size([16]) 16
        # linear torch.float32 torch.Size([2, 16]) 32
        # output torch.float32 torch.Size([2, 16]) 32
    Args:
         module (GraphModule): The module to be executed

    """

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
        result, flop_count, mem_stat = super().run_node(n)

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                return _extract_tensor_metadata(obj)
            else:
                return TensorMetadata(None, None, False, None, 0, False)

        meta = tree_map(extract_tensor_meta, result)
        n.meta['tensor_meta'] = meta

        # TODO: the attribute node_size should be removed in the future
        setattr(n, 'node_size', mem_stat[1])
        setattr(n, 'fwd_flop', flop_count[0])
        setattr(n, 'bwd_flop', flop_count[1])
        setattr(n, 'fwd_tmp', mem_stat[0])
        setattr(n, 'fwd_out', mem_stat[1])
        setattr(n, 'bwd_tmp', mem_stat[2])
        setattr(n, 'bwd_out', mem_stat[3])
        n.meta['type'] = type(result)

        for param in self.module.parameters():
            param.grad = None
        return result

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        result = super().placeholder(target, args, kwargs)
        # A placeholder node only has activation
        return result, (0, 0), (0, activation_size(result), 0, 0)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        return super().get_attr(target, args, kwargs), (0, 0), (0, 0, 0, 0)

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        assert not isinstance(target, str)
        return profile_function(target)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        return profile_method(target)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        # Retrieve executed args and kwargs values from the environment
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return profile_module(submod)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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
            flop_count (Tuple): The flop count for (fwd_flop, bwd_flop).
            mem_stat (Tuple): The memory statistics for (fwd_tmp, fwd_out, bwd_tmp, bwd_out)
        """
        return args[0], (0, 0), (0, 0, 0, 0)

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
