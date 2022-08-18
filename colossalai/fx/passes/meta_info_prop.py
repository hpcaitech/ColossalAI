import torch
import torch.fx
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict
from functools import reduce
from torch.fx._compatibility import compatibility
from torch.fx.immutable_collections import immutable_dict, immutable_list


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


def _compute_activation_size(node_metadata: any) -> int:
    """
    Compute numel of a node with ``tensor_meta`` attribute.
    """
    node_numel = 0

    if isinstance(node_metadata, TensorMetadata):
        node_numel += node_metadata.numel * torch.tensor([], dtype=node_metadata.dtype).element_size()
    elif isinstance(node_metadata, dict):
        value_list = [v for _, v in node_metadata.items()]
        node_numel += _compute_activation_size(value_list)
    else:
        for element in node_metadata:
            node_numel += _compute_activation_size(element)

    return node_numel


def _map_aggregate(arg, fn):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(arg, torch.Size):
        return fn(arg)
    if isinstance(arg, tuple):
        return tuple(map_aggregate(elem, fn) for elem in arg)
    elif isinstance(arg, list):
        return immutable_list(map_aggregate(elem, fn) for elem in arg)
    elif isinstance(arg, dict):
        return immutable_dict((k, map_aggregate(v, fn)) for k, v in arg.items())
    elif isinstance(arg, slice):
        return slice(map_aggregate(arg.start, fn), map_aggregate(arg.stop, fn), map_aggregate(arg.step, fn))
    else:
        return fn(arg)


@compatibility(is_backward_compatible=True)
class MetaInfoProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
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

    def run_node(self, n: Node) -> Any:
        # TODO: We might run_node(n) with meta data, and count FLOPS for each node
        result = super().run_node(n)

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                return _extract_tensor_metadata(obj)
            else:
                return TensorMetadata(None, None, False, None, 0, False)

        meta = _map_aggregate(result, extract_tensor_meta)
        n.meta['tensor_meta'] = meta

        total_activation_size = 0
        total_param_size = 0
        if n.op == 'call_module':
            target_module = n.graph.owning_module.get_submodule(n.target)
            if not getattr(target_module, 'inplace', False):
                total_activation_size = _compute_activation_size(n.meta['tensor_meta'])
            for param in target_module.parameters():
                total_param_size += param.numel() * torch.tensor([], dtype=param.dtype).element_size()
        elif n.op == 'call_function':
            if 'inplace' not in n.kwargs:
                total_activation_size = _compute_activation_size(n.meta['tensor_meta'])
        else:
            total_activation_size = _compute_activation_size(n.meta['tensor_meta'])

        setattr(n, 'node_size', total_activation_size + total_param_size)
        setattr(n, 'param_size', total_param_size)
        setattr(n, 'activation_size', total_activation_size)
        n.meta['type'] = type(result)
        return result

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
