import uuid
from dataclasses import asdict
from typing import List

import torch
import torch.fx
from torch.fx import GraphModule
from torch.fx.node import Node

from colossalai.auto_parallel.meta_profiler import MetaInfo
from colossalai.auto_parallel.passes.constants import OUTPUT_SAVED_MOD, OUTPUT_SAVED_OPS
from colossalai.fx._compatibility import compatibility
from colossalai.fx.profiler import GraphInfo


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


@compatibility(is_backward_compatible=False)
class MetaInfoProp:

    def __init__(self, module: GraphModule) -> None:
        self.module = module
        self.func_dict = {
            'placeholder': self.placeholder_handler,
            'get_attr': self.get_attr_handler,
            'output': self.output_handler,
            'call_function': self.node_handler,
            'call_module': self.node_handler,
            'call_method': self.node_handler,
        }

    def _set_data_ptr(self, x):
        """
        Set uuid to tensor
        """
        if isinstance(x, torch.Tensor):
            if not x.data_ptr():
                data_ptr = uuid.uuid4()
                x.data_ptr = lambda: data_ptr

    def _is_inplace(self, node: Node):
        """
        Check if the node is inplace operation.
        """
        if node.op == 'call_module':
            return node.graph.owning_module.get_submodule(node.target).__class__ in OUTPUT_SAVED_MOD
        elif node.op == "call_function":
            return node.target in OUTPUT_SAVED_OPS
        return False

    def run(self) -> GraphModule:
        """
        Run the meta information propagation pass on the module.
        """
        for node in self.module.graph.nodes:
            node: Node
            self.func_dict[node.op](node)

    @compatibility(is_backward_compatible=False)
    def placeholder_handler(self, node: Node) -> None:
        """
        Handle the placeholder node.
        """
        graph_info = GraphInfo()
        out = _normalize_tuple(getattr(node, '_meta_data', None))
        graph_info.fwd_out = list(out) if out[0] is not None else []
        node.meta = {**asdict(graph_info)}

    @compatibility(is_backward_compatible=False)
    def get_attr_handler(self, node: Node) -> None:
        """
        Handle the get_attr node.
        """
        graph_info = GraphInfo()
        node.meta = {**asdict(graph_info)}

    @compatibility(is_backward_compatible=False)
    def output_handler(self, node: Node) -> None:
        """
        Handle the output node.
        """
        graph_info = GraphInfo()
        output_tensors = []
        for par in node._input_nodes:
            if par.meta:
                output_tensors += par.meta["fwd_out"]
        graph_info.fwd_in = output_tensors
        node.meta = {**asdict(graph_info)}

    @compatibility(is_backward_compatible=False)
    def node_handler(self, node: Node) -> None:
        """
        Handle other kind of nodes
        """
        assert hasattr(node, 'best_metainfo'), f"Cannot find best_metainfo in node {node}, {node.op}"
        graph_info = GraphInfo()
        meta_info = node.best_metainfo
        meta_info: MetaInfo

        # set data_ptr for input_tensor in MetaInfo class
        input_tensors: List[torch.Tensor] = meta_info.fwd_in
        buffer_tensors: List[torch.Tensor] = meta_info.fwd_buffer
        output_tensors: List[torch.Tensor] = meta_info.fwd_out

        if self._is_inplace(node):
            # inplace operation will not create new tensor, and it only has one parent node
            # TODO: Verify this observation
            # set data_ptr for input_tensor, buffer_tensor and output_tensor of current node
            parent_node = list(node._input_nodes.keys())[0]
            parent_tensor = parent_node.meta.get("fwd_out")[0]
            parent_tensor: torch.Tensor
            for tensor in input_tensors:
                tensor.data_ptr = parent_tensor.data_ptr
            for tensor in buffer_tensors:
                tensor.data_ptr = parent_tensor.data_ptr
            for tensor in output_tensors:
                tensor.data_ptr = parent_tensor.data_ptr

        else:
            for par in node._input_nodes:
                # set data_ptr for the input_tensor of current node from the output_tensor of its parent node
                for tensor in par.meta.get("fwd_out", []):
                    tensor: torch.Tensor
                    target_input_tensor = next(
                        (x for x in input_tensors if not x.data_ptr() and x.shape == tensor.shape), None)
                    if target_input_tensor is not None:
                        target_input_tensor.data_ptr = tensor.data_ptr

            # set data_ptr for tensor in input_tensor that is not set
            for tensor in input_tensors:
                if not tensor.data_ptr():
                    self._set_data_ptr(tensor)

            # set data_ptr for buffer_tensor
            for tensor in buffer_tensors:
                self._set_data_ptr(tensor)

            # set data_ptr for output_tensor
            for tensor in output_tensors:
                self._set_data_ptr(tensor)

        # attach them to graph_info
        graph_info.fwd_in = input_tensors
        graph_info.fwd_tmp = buffer_tensors
        graph_info.fwd_out = output_tensors

        # fetch other memory informations
        memory_cost = meta_info.memory_cost
        graph_info.fwd_mem_tmp = memory_cost.fwd.temp
        graph_info.fwd_mem_out = memory_cost.fwd.activation
        graph_info.bwd_mem_tmp = memory_cost.bwd.temp
        graph_info.bwd_mem_out = memory_cost.bwd.activation

        # fetch flop information
        # here we use fwd_time and bwd_time to deal with the case that
        # communication cost is a float
        compute_cost = meta_info.compute_cost
        graph_info.fwd_time = compute_cost.fwd
        graph_info.bwd_time = compute_cost.bwd

        node.meta = {**asdict(graph_info)}
