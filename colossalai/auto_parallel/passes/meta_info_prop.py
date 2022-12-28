import uuid
from dataclasses import asdict
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
import torch.fx
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_map

from colossalai.auto_parallel.meta_profiler import MetaInfo
from colossalai.fx._compatibility import compatibility, is_compatible_with_meta
from colossalai.fx.profiler import GraphInfo
from colossalai.fx.profiler.constants import OUTPUT_SAVED_MOD, OUTPUT_SAVED_OPS


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
        if node.op == 'call_method':
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
        graph_info.fwd_out = list(out)
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
        assert hasattr(node, 'best_metainfo'), f"Cannot find best_metainfo in node {node}"
        graph_info = GraphInfo()
        meta_info = node.best_metainfo
        meta_info: MetaInfo

        # set data_ptr for input_tensor in MetaInfo class
        input_tensor: List[torch.Tensor] = meta_info.fwd_in
        buffer_tensor: List[torch.Tensor] = meta_info.fwd_buffer
        output_tensor: List[torch.Tensor] = meta_info.fwd_out

        if len(input_tensor) > 0:
            for par in node._input_nodes:
                if par.meta:
                    if len(par.meta["fwd_out"]) > 0:
                        # set data_ptr for the input_tensor of current node from the output_tensor of its parent node
                        for tensor in par.meta["fwd_out"]:
                            tensor: torch.Tensor
                            target_tensor = next(
                                (x for x in input_tensor if not x.data_ptr() and x.shape == tensor.shape), None)
                            target_tensor.data_ptr = tensor.data_ptr

            # set data_ptr for tensor in input_tensor that is not set
            for tensor in input_tensor:
                if not tensor.data_ptr():
                    self._set_data_ptr(tensor)

        # attach it to graph_info
        graph_info.fwd_in = input_tensor

        if self._is_inplace(node):
            # inplace operation will not create new tensor
            # set data_ptr for buffer_tensor and output_tensor of current node
            for tensor in input_tensor:
                tensor: torch.Tensor
                target_buffer_tensor = next((x for x in buffer_tensor if not x.data_ptr() and x.shape == tensor.shape),
                                            None)
                target_output_tensor = next((x for x in output_tensor if not x.data_ptr() and x.shape == tensor.shape),
                                            None)
                target_buffer_tensor.data_ptr = tensor.data_ptr
                target_output_tensor.data_ptr = tensor.data_ptr
            # attach them to graph_info
            graph_info.fwd_tmp = buffer_tensor
            graph_info.fwd_out = output_tensor

        else:
            # set data_ptr for buffer_tensor
            for tensor in buffer_tensor:
                self._set_data_ptr(tensor)
            # attach it to graph_info
            graph_info.fwd_tmp = buffer_tensor

            # set data_ptr for output_tensor
            for tensor in output_tensor:
                self._set_data_ptr(tensor)
            # attach it to graph_info
            graph_info.fwd_out = output_tensor

        # fetch other memory informations
        memory_cost = meta_info.memory_cost
        graph_info.fwd_mem_tmp = memory_cost.fwd.temp
        graph_info.bwd_mem_tmp = memory_cost.bwd.temp

        node.meta = {**asdict(graph_info)}
