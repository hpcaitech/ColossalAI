import functools
import torch
from colossalai.tensor import ColoTensor
from typing import Callable, List
from colossalai.nn._ops._utils import convert_to_colo_tensor


def register_colo_graph(input_pos: List[int], param_pos: List[int]) -> Callable:
    """register_colo_graph 
    Register a Op (Layer) to ColoGraph.
    Recoders the input args in types of ColoTensor to the Graph.
    Args:
        func (Callable): a function implements the Op.

    Returns:
        Callable: wrapper function.
    """

    def register_colo_graph_decorator(func):
        from colossalai.nn.graph import GraphOpNode, GraphGlobalEnv

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            param_list = []
            input_list = []
            # TODO(jiaruifang) find the pg
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and idx in input_pos:
                    input_list.append(convert_to_colo_tensor(arg))
                if isinstance(arg, torch.Tensor) and idx in param_pos:
                    param_list.append(convert_to_colo_tensor(arg))
            # building the computing graph, inputs -> op
            if GraphGlobalEnv().graph_building:
                cur_op_node = GraphOpNode('linear', param_list)
                # TODO supports a list of ColoTensor as args
                if len(input_list) > 0:
                    cur_op_node.add_prev_tensor(input_list[0])

            outputs = func(*args, **kwargs)

            # building the computing graph, op -> output
            if GraphGlobalEnv().graph_building:
                # TODO supports a list of ColoTensor as args
                if isinstance(outputs[0], ColoTensor):
                    cur_op_node.add_post_tensor(outputs[0])
            return outputs

        return wrapper

    return register_colo_graph_decorator
