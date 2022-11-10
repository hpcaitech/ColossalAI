from copy import deepcopy
from typing import Dict, List

import torch
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationData,
    OperationDataType,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.comm_spec import CommSpec
from colossalai.tensor.shape_consistency import ShapeConsistencyManager

shape_consistency_manager = ShapeConsistencyManager()


def runtime_apply(node: Node, origin_dict: Dict, input_dict: Dict, node_index: int, user_node_index: int):
    """
    This method will be invoked during runtime to do the shape consistency, which make sure the activations is converted into
    the user node expected form.
    """
    origin_sharding_spec = origin_dict[node_index]
    target_sharding_spec = input_dict[node_index][user_node_index]
    return shape_consistency_manager.apply_for_autoparallel_runtime(node, origin_sharding_spec, target_sharding_spec)


def runtime_comm_spec_apply(tensor: torch.Tensor, comm_actions_dict: Dict, node_index: int, op_data_name: str):
    """
    This method will be invoked during runtime to apply the comm action following the instruction of comm spec.
    """
    comm_action = comm_actions_dict[node_index][op_data_name]
    if isinstance(comm_action.comm_spec, CommSpec):
        rst = comm_action.comm_spec.covert_spec_to_action(tensor)
    else:
        origin_sharding_spec = comm_action.comm_spec['src_spec']
        tgt_sharding_spec = comm_action.comm_spec['tgt_spec']
        rst = shape_consistency_manager.apply_for_autoparallel_runtime(tensor, origin_sharding_spec, tgt_sharding_spec)
    return rst


def _preprocess_graph(nodes: List[Node]):
    """
    This method is used to extract all the placeholders with sharding information,
    and mapping the nodes into the index of the origin graph.
    """
    # mapping the node into the origin graph index
    node_to_index_dict = {}
    index = 0
    for node in nodes:
        if node.target == 'sharding_spec_convert_dict':
            input_dict_node = node
            continue
        if node.target == 'origin_node_sharding_spec_dict':
            origin_dict_node = node
            continue
        if node.target == 'comm_actions_dict':
            comm_actions_dict_node = node
            continue
        if not hasattr(node, 'best_strategy'):
            continue
        node_to_index_dict[node] = index
        index += 1

    return input_dict_node, origin_dict_node, comm_actions_dict_node, node_to_index_dict


def _shape_consistency_apply(gm: torch.fx.GraphModule):
    """
    This pass is used to add the shape consistency node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    input_dict_node, origin_dict_node, _, node_to_index_dict = _preprocess_graph(nodes)

    for node in nodes:
        if not hasattr(node, 'best_strategy') or node.op == 'output':
            continue

        for user_node_index, user_node in enumerate(node.strategies_vector.successor_nodes):
            with mod_graph.inserting_before(user_node):
                shape_consistency_node = mod_graph.create_node('call_function',
                                                               runtime_apply,
                                                               args=(node, origin_dict_node, input_dict_node,
                                                                     node_to_index_dict[node], user_node_index))
            new_args = list(user_node.args)
            new_kwargs = dict(user_node.kwargs)
            # the origin node may be a positional argument or key word argument of user node
            if node in new_args:
                # substitute the origin node with shape_consistency_node
                origin_index_args = new_args.index(node)
                new_args[origin_index_args] = shape_consistency_node
                user_node.args = tuple(new_args)
            elif str(node) in new_kwargs:
                # substitute the origin node with shape_consistency_node
                new_kwargs[str(node)] = shape_consistency_node
                user_node.kwargs = new_kwargs

    return gm


def _comm_spec_apply(gm: torch.fx.GraphModule):
    """
    This pass is used to add the comm spec apply node to the origin graph.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    _, _, comm_actions_dict_node, node_to_index_dict = _preprocess_graph(nodes)

    for node in nodes:
        if not hasattr(node, 'best_strategy') or node.op == 'output':
            continue

        comm_actions = node.best_strategy.communication_actions
        for op_data, comm_action in comm_actions.items():

            if comm_action.comm_type == CommType.HOOK:
                continue
            if comm_action.comm_type == CommType.BEFORE:
                if op_data.type == OperationDataType.OUTPUT:
                    comm_object = node
                elif comm_action.key_for_kwarg is not None:
                    comm_object = node.kwargs[comm_action.key_for_kwarg]
                else:
                    comm_object = node.args[comm_action.arg_index]
                with mod_graph.inserting_before(node):
                    comm_spec_apply_node = mod_graph.create_node('call_function',
                                                                 runtime_comm_spec_apply,
                                                                 args=(comm_object, comm_actions_dict_node,
                                                                       node_to_index_dict[node], op_data.name))
                # the origin node may be a positional argument or key word argument of user node
                if comm_action.key_for_kwarg is not None:
                    # substitute the origin node with comm_spec_apply_node
                    new_kwargs = dict(node.kwargs)
                    new_kwargs[comm_action.key_for_kwarg] = comm_spec_apply_node
                    node.kwargs = new_kwargs
                else:
                    # substitute the origin node with comm_spec_apply_node
                    new_args = list(node.args)
                    new_args[comm_action.arg_index] = comm_spec_apply_node
                    node.args = tuple(new_args)

            elif comm_action.comm_type == CommType.AFTER:
                with mod_graph.inserting_after(node):
                    comm_spec_apply_node = mod_graph.create_node('call_function',
                                                                 runtime_comm_spec_apply,
                                                                 args=(node, comm_actions_dict_node,
                                                                       node_to_index_dict[node], op_data.name))
                user_list = list(node.users.keys())
                for user in user_list:
                    if user == comm_spec_apply_node:
                        continue
                    new_args = list(user.args)
                    new_kwargs = dict(user.kwargs)
                    # the origin node may be a positional argument or key word argument of user node
                    if node in new_args:
                        # substitute the origin node with comm_spec_apply_node
                        new_args[new_args.index(node)] = comm_spec_apply_node
                        user.args = tuple(new_args)
                    elif str(node) in new_kwargs:
                        # substitute the origin node with comm_spec_apply_node
                        new_kwargs[str(node)] = comm_spec_apply_node
                        user.kwargs = new_kwargs
    return gm


def runtime_apply_pass(gm: torch.fx.GraphModule):
    """
    The method manages all the passes acting on the distributed training runtime.
    """
    gm = _shape_consistency_apply(gm)
    gm = _comm_spec_apply(gm)

    return gm
