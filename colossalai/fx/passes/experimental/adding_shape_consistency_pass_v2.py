import builtins
import copy
import operator
from ast import NodeTransformer
from copy import deepcopy
from typing import List

import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import CommAction, CommType, OperationDataType
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.passes.split_module import split_module
from colossalai.tensor.comm_spec import CollectiveCommPattern, CommSpec, _all_reduce, pattern_to_func_dict
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec

shape_consistency_manager = ShapeConsistencyManager()


def runtime_apply(node, origin_dict, input_dict, node_index, user_node_index):
    origin_sharding_spec = origin_dict[node_index]
    target_sharding_spec = input_dict[node_index][user_node_index]
    return shape_consistency_manager.apply_for_autoparallel_runtime(node, origin_sharding_spec, target_sharding_spec)


def runtime_comm_spec_apply(tensor, comm_actions_dict, node_index, op_data):

    comm_action = comm_actions_dict[node_index][op_data]
    if isinstance(comm_action.comm_spec, CommSpec):
        rst = comm_action.comm_spec.covert_spec_to_action(tensor)
    else:
        origin_sharding_spec = comm_action.comm_spec['src_spec']
        tgt_sharding_spec = comm_action.comm_spec['tgt_spec']
        rst = shape_consistency_manager.apply_for_autoparallel_runtime(tensor, origin_sharding_spec, tgt_sharding_spec)
    return rst


def solution_annotatation_pass(gm: torch.fx.GraphModule, solution: List[int], device_mesh):
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    # the dict to get origin sharding spec of node
    origin_node_sharding_spec_dict = {}
    for node_index, (node, strategy_index) in enumerate(zip(nodes, solution)):
        strategies_vector = node.strategies_vector
        setattr(node, 'best_strategy', strategies_vector[strategy_index])
        setattr(node, 'sharding_spec', strategies_vector[strategy_index].get_sharding_spec_by_name(str(node)))
        origin_node_sharding_spec_dict[node_index] = strategies_vector[strategy_index].get_sharding_spec_by_name(
            str(node))

    # apply the sharding spec of parameters
    for node in nodes:
        if node.op == 'call_module':
            target_module = node.graph.owning_module.get_submodule(node.target)
            for name, param in target_module.named_parameters():
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                if target_sharding_spec.dim_partition_dict != {}:
                    origin_sharding_spec = ShardingSpec(device_mesh, param.shape, {})
                    setattr(param, 'sharding_spec', origin_sharding_spec)
                    param_sharded = torch.nn.Parameter(
                        shape_consistency_manager.apply_for_autoparallel_runtime(param.data, param.sharding_spec,
                                                                                 target_sharding_spec).detach().clone())
                else:
                    param_sharded = param
                setattr(target_module, name, param_sharded)
                comm_actions = node.best_strategy.communication_actions
                for operation_data, comm_action in comm_actions.items():
                    comm_spec_to_use = comm_action.comm_spec
                    if operation_data.type == OperationDataType.PARAM and operation_data.name == name and comm_action.comm_type == CommType.HOOK:

                        def hook_fn(grad):
                            _all_reduce(grad, comm_spec_to_use)

                        param_sharded.register_hook(hook_fn)

            sharded_buffer_dict = {}
            for name, buffer in target_module.named_buffers():
                origin_sharding_spec = ShardingSpec(device_mesh, buffer.shape, {})
                setattr(buffer, 'sharding_spec', origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                buffer_sharded = shape_consistency_manager.apply(buffer, target_sharding_spec)
                sharded_buffer_dict[name] = buffer_sharded

            for name, buffer_sharded in sharded_buffer_dict.items():
                setattr(target_module, name, buffer_sharded.detach().clone())

    # the dict to get input sharding specs of user node
    sharding_spec_convert_dict = {}
    for index, node in enumerate(nodes):
        target_sharding_specs = []
        for user_node in node.strategies_vector.successor_nodes:
            target_sharding_spec = user_node.best_strategy.get_sharding_spec_by_name(str(node.name))
            target_sharding_specs.append(target_sharding_spec)
        sharding_spec_convert_dict[index] = target_sharding_specs

    # the dict to record comm actions of nodes
    comm_actions_dict = {}
    for index, node in enumerate(nodes):
        comm_action_dict = {}
        for op_data, comm_action in node.best_strategy.communication_actions.items():
            comm_action_dict[op_data.name] = comm_action
        comm_actions_dict[index] = comm_action_dict

    # add above dicts into graph
    for node in nodes:
        if node.op != 'placeholder':
            with mod_graph.inserting_before(node):
                input_specs_node = mod_graph.create_node('placeholder', target='sharding_spec_convert_dict')
                origin_specs_node = mod_graph.create_node('placeholder', target='origin_node_sharding_spec_dict')
                comm_actions_dict_node = mod_graph.create_node('placeholder', target='comm_actions_dict')
            break

    return sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict


def shape_consistency_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)
    input_dict_node = None
    origin_dict_node = None

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
    assert input_dict_node is not None

    # add shape consistency apply function into graph
    for node in nodes:
        if not hasattr(node, 'best_strategy') or node.op == 'output':
            continue

        for user_node in node.strategies_vector.successor_nodes:
            user_node_index = user_node.strategies_vector.predecessor_nodes.index(node)
            with mod_graph.inserting_before(user_node):
                shape_consistency_node = mod_graph.create_node('call_function',
                                                               runtime_apply,
                                                               args=(node, origin_dict_node, input_dict_node,
                                                                     node_to_index_dict[node], user_node_index))

            origin_index_args = user_node.args.index(node)
            new_args = list(user_node.args)
            new_args[origin_index_args] = shape_consistency_node
            user_node.args = new_args

        comm_actions = node.best_strategy.communication_actions
        for op_data, comm_action in comm_actions.items():
            comm_object = node.args[comm_action.arg_index]
            if op_data.type == OperationDataType.PARAM:
                continue
            if comm_action.comm_type == CommType.BEFORE:
                with mod_graph.inserting_before(node):
                    comm_spec_apply_node = mod_graph.create_node('call_function',
                                                                 runtime_comm_spec_apply,
                                                                 args=(comm_object, comm_actions_dict_node,
                                                                       node_to_index_dict[node], op_data.name))
                new_args = list(node.args)
                new_args[comm_action.arg_index] = comm_spec_apply_node
                node.args = new_args
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
                    new_args[new_args.index(node)] = comm_spec_apply_node
                    user.args = tuple(new_args)
            # TODO: consider other OperationDataType, such as OperationDataType.OUTPUT
    return gm
