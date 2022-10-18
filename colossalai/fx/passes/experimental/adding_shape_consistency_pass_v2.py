from ast import NodeTransformer
import torch
from typing import List
from torch.fx import symbolic_trace
from torch.fx.node import Node
from colossalai.fx.passes.split_module import split_module
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
import builtins
import operator
from copy import deepcopy

shape_consistency_manager = ShapeConsistencyManager()


class ConsistencyApply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, node, origin_dict, input_dict, node_index, user_node_index):
        ctx.origin_sharding_spec = origin_dict[node_index]
        ctx.target_sharding_spec = input_dict[node_index][user_node_index]
        return shape_consistency_manager.apply_for_autoparallel_runtime(node, ctx.origin_sharding_spec,
                                                                        ctx.target_sharding_spec)

    @staticmethod
    def backward(ctx, node_grad):
        return shape_consistency_manager.apply_for_autoparallel_runtime(
            node_grad, ctx.target_sharding_spec, ctx.origin_sharding_spec), None, None, None, None


def runtime_apply_for_leaf_node(node, origin_dict, input_dict, node_index, user_node_index):
    return ConsistencyApply.apply(node, origin_dict, input_dict, node_index, user_node_index)


def runtime_apply(node, origin_dict, input_dict, node_index, user_node_index):
    origin_sharding_spec = origin_dict[node_index]
    target_sharding_spec = input_dict[node_index][user_node_index]
    return shape_consistency_manager.apply_for_autoparallel_runtime(node, origin_sharding_spec, target_sharding_spec)


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
                origin_sharding_spec = ShardingSpec(device_mesh, param.shape, {})
                setattr(param, 'sharding_spec', origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                shape_consistency_manager.apply(param, target_sharding_spec)

            for name, buffer in target_module.named_buffers():
                origin_sharding_spec = ShardingSpec(device_mesh, buffer.shape, {})
                setattr(buffer, 'sharding_spec', origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                shape_consistency_manager.apply(buffer, target_sharding_spec)

    # the dict to get input sharding specs of user node
    sharding_spec_convert_dict = {}
    for index, node in enumerate(nodes):
        target_sharding_specs = []
        for user_node in node.strategies_vector.successor_nodes:
            target_sharding_spec = user_node.best_strategy.get_sharding_spec_by_name(str(node.name))
            target_sharding_specs.append(target_sharding_spec)
        sharding_spec_convert_dict[index] = target_sharding_specs

    # add above dicts into graph
    for node in nodes:
        if node.op != 'placeholder':
            with mod_graph.inserting_before(node):
                input_specs_node = mod_graph.create_node('placeholder', target='sharding_spec_convert_dict')
                origin_specs_node = mod_graph.create_node('placeholder', target='origin_node_sharding_spec_dict')
            break

    return sharding_spec_convert_dict, origin_node_sharding_spec_dict


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
            if user_node.op != "output":
                with mod_graph.inserting_before(user_node):
                    shape_consistency_node = mod_graph.create_node('call_function',
                                                                   runtime_apply,
                                                                   args=(node, origin_dict_node, input_dict_node,
                                                                         node_to_index_dict[node], user_node_index))
            else:
                # we need to call an autograd.Function for leaf node
                with mod_graph.inserting_before(user_node):
                    shape_consistency_node = mod_graph.create_node('call_function',
                                                                   runtime_apply_for_leaf_node,
                                                                   args=(node, origin_dict_node, input_dict_node,
                                                                         node_to_index_dict[node], user_node_index))

            origin_index_args = user_node.args.index(node)
            new_args = list(user_node.args)
            new_args[origin_index_args] = shape_consistency_node
            user_node.args = new_args

    return gm
