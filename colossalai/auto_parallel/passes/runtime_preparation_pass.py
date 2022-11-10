from copy import deepcopy
from typing import List

import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationDataType,
    ShardingStrategy,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.comm_spec import _all_reduce
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

shape_consistency_manager = ShapeConsistencyManager()


def _solution_annotatation(gm: torch.fx.GraphModule, solution: List[int]):
    """
    This method is used to stick the solution strategy to the nodes and add the information
    required in runtime into graph as placeholder nodes.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    # the dict to get origin sharding spec of node
    origin_node_sharding_spec_dict = {}
    for node_index, (node, strategy_index) in enumerate(zip(nodes, solution)):
        strategies_vector = node.strategies_vector
        # stick the solution strategy to the corresponding node
        setattr(node, 'best_strategy', strategies_vector[strategy_index])
        setattr(node, 'sharding_spec', strategies_vector[strategy_index].get_sharding_spec_by_name(str(node)))
        origin_node_sharding_spec_dict[node_index] = strategies_vector[strategy_index].get_sharding_spec_by_name(
            str(node))

    # the dict to get input sharding specs of user node
    sharding_spec_convert_dict = {}
    # the dict to record comm actions of nodes
    comm_actions_dict = {}
    for index, node in enumerate(nodes):
        target_sharding_specs = []
        for user_node in node.strategies_vector.successor_nodes:
            target_sharding_spec = user_node.best_strategy.get_sharding_spec_by_name(str(node.name))
            target_sharding_specs.append(target_sharding_spec)
        sharding_spec_convert_dict[index] = target_sharding_specs
        # the get_attr node strategy is kind of pending strategy, which means we will change it
        # to the same strategy of the user node.
        if node.op == 'get_attr':
            assert len(target_sharding_specs) == 1, f'sharing weight is not supported in current version.'
            new_sharding_spec = target_sharding_specs[0]
            user_strategy = node.strategies_vector.successor_nodes[0].best_strategy
            op_data_in_user = user_strategy.get_op_data_by_name(str(node))
            origin_node_sharding_spec_dict[index] = new_sharding_spec
            origin_pending_strategy = node.best_strategy
            origin_op_data = origin_pending_strategy.get_op_data_by_name(str(node))
            new_sharding_specs = origin_pending_strategy.sharding_specs
            new_sharding_specs[origin_op_data] = new_sharding_spec
            new_communication_actions = {}
            if op_data_in_user in user_strategy.communication_actions:
                new_communication_action = user_strategy.communication_actions.pop(op_data_in_user)
                new_communication_action.arg_index = 0
                new_communication_actions[origin_op_data] = new_communication_action
            new_strategy = ShardingStrategy(name=str(new_sharding_spec.sharding_sequence),
                                            sharding_specs=new_sharding_specs,
                                            compute_cost=origin_pending_strategy.compute_cost,
                                            communication_cost=origin_pending_strategy.communication_cost,
                                            memory_cost=origin_pending_strategy.memory_cost,
                                            communication_actions=new_communication_actions)
            setattr(node, 'best_strategy', new_strategy)
            setattr(node, 'sharding_spec', new_sharding_spec)
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
    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict


def _module_params_sharding(gm: torch.fx.GraphModule, device_mesh):
    """
    Apply the sharding action to the module parameters and buffers following the
    instructions of solver solution.
    """
    mod_graph = gm.graph
    nodes = tuple(mod_graph.nodes)

    for node in nodes:
        if node.op == 'call_module':
            target_module = node.graph.owning_module.get_submodule(node.target)

            for name, param in target_module.named_parameters():
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                # apply the sharding spec of parameters
                if target_sharding_spec.dim_partition_dict != {}:
                    origin_sharding_spec = ShardingSpec(device_mesh, param.shape, {})
                    setattr(param, 'sharding_spec', origin_sharding_spec)
                    # TODO: build a ColoParamter class to manager the distributed parameters
                    param_sharded = torch.nn.Parameter(
                        shape_consistency_manager.apply_for_autoparallel_runtime(param.data, param.sharding_spec,
                                                                                 target_sharding_spec).detach().clone())
                else:
                    param_sharded = param
                setattr(target_module, name, param_sharded)
                comm_actions = node.best_strategy.communication_actions
                for operation_data, comm_action in comm_actions.items():
                    comm_spec_to_use = comm_action.comm_spec
                    # register hook to the parameters
                    if operation_data.type == OperationDataType.PARAM and operation_data.name == name and comm_action.comm_type == CommType.HOOK:

                        def wrapper(param, comm_spec):

                            def hook_fn(grad):
                                _all_reduce(grad, comm_spec)

                            param.register_hook(hook_fn)

                        wrapper(param_sharded, comm_spec_to_use)

            sharded_buffer_dict = {}
            # apply the sharding spec of buffers
            for name, buffer in target_module.named_buffers():
                origin_sharding_spec = ShardingSpec(device_mesh, buffer.shape, {})
                setattr(buffer, 'sharding_spec', origin_sharding_spec)
                target_sharding_spec = node.best_strategy.get_sharding_spec_by_name(name)
                buffer_sharded = shape_consistency_manager.apply(buffer, target_sharding_spec)
                sharded_buffer_dict[name] = buffer_sharded

            for name, buffer_sharded in sharded_buffer_dict.items():
                setattr(target_module, name, buffer_sharded.detach().clone())

        if node.op == 'get_attr':
            root = node.graph.owning_module
            atoms = node.target.split(".")
            attr_len = len(atoms)
            if attr_len == 1:
                target_module = root
                target = getattr(root, atoms[0])
            else:
                target_module = root.get_submodule(atoms[-2])
                target = getattr(target_module, atoms[-1])

            target_sharding_spec = node.sharding_spec
            if target_sharding_spec.dim_partition_dict != {}:
                origin_sharding_spec = ShardingSpec(device_mesh, target.shape, {})
                setattr(target, 'sharding_spec', origin_sharding_spec)
                # TODO: build a ColoParamter class to manager the distributed parameters
                target_sharded = torch.nn.Parameter(
                    shape_consistency_manager.apply_for_autoparallel_runtime(target.data, target.sharding_spec,
                                                                             target_sharding_spec).detach().clone())
            else:
                target_sharded = target
            setattr(target_module, atoms[-1], target_sharded)

            comm_actions = node.best_strategy.communication_actions
            for operation_data, comm_action in comm_actions.items():
                comm_spec_to_use = comm_action.comm_spec
                # register hook to the parameters
                if isinstance(node._meta_data, torch.nn.parameter.Parameter) and comm_action.comm_type == CommType.HOOK:

                    def wrapper(param, comm_spec):

                        def hook_fn(grad):
                            _all_reduce(grad, comm_spec)

                        param.register_hook(hook_fn)

                    wrapper(target_sharded, comm_spec_to_use)
    return gm


def implicit_comm_action_apply(gm: torch.fx.GraphModule):
    """
    replace the origin kernel into kernel with implicit communication inside.
    """
    pass


def runtime_preparation_pass(gm: torch.fx.GraphModule, solution: List[int], device_mesh: DeviceMesh):
    gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict = _solution_annotatation(
        gm, solution)
    # TODO: the pass below should be uncommented after the implementation of implicit_comm_action_apply_pass completed.
    # gm = implicit_comm_action_apply(gm)
    gm = _module_params_sharding(gm, device_mesh)

    return gm, sharding_spec_convert_dict, origin_node_sharding_spec_dict, comm_actions_dict
