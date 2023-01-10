import builtins
import math
import operator
from copy import deepcopy
from typing import Dict, List

import torch
from torch.fx import Graph, Node

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from ._utils import generate_resharding_costs, generate_sharding_spec
from .constants import *
from .op_handler import *
from .options import SolverOptions
from .sharding_strategy import ShardingStrategy, StrategiesVector

__all__ = ['StrategiesConstructor']


class StrategiesConstructor:
    """
    StrategiesConstructor is used to construct the parallelization plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        device_mesh (DeviceMesh): a DeviceMesh object which contains the meta information about the cluster.
        solver_options (SolverOptions): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, device_mesh: DeviceMesh, solver_options: SolverOptions):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.device_mesh = device_mesh
        self.leaf_strategies = []
        self.strategy_map = {}
        self.solver_options = solver_options

    def remove_duplicated_strategy(self, strategies_vector):
        '''
        In build_strategies_and_cost method, we may produce some duplicated strategies.
        In this method, we will remove the duplicated strategies depending on the strategies name.
        '''
        name_checklist = []
        remove_list = []
        for strategy in strategies_vector:
            if strategy.name not in name_checklist:
                name_checklist.append(strategy.name)
            else:
                remove_list.append(strategy)

        for strategy in remove_list:
            strategies_vector.remove(strategy)

    def _is_bcast_matmul(self, node):
        is_bcast_matmul = False
        if node.target is torch.matmul and len(node.args) == 2:
            lhs_data = node.args[0]._meta_data
            rhs_data = node.args[1]._meta_data
            if lhs_data.dim() >= 3 and rhs_data.dim() >= 3:
                is_bcast_matmul = True
        return is_bcast_matmul

    def build_strategies_and_cost(self):
        for node in self.nodes:
            strategies_vector = StrategiesVector(node)
            input_nodes_len = 0
            for check_node in strategies_vector.predecessor_nodes:
                if isinstance(check_node._meta_data, torch.Tensor):
                    input_nodes_len += 1
            # input_nodes_len = len(strategies_vector.predecessor_nodes)
            # placeholder node
            if node.op == 'placeholder':
                # For placeholder nodes, if solver_options.fast is True, we just let them in
                # fully replicate status, then strategies of following node will be treated equally due
                # to replicate status has no resharding cost to other status. At the same time, the searching
                # space is smaller than enumerating all the possible sharding spec for the placeholder node.
                # Otherwise, all the possible sharding spec for the placeholder node will be enumerated.

                if self.solver_options.fast:
                    # create sharding strategy for placeholder
                    name = 'Replica Placeholder'
                    dim_partition_dict = {}
                    output_sharding_spec = generate_sharding_spec(node, self.device_mesh, dim_partition_dict)
                    # TODO: use meta_info_prop to profile memory cost
                    memory_cost = 0
                    sharding_strategy_placeholder = ShardingStrategy(name,
                                                                     output_sharding_spec,
                                                                     memory_cost=memory_cost)
                    strategies_vector.append(sharding_strategy_placeholder)

            # get_attr node
            if node.op == 'get_attr':
                # Same as placeholder nodes, if solver_options.fast is True, we just let them in
                # fully replicate status, then strategies of following node will be treated equally due
                # to replicate status has no resharding cost to other status. At the same time, the searching
                # space is smaller than enumerating all the possible sharding spec for the get_attr node.
                # Otherwise, all the possible sharding spec for the get_attr node will be enumerated.
                if self.solver_options.fast:
                    # create sharding strategy for get_attr
                    name = 'Replica Attribute'
                    dim_partition_dict = {}
                    output_sharding_spec = generate_sharding_spec(node, self.device_mesh, dim_partition_dict)
                    # TODO: use meta_info_prop to profile memory cost
                    memory_cost = 0
                    sharding_strategy_attribute = ShardingStrategy(name, output_sharding_spec, memory_cost=memory_cost)
                    strategies_vector.append(sharding_strategy_attribute)

            # call_module node
            if node.op == 'call_module':

                target = node.target
                submod = self.root_module.get_submodule(target)
                submod_type = type(submod)

                # conv module
                if submod_type in CONV_MODULE_OP:
                    # use ConvHandler to create sharding strategies for conv module node
                    conv_handler = ConvHandler(node, self.device_mesh, strategies_vector)
                    conv_handler.register_strategy()

                # linear module
                elif submod_type in LINEAR_MODULE_OP:
                    # use DotHandler to create sharding strategies for linear module node
                    dot_handler = DotHandler(node, self.device_mesh, strategies_vector)
                    dot_handler.register_strategy()

                # element-wise module
                elif submod_type in ELEMENTWISE_MODULE_OP:
                    unary_elementwise_handler = UnaryElementwiseHandler(node, self.device_mesh, strategies_vector)
                    unary_elementwise_handler.register_strategy()

                # BatchNormNd module
                elif submod_type in BATCHNORM_MODULE_OP:
                    # create sharding strategy for element-wise module
                    norm_handler = BatchNormHandler(node, self.device_mesh, strategies_vector)
                    norm_handler.register_strategy()
                    # for strategy in norm_handler.strategies_vector:
                    #     print(f'{strategy.name}, computation_cost: {strategy.compute_cost}, memory_cost: {strategy.memory_cost}')
                    # assert False

                # MaxPool module
                elif submod_type in POOL_MODULE_OP:
                    # TODO: add sharding constraints on image dimension
                    # e.g.: for a 2D pooling input NCHW, we should promise no sharding happens on H and W dimension

                    # create sharding strategy for element-wise module
                    assert input_nodes_len == 1, f'Temporally, we just support single input element-wise op.'
                    input_node = strategies_vector.predecessor_nodes[0]
                    # For element-wise module, we keep the sharding spec of output node same as
                    # the input. Therefore, the different strategies of input node with same
                    # output sharding spec will generate same strategy for element-wise module.
                    sharding_spec_checklist = []
                    for strategy in input_node.strategies_vector:
                        # It looks a little bit confusing, the input of the processing node
                        # is the output of the input_node.
                        input_sharding_spec = strategy.output_sharding_spec
                        assert isinstance(input_sharding_spec,
                                          ShardingSpec), f'The input node should NOT be a tuple of tensor.'
                        if input_sharding_spec in sharding_spec_checklist:
                            continue

                        sharding_spec_checklist.append(input_sharding_spec)
                        dim_partition_dict = deepcopy(input_sharding_spec.dim_partition_dict)
                        output_sharding_spec = generate_sharding_spec(node, self.device_mesh, dim_partition_dict)

                        name = f'{input_sharding_spec.sharding_sequence} -> {output_sharding_spec.sharding_sequence}'

                        # TODO: use meta_info_prop to profile memory cost and compute cost
                        compute_cost = node._meta_data.numel()
                        memory_cost = 0
                        resharding_costs = generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                     [input_sharding_spec])

                        sharding_strategy = ShardingStrategy(name,
                                                             output_sharding_spec,
                                                             compute_cost=compute_cost,
                                                             memory_cost=memory_cost,
                                                             resharding_costs=resharding_costs,
                                                             input_shardings=[input_sharding_spec])
                        strategies_vector.append(sharding_strategy)

                # embedding module
                elif submod_type in EMBEDDING_MODULE_OP:
                    embedding_handler = EmbeddingHandler(node, self.device_mesh, strategies_vector)
                    embedding_handler.register_strategy()

                # layernorm module
                elif submod_type in LAYERNORM_MODULE_OP:
                    layernorm_handler = LayerNormHandler(node, self.device_mesh, strategies_vector)
                    layernorm_handler.register_strategy()
                # other module
                else:
                    raise RuntimeError(f'{submod_type} module is NOT supported now.')

            # call_function node
            if node.op == 'call_function':
                target = node.target
                # conv function
                if target in CONV_FUNC_OP:
                    # use ConvHandler to create sharding strategies for conv node
                    # TODO: the operator_handler does NOT support function node processing now.
                    conv_handler = ConvHandler(node, self.device_mesh, strategies_vector)
                    conv_handler.register_strategy()

                # linear function
                elif target in LINEAR_FUNC_OP and not self._is_bcast_matmul(node):
                    # use DotHandler to create sharding strategies for linear node
                    # TODO: the operator_handler does NOT support function node processing now.
                    linear_handler = DotHandler(node, self.device_mesh, strategies_vector)
                    linear_handler.register_strategy()

                # where function
                elif target == torch.where:
                    if input_nodes_len == 1:
                        # both of x and y are scalar
                        pass

                    elif input_nodes_len == 2:
                        # one of x or y is type of scalar
                        pass

                    else:
                        # general case
                        where_handler = WhereHandler(node, self.device_mesh, strategies_vector)
                        where_handler.register_strategy()

                # reshape function
                elif target in RESHAPE_FUNC_OP:
                    # use ReshapeHandler to create sharding strategies for rehsape node
                    reshape_handler = ReshapeHandler(node, self.device_mesh, strategies_vector)
                    reshape_handler.register_strategy()

                # element-wise function
                elif target in ELEMENTWISE_FUNC_OP or (target in BCAST_FUNC_OP and input_nodes_len == 1):
                    unary_elementwise_handler = UnaryElementwiseHandler(node, self.device_mesh, strategies_vector)
                    unary_elementwise_handler.register_strategy()

                # bcast op
                elif target in BCAST_FUNC_OP:
                    if isinstance(node._meta_data, torch.Tensor):
                        bcast_op_handler = BcastOpHandler(node, self.device_mesh, strategies_vector)
                        bcast_op_handler.register_strategy()

                # torch.var_mean
                elif target == torch.var_mean:
                    dim = node.kwargs['dim']
                    input_tensor_node = strategies_vector.predecessor_nodes[0]
                    for strategy in input_tensor_node.strategies_vector:
                        input_sharding_spec = strategy.output_sharding_spec
                        assert isinstance(input_sharding_spec,
                                          ShardingSpec), f'The input node should NOT be a tuple of tensor.'
                        entire_shape_input = input_sharding_spec.entire_shape
                        dim_partition_dict_input = input_sharding_spec.dim_partition_dict
                        name = f'{new_input_sharding_spec.sharding_sequence} -> ({output_sharding_spec.sharding_sequence}, {output_sharding_spec.sharding_sequence})'
                        if dim in dim_partition_dict_input:
                            # We need to make the action dimension in replicate status
                            dim_partition_dict_for_input = deepcopy(dim_partition_dict_input)
                            dim_partition_dict_for_input.pop(dim)
                            new_input_sharding_spec = ShardingSpec(self.device_mesh,
                                                                   entire_shape_input,
                                                                   dim_partition_dict=dim_partition_dict_for_input)
                            entire_shape_output = deepcopy(entire_shape_input)
                            entire_shape_output.pop(dim)
                            dim_partition_dict_for_output = deepcopy(dim_partition_dict_for_input)
                            output_sharding_spec = ShardingSpec(self.device_mesh,
                                                                entire_shape_output,
                                                                dim_partition_dict=dim_partition_dict_for_input)
                            # TODO: use meta_info_prop to profile origin memory cost and compute cost, then divide them depending on sharding spec.
                            compute_cost = 0
                            memory_cost = 0
                            resharding_costs = generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                         [new_input_sharding_spec])
                            sharding_strategy = ShardingStrategy(name, (output_sharding_spec, output_sharding_spec),
                                                                 compute_cost=compute_cost,
                                                                 memory_cost=memory_cost,
                                                                 resharding_costs=resharding_costs,
                                                                 input_shardings=[new_input_sharding_spec])

                        else:
                            entire_shape_output = deepcopy(entire_shape_input)
                            entire_shape_output.pop(dim)
                            dim_partition_dict_for_output = deepcopy(dim_partition_dict_input)
                            output_sharding_spec = ShardingSpec(self.device_mesh,
                                                                entire_shape_output,
                                                                dim_partion_dict=dim_partition_dict_input)
                            # TODO: use meta_info_prop to profile origin memory cost and compute cost, then divide them depending on sharding spec.
                            compute_cost = 0
                            memory_cost = 0
                            resharding_costs = generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                         [input_sharding_spec])
                            sharding_strategy = ShardingStrategy(name, (output_sharding_spec, output_sharding_spec),
                                                                 compute_cost=compute_cost,
                                                                 memory_cost=memory_cost,
                                                                 resharding_costs=resharding_costs,
                                                                 input_shardings=[input_sharding_spec])

                        strategies_vector.append(sharding_strategy)

                # operator.getitem
                elif target == operator.getitem:
                    index = node.args[1]
                    input_tensor_node = strategies_vector.predecessor_nodes[0]
                    for strategy in input_tensor_node.strategies_vector:
                        if isinstance(strategy.output_sharding_spec, ShardingSpec):
                            input_sharding_spec = strategy.output_sharding_spec
                        else:
                            input_sharding_spec = strategy.output_sharding_spec[index]
                        assert isinstance(input_sharding_spec, ShardingSpec), f'This assertion is used to debug.'
                        dim_partition_dict_for_output = deepcopy(input_sharding_spec.dim_partition_dict)
                        entire_shape_output = deepcopy(input_sharding_spec.entire_shape)
                        output_sharding_spec = ShardingSpec(self.device_mesh,
                                                            entire_shape_output,
                                                            dim_partition_dict=dim_partition_dict_for_output)
                        # TODO: use meta_info_prop to profile origin memory cost and compute cost, then divide them depending on sharding spec.
                        compute_cost = 0
                        memory_cost = 0
                        resharding_costs = generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                     [input_sharding_spec],
                                                                     index=index)
                        # to prevent the resharding happening, set their resharding cost to inf.
                        resharding_costs[input_tensor_node] = [
                            cost if cost == 0 else INFINITY_COST for cost in resharding_costs[input_tensor_node]
                        ]
                        sharding_strategy = ShardingStrategy(name,
                                                             output_sharding_spec,
                                                             compute_cost=compute_cost,
                                                             memory_cost=memory_cost,
                                                             resharding_costs=resharding_costs,
                                                             input_shardings=[strategy.output_sharding_spec])
                        strategies_vector.append(sharding_strategy)

                # torch.arange function
                elif target == torch.arange:
                    name = f'FULLY REPLICATED ARANGE'
                    entire_shape_output = node._meta_data.shape
                    dim_partition_dict_for_output = {}
                    output_sharding_spec = ShardingSpec(self.device_mesh,
                                                        entire_shape_output,
                                                        dim_partition_dict=dim_partition_dict_for_output)
                    memory_cost = node._meta_data.numel()
                    sharding_strategy = ShardingStrategy(name,
                                                         output_sharding_spec,
                                                         compute_cost=0,
                                                         memory_cost=memory_cost)
                    strategies_vector.append(sharding_strategy)

                # op list to be processed to support gpt2
                elif target in (builtins.getattr, operator.le, torch.addmm):
                    pass
                # other function
                else:
                    raise RuntimeError(f'{target} function is NOT supported now.')

            # call_method node
            if node.op == 'call_method':
                method = getattr(node.args[0]._meta_data.__class__, node.target)
                if method in (torch.Tensor.size,):
                    pass
                elif method in ELEMENTWISE_METHOD_OP:
                    unary_elementwise_handler = UnaryElementwiseHandler(node, self.device_mesh, strategies_vector)
                    unary_elementwise_handler.register_strategy()

                elif method in RESHAPE_METHOD_OP:
                    reshape_handler = ReshapeHandler(node, self.device_mesh, strategies_vector)
                    reshape_handler.register_strategy()
                    # print(strategies_vector)
                    # if len(strategies_vector) == 0:
                    #     print(node)
                    #     assert False
                else:
                    raise RuntimeError(f'{method} function is NOT supported now.')

            # output node
            if node.op == 'output':
                if self.solver_options.fast:
                    # create sharding strategy for output
                    name = 'Replica Output'
                    input_nodes = strategies_vector.predecessor_nodes
                    input_sharding_specs = []
                    for input_node in input_nodes:
                        dim_partition_dict_for_input = {}
                        entire_shape = input_node._meta_data.shape
                        sharding_spec = ShardingSpec(self.device_mesh,
                                                     entire_shape,
                                                     dim_partition_dict=dim_partition_dict_for_input)
                        input_sharding_specs.append(sharding_spec)

                    dim_partition_dict = {}
                    output_sharding_spec = input_sharding_specs
                    # TODO: use meta_info_prop to profile memory cost
                    memory_cost = 0
                    resharding_costs = generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                 input_sharding_specs)

                    # clear the resharding cost for the output node
                    # TODO: we may remove this in final version
                    for prev_node, resharding_cost_list in resharding_costs.items():
                        resharding_costs[prev_node] = [0] * len(resharding_cost_list)

                    sharding_strategy_attribute = ShardingStrategy(name,
                                                                   output_sharding_spec,
                                                                   memory_cost=memory_cost,
                                                                   resharding_costs=resharding_costs,
                                                                   input_shardings=tuple(input_sharding_specs))
                    strategies_vector.append(sharding_strategy_attribute)

            self.remove_duplicated_strategy(strategies_vector)
            setattr(node, 'strategies_vector', strategies_vector)
            self.leaf_strategies.append(strategies_vector)
            self.strategy_map[node] = strategies_vector

        # remove no strategy nodes
        remove_list = []
        for strategies_vector in self.leaf_strategies:
            if len(strategies_vector) == 0:
                remove_list.append(strategies_vector.node)
        for node in remove_list:
            if node.strategies_vector in self.leaf_strategies:
                self.leaf_strategies.remove(node.strategies_vector)
            if node in self.strategy_map:
                self.strategy_map.pop(node)
