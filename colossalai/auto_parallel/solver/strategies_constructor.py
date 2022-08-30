from torch.fx import Graph, Node
from colossalai.tensor.sharding_spec import ShardingSpec
from .sharding_strategy import ShardingStrategy, StrategiesVector
from .conv_handler import ConvHandler
from .constants import *
from copy import deepcopy
import math
import torch
import operator
from typing import Dict, List


class StrategiesConstructor:

    def __init__(self, graph, device_mesh, shape_consistency_manager, solver_options):
        self.graph = graph
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.device_mesh = device_mesh
        self.leaf_strategies = []
        self.strategy_map = {}
        self.shape_consistency_manager = shape_consistency_manager
        self.solver_options = solver_options

    def _generate_sharding_spec(self, node: Node, dim_partition_dict: Dict[int, List[int]]) -> ShardingSpec:
        """
        Generate the sharding spec of the tensor based on the given dim_partition_dict 
        where the key is the tensor dimension and the value is the mesh dimension for sharding.
        """
        meta_tensor = node._meta_data
        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                     entire_shape=meta_tensor.shape,
                                     dim_partition_dict=dim_partition_dict)
        return sharding_spec

    def _generate_resharding_costs(self, input_nodes, target_sharding_specs):
        '''
        Compute the resharding costs with this specific strategy.

        Argument:
            sharding_spec_for_input(ShardingSpec): ShardingSpec of the input node.
        '''
        resharding_costs = {}
        for input_node, target_sharding_spec in zip(input_nodes, target_sharding_specs):
            resharding_costs[input_node] = []
            for strategy in input_node.strategies_vector:
                input_sharding_spec = strategy.output_sharding_spec
                assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'
                _, _, resharding_cost = self.shape_consistency_manager.shape_consistency(
                    input_sharding_spec, target_sharding_spec)
                resharding_costs[input_node].append(resharding_cost)
        return resharding_costs

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

    def build_strategies_and_cost(self):
        for node in self.nodes:
            strategies_vector = StrategiesVector(node)
            # placeholder node
            if node.op == 'placeholder':
                # For placeholder nodes, if solver_options['fast_mode'] is True, we just let them in
                # fully replicate status, then strategies of following node will be treated equally due
                # to replicate status has no resharding cost to other status. At the same time, the searching
                # space is smaller than enumerating all the possible sharding spec for the placeholder node.
                # Otherwise, all the possible sharding spec for the placeholder node will be enumerated.

                if self.solver_options['fast_mode']:
                    # create sharding strategy for placeholder
                    name = 'Replica Placeholder'
                    dim_partition_dict = {}
                    output_sharding_spec = self._generate_sharding_spec(node, dim_partition_dict)
                    # TODO: use meta_info_prop to profile memory cost
                    memory_cost = 0
                    sharding_strategy_placeholder = ShardingStrategy(name,
                                                                     output_sharding_spec,
                                                                     memory_cost=memory_cost)
                    strategies_vector.append(sharding_strategy_placeholder)

            # get_attr node
            if node.op == 'get_attr':
                # Same as placeholder nodes, if solver_options['fast_mode'] is True, we just let them in
                # fully replicate status, then strategies of following node will be treated equally due
                # to replicate status has no resharding cost to other status. At the same time, the searching
                # space is smaller than enumerating all the possible sharding spec for the get_attr node.
                # Otherwise, all the possible sharding spec for the get_attr node will be enumerated.
                if self.solver_options['fast_mode']:
                    # create sharding strategy for get_attr
                    name = 'Replica Attribute'
                    dim_partition_dict = {}
                    output_sharding_spec = self._generate_sharding_spec(node, dim_partition_dict)
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
                    conv_handler = ConvHandler(node, self.device_mesh, strategies_vector,
                                               self.shape_consistency_manager)
                    conv_handler.register_strategy()

                # linear module
                elif submod_type in LINEAR_MODULE_OP:
                    # use DotHandler to create sharding strategies for linear module node
                    dot_handler = DotHandler(node, self.device_mesh, strategies_vector, self.shape_consistency_manager)
                    dot_handler.register_strategy()

                # element-wise module
                elif submod_type in ELEMENTWISE_MODULE_OP:
                    # create sharding strategy for element-wise module
                    assert len(strategies_vector.predecessor_nodes
                              ) == 1, f'Temporally, we just support single input element-wise op.'
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
                        output_sharding_spec = self._generate_sharding_spec(node, dim_partition_dict)

                        name = f'{input_sharding_spec.sharding_sequence} -> {output_sharding_spec.sharding_sequence}'

                        # TODO: use meta_info_prop to profile memory cost and compute cost
                        compute_cost = node._meta_data.numel()
                        memory_cost = 0
                        resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                           [input_sharding_spec])

                        # to prevent the resharding happening, set their resharding cost to inf.
                        resharding_costs[input_node] = [
                            cost if cost == 0 else math.inf for cost in resharding_costs[input_node]
                        ]
                        sharding_strategy = ShardingStrategy(name,
                                                             output_sharding_spec,
                                                             compute_cost=compute_cost,
                                                             memory_cost=memory_cost,
                                                             resharding_costs=resharding_costs,
                                                             input_shardings=[input_sharding_spec])
                        strategies_vector.append(sharding_strategy)

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
                    conv_handler = ConvHandler(node, self.device_mesh, strategies_vector,
                                               self.shape_consistency_manager)
                    conv_handler.register_strategy()

                # linear function
                elif target in LINEAR_FUNC_OP:
                    # use DotHandler to create sharding strategies for linear node
                    # TODO: the operator_handler does NOT support function node processing now.
                    linear_handler = DotHandler(node, self.device_mesh, strategies_vector,
                                                self.shape_consistency_manager)
                    linear_handler.register_strategy()

                # element-wise function
                elif target in ELEMENTWISE_FUNC_OP:
                    # TODO: integrate element-wise func and module together
                    # create sharding strategy for element-wise function
                    assert len(strategies_vector.predecessor_nodes
                              ) == 1, f'Temporally, we just support single input element-wise op.'
                    input_node = strategies_vector.predecessor_nodes[0]
                    # For element-wise function, we keep the sharding spec of output node same as
                    # the input. Therefore, the different strategies of input node with same
                    # output sharding spec will generate same strategy for element-wise function.
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
                        output_sharding_spec = self._generate_sharding_spec(node, dim_partition_dict)
                        name = f'{input_sharding_spec.sharding_sequence} -> {output_sharding_spec.sharding_sequence}'
                        # TODO: use meta_info_prop to profile memory cost and compute cost
                        compute_cost = node._meta_data.numel()
                        memory_cost = 0

                        resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                           [input_sharding_spec])

                        # to prevent the resharding happening, set their resharding cost to inf.
                        resharding_costs[input_node] = [
                            0 if cost == 0 else math.inf for cost in resharding_costs[input_node]
                        ]
                        sharding_strategy = ShardingStrategy(name,
                                                             output_sharding_spec,
                                                             compute_cost=compute_cost,
                                                             memory_cost=memory_cost,
                                                             resharding_costs=resharding_costs,
                                                             input_shardings=[input_sharding_spec])
                        strategies_vector.append(sharding_strategy)

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
                            resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
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
                            resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
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
                        input_sharding_spec = input_tensor_node.output_sharding_spec[index]
                        assert isinstance(input_sharding_spec, ShardingSpec), f'This assertion is used to debug.'
                        dim_partition_dict_for_output = deepcopy(input_sharding_spec.dim_partition_dict)
                        entire_shape_output = deepcopy(input_sharding_spec.entire_shape)
                        output_sharding_spec = ShardingSpec(self.device_mesh,
                                                            entire_shape_output,
                                                            dim_partition_dict=dim_partition_dict_for_output)
                        # TODO: use meta_info_prop to profile origin memory cost and compute cost, then divide them depending on sharding spec.
                        compute_cost = 0
                        memory_cost = 0
                        resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                           [input_sharding_spec])
                        # to prevent the resharding happening, set their resharding cost to inf.
                        resharding_costs[input_tensor_node] = [
                            cost if cost == 0 else math.inf for cost in resharding_costs[input_tensor_node]
                        ]
                        sharding_strategy = ShardingStrategy(name,
                                                             output_sharding_spec,
                                                             compute_cost=compute_cost,
                                                             memory_cost=memory_cost,
                                                             resharding_costs=resharding_costs,
                                                             input_shardings=[input_tensor_node.output_sharding_spec])
                        strategies_vector.append(sharding_strategy)

                # other function
                else:
                    raise RuntimeError(f'{target} function is NOT supported now.')

            # output node
            if node.op == 'output':
                if self.solver_options['fast_mode']:
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
                    resharding_costs = self._generate_resharding_costs(strategies_vector.predecessor_nodes,
                                                                       input_sharding_specs)
                    sharding_strategy_attribute = ShardingStrategy(name,
                                                                   output_sharding_spec,
                                                                   memory_cost=memory_cost,
                                                                   resharding_costs=resharding_costs)
                    strategies_vector.append(sharding_strategy_attribute)

            self.remove_duplicated_strategy(strategies_vector)
            setattr(node, 'strategies_vector', strategies_vector)
            self.leaf_strategies.append(strategies_vector)
            self.strategy_map[node] = strategies_vector
