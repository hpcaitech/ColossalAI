import operator
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List

import torch
from colossalai.auto_parallel.tensor_shard.deprecated._utils import \
    ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import (ShardingStrategy, StrategiesVector)
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from .operator_handler import OperatorHandler

__all__ = ['EmbeddingHandler']


class EmbeddingHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of Embedding operators(such as nn.embedding).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, total_sharding_size):
        input_shape = self.input_data.shape
        weight_shape = self.weight.shape
        input_shape_product = reduce(operator.mul, input_shape, 1)
        weight_shape_product = reduce(operator.mul, weight_shape, 1)
        compute_cost = input_shape_product * weight_shape_product * 2 / total_sharding_size
        return compute_cost

    def _generate_memory_cost(self, sharding_size_forward, sharding_size_backward_activation, sharding_size_weight):
        '''
        Compute the memory cost per device with this specific strategy.

        Argument:
            sharding_size_forward(int): The forward activation will be divided
                into sharding_size_forward number partions.
            sharding_size_backward_activation(int): The backward activation will 
                be divided into sharding_size_backward_activation number partions.
            sharding_size_weight(int): The backward weight will be divided
                into sharding_size_weight number partions.

        Return:
            memory_cost(Tuple[float]): Memory cost per device with this 
                specific strategy, the first element of this tuple is forward
                memory cost, and the second element of this tuple is backward
                memory cost.
            memory_cost_forward(float): Memory cost of forward activation per 
                device with this specific strategy.
            memory_cost_backward_activation(float): Memory cost of backward activation 
                per device with this specific strategy.
        '''
        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel_output = self.output_data.numel()
        numel_input = self.input_data.numel()
        numel_weight = self.weight.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()

        # forward memory_cost
        memory_cost_forward_activation = numel_output * size_per_elem_bytes / sharding_size_forward
        memory_cost_forward_weight = numel_weight * size_per_elem_bytes / sharding_size_weight
        memory_cost_forward = memory_cost_forward_activation + memory_cost_forward_weight

        # backward memory_cost
        memory_cost_backward_activation = numel_input * size_per_elem_bytes / sharding_size_backward_activation
        memory_cost_backward_weight = numel_weight * size_per_elem_bytes / sharding_size_weight
        memory_cost_backward = memory_cost_backward_activation + memory_cost_backward_weight

        # memory_cost pair
        memory_cost = (memory_cost_forward, memory_cost_backward)

        return memory_cost, memory_cost_forward_activation, memory_cost_backward_activation, memory_cost_backward_weight

    @ignore_sharding_exception
    def split_weight_both_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'RRS{mesh_dim_1} = RR x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {2: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[0] * self.device_mesh.shape[1]
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = 1
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        memory_cost, memory_cost_forward_activation, memory_cost_backward_activation, _ = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # compute the communication cost of this strategy during forward phase
        communication_cost_forward = self.device_mesh.all_reduce_cost(memory_cost_forward_activation, mesh_dim_0)
        # compute the communication cost of this strategy during backward phase
        communication_cost_backward = self.device_mesh.all_reduce_cost(memory_cost_backward_activation, mesh_dim_1)
        communication_cost = communication_cost_forward + communication_cost_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_both_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}S{mesh_dim_1}R = S{mesh_dim_0}S{mesh_dim_1} x RR'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        total_sharding_size = self.device_mesh.shape[0] * self.device_mesh.shape[1]
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_weight = 1
        memory_cost, memory_cost_forward_activation, memory_cost_backward_activation, memory_cost_backward_weight = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # This strategy do not need to do all_reduce during forward phase
        communication_cost_forward = 0
        # compute the communication cost of this strategy during backward phase
        communication_cost_backward_activation = 0
        communication_cost_backward_weight = self.device_mesh.flatten_device_mesh.all_reduce_cost(
            memory_cost_backward_weight, 0)
        communication_cost_backward = communication_cost_backward_activation + communication_cost_backward_weight
        communication_cost = communication_cost_forward + communication_cost_backward
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        '''
        Generate every possible strategies for a Conv node, and record all strategies into the strategies_vector.
        '''
        # RRS = RR x SS
        self.split_weight_both_dim(0, 1)
        self.split_weight_both_dim(1, 0)

        # SSR = SS x RR
        self.split_input_both_dim(0, 1)
        self.split_input_both_dim(1, 0)

        return self.strategies_vector
