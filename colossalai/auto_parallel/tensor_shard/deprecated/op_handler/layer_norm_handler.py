import operator
from functools import reduce

import torch

from colossalai.auto_parallel.tensor_shard.deprecated._utils import (
    enumerate_all_possible_1d_sharding,
    enumerate_all_possible_2d_sharding,
    generate_sharding_size,
    ignore_sharding_exception,
)
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector

from .operator_handler import OperatorHandler

__all__ = ['LayerNormHandler']


class LayerNormHandler(OperatorHandler):
    """
    A OperatorHandler which deals with the sharding strategies of normalization.

    Note: To keep the math consistency, LayerNorm do not allow shards on hidden dimension.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.bias = self.module_named_parameters['bias']
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, total_sharding_size):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be devided by TFLOPS, now it just shows the computation size.

        Argument:
            bs(int): Batch size of the input data.
            channel_in(int): The channel dimension of input data.

        Return:
            compute_cost(float): Computation cost per device with this specific strategy
        '''
        # TODO: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        # TODO: a constant coefficient need to be added.

        norm_kernel_size = self.weight.shape
        # in LayerNorm context, batch dimensions mean all the dimensions do not join the normalization.
        input_batch_shape = self.input_data.shape[:-len(norm_kernel_size)]
        input_batch_product = reduce(operator.mul, input_batch_shape, 1)
        norm_kernel_product = reduce(operator.mul, norm_kernel_size, 1)
        forward_compute_cost = input_batch_product * norm_kernel_product / total_sharding_size
        backward_activation_compute_cost = input_batch_product * norm_kernel_product / total_sharding_size
        # To compute gradient of on norm kernel element requires input_batch_product times computation, so
        # the total cost is input_batch_product * norm_kernel_product
        backward_weight_compute_cost = input_batch_product * norm_kernel_product / total_sharding_size
        backward_compute_cost = backward_activation_compute_cost + backward_weight_compute_cost
        compute_cost = forward_compute_cost + backward_compute_cost
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
        # this operation will not change the shape of input
        numel_input = numel_output
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

    def _generate_strategy_with_dim_partition(self, dim_partition):
        dim_partition_dict_for_input = dim_partition
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = dim_partition
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_input.sharding_sequence} x {sharding_spec_for_weight.sharding_sequence}'
        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        total_sharding_size = generate_sharding_size(dim_partition, self.device_mesh)
        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # compute the memory cost of this strategy
        sharding_size_forward = generate_sharding_size(dim_partition_dict_for_input, self.device_mesh)
        sharding_size_backward_activation = generate_sharding_size(dim_partition_dict_for_output, self.device_mesh)
        sharding_size_weight = generate_sharding_size(dim_partition_dict_for_weight, self.device_mesh)
        memory_cost, _, _, memory_cost_backward_weight = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        total_mesh_dim_list = []
        for mesh_dim_list in dim_partition.values():
            total_mesh_dim_list.extend(mesh_dim_list)

        # This strategy do not need to do all_reduce operation for activation
        communication_cost_forward_activation = 0
        communication_cost_backward_activation = 0
        if len(total_mesh_dim_list) == 1:
            communication_cost_backward_weight = self.device_mesh.all_reduce_cost(memory_cost_backward_weight,
                                                                                  total_mesh_dim_list[0])
        else:
            assert len(total_mesh_dim_list) == 2, f'temporally we just support 2d device mesh.'
            communication_cost_backward_weight = self.device_mesh.flatten_device_mesh.all_reduce_cost(
                memory_cost_backward_weight, 0)
        communication_cost = communication_cost_forward_activation + communication_cost_backward_activation + communication_cost_backward_weight

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_batch_single_mesh_dim(self, mesh_dim_0):
        batch_dimension_length = self.input_data.dim() - self.weight.dim()
        dim_partition_list = enumerate_all_possible_1d_sharding(mesh_dim_0, batch_dimension_length)
        for dim_partition in dim_partition_list:
            self._generate_strategy_with_dim_partition(dim_partition)

    @ignore_sharding_exception
    def split_input_batch_both_mesh_dim(self, mesh_dim_0, mesh_dim_1):
        batch_dimension_length = self.input_data.dim() - self.weight.dim()
        dim_partition_list = enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, batch_dimension_length)
        for dim_partition in dim_partition_list:
            self._generate_strategy_with_dim_partition(dim_partition)

    @ignore_sharding_exception
    def non_split(self):
        name = f'RR = RR x R'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        total_sharding_size = 1
        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # compute the memory cost of this strategy
        sharding_size_forward = 1
        sharding_size_backward_activation = 1
        sharding_size_weight = 1
        memory_cost, _, _, _ = self._generate_memory_cost(sharding_size_forward, sharding_size_backward_activation,
                                                          sharding_size_weight)

        # This strategy do not need to do all_reduce operation
        communication_cost = 0
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
        Generate every possible strategies for a BatchNorm node, and record all strategies into the strategies_vector.

        Example:
            norm_handler = BatchNormHandler(node,  strategies_vector,
                                               self.shape_consistency_manager)
            norm_handler.register_strategy()
            for strategy in norm_handler.strategies_vector:
                print(f'{strategy.name}, computation_cost: {strategy.compute_cost}, memory_cost: {strategy.memory_cost}')

        Output:
            RS0 = RS0 x S0, computation_cost: 131072, memory_cost: 524288.0
            RS1 = RS1 x S1, computation_cost: 131072, memory_cost: 524288.0
            RR = RR x R, computation_cost: 262144, memory_cost: 1048576
            RS01 = RS01 x S01, computation_cost: 65536, memory_cost: 262144.0
        '''

        # SR = SR x R with single mesh dim on batch dimensions
        self.split_input_batch_single_mesh_dim(0)
        self.split_input_batch_single_mesh_dim(1)

        # SR = SR x R with both mesh dims on batch dimensions
        self.split_input_batch_both_mesh_dim(0, 1)

        # RR = RR x R
        self.non_split()

        return self.strategies_vector
