import operator
from functools import reduce

import torch
from colossalai.auto_parallel.tensor_shard.deprecated._utils import \
    ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import (ShardingStrategy, StrategiesVector)

from .operator_handler import OperatorHandler

__all__ = ['BatchNormHandler']


class BatchNormHandler(OperatorHandler):
    """
    A OperatorHandler which deals with the sharding strategies of normalization.

    To keep the math consistency, there are two way to do BatchNorm if the input
    shards on batch dimension:
    1. We gather the input partitions through batch dimension, then do the normal BatchNorm.
    2. We do the SyncBatchNorm on the each input partition seperately, the SyncBN op will help
       us to keep the computing correctness.
    In this handler, both methods will be considered.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.bias = self.module_named_parameters['bias']
        self.output_data = self.node._meta_data
        self._sanity_check()

    def _sanity_check(self):
        '''
        In sanity check, we need make sure the input data having correct dimension size.
        For BatchNorm1d, the dim of input data should be 3([N, C, L]).
        For BatchNorm2d, the dim of input data should be 4([N, C, H, W]).
        For BatchNorm3d, the dim of input data should be 5([N, C, H, W, D]).
        '''
        assert self.input_data.dim() in (3, 4,
                                         5), f'We suppose the dim of input fed into conv op should in range of [3, 5].'

    def _generate_compute_cost(self, bs, channel_in):
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
        # 1D: (L) * N * Cin
        # 2D: (H * W) * N  * Cin
        # 3D: (H * W  * D) * N  * Cin

        input_size = self.input_data.shape[2:]
        input_size_product = reduce(operator.mul, input_size, 1)
        forward_compute_cost = input_size_product * bs * channel_in
        backward_activation_compute_cost = input_size_product * bs * channel_in
        backward_weight_compute_cost = input_size_product * bs * channel_in
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

        return memory_cost, memory_cost_forward_activation, memory_cost_backward_activation

    @ignore_sharding_exception
    def split_input_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0} = RS{mesh_dim_0} x S{mesh_dim_0}'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_0]
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0]
        memory_cost, _, _ = self._generate_memory_cost(sharding_size_forward, sharding_size_backward_activation,
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

        # shard the output batch dimension to get all possible sharding strategy from this basic strategy
        new_name = f'S{mesh_dim_1}S{mesh_dim_0} = RS{mesh_dim_0} x S{mesh_dim_0}'

        dim_partition_dict_for_output = {0: [mesh_dim_1], 1: [mesh_dim_0]}
        new_sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)
        # the computation cost is all the same
        new_compute_cost = compute_cost

        # the memory cost need to be recomputed
        # compute the memroy cost of new strategy
        new_sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0]
        new_memory_cost, _, memory_cost_backward_activation = self._generate_memory_cost(
            new_sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # the communication cost need to count the sharding cost into this strategy
        # compute the communication cost of new strategy
        origin_communication_cost = communication_cost
        tiny_shard_cost = 10
        new_forward_communication_cost = tiny_shard_cost
        # we need to all gather the batch dimension for the basic strategy
        new_backward_communication_cost = self.device_mesh.all_gather_cost(memory_cost_backward_activation, mesh_dim_1)
        new_communication_cost = origin_communication_cost + new_forward_communication_cost + new_backward_communication_cost

        sharding_strategies = ShardingStrategy(new_name,
                                               output_sharding_spec=new_sharding_spec_for_output,
                                               compute_cost=new_compute_cost,
                                               communication_cost=new_communication_cost,
                                               memory_cost=new_memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_channel_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}'

        dim_partition_dict_for_input = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // (self.device_mesh.shape[mesh_dim_0] *
                                                  self.device_mesh.shape[mesh_dim_1])
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        memory_cost, _, _ = self._generate_memory_cost(sharding_size_forward, sharding_size_backward_activation,
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

    @ignore_sharding_exception
    def non_split(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RR x R'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = 1
        sharding_size_backward_activation = 1
        sharding_size_weight = 1
        memory_cost, _, _ = self._generate_memory_cost(sharding_size_forward, sharding_size_backward_activation,
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

        def _construct_batch_sharding_strategies(mesh_dim_list, new_name):
            dim_partition_dict_for_output = {0: mesh_dim_list}
            new_sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

            # the computation cost is all the same
            new_compute_cost = compute_cost

            # the memory cost need to be recomputed
            new_sharding_size_input = 1
            for mesh_dim in mesh_dim_list:
                new_sharding_size_input = new_sharding_size_input * self.device_mesh.shape[mesh_dim]
            new_memory_cost, _, memory_cost_backward_activation = self._generate_memory_cost(
                new_sharding_size_input, sharding_size_backward_activation, sharding_size_weight)

            # the communication cost need to count the sharding cost into this strategy
            origin_communication_cost = communication_cost
            tiny_shard_cost = 10
            new_forward_communication_cost = tiny_shard_cost
            if len(mesh_dim_list) == 1:
                new_backward_communication_cost = self.device_mesh.all_gather_cost(memory_cost_backward_activation,
                                                                                   mesh_dim_list[0])
            else:
                new_backward_communication_cost = self.device_mesh.flatten_device_mesh.all_gather_cost(
                    memory_cost_backward_activation, 0)
            new_communication_cost = origin_communication_cost + new_forward_communication_cost + new_backward_communication_cost

            new_sharding_strategy = ShardingStrategy(new_name,
                                                     output_sharding_spec=new_sharding_spec_for_output,
                                                     compute_cost=new_compute_cost,
                                                     communication_cost=new_communication_cost,
                                                     memory_cost=new_memory_cost,
                                                     resharding_costs=resharding_costs,
                                                     input_shardings=(sharding_spec_for_input,
                                                                      sharding_spec_for_weight))

            return new_sharding_strategy

        # shard the output batch dimension to get all possible sharding strategy from this basic strategy
        # shard on mesh_dim_0
        new_name = f'S{mesh_dim_0}R = RR x R'
        mesh_dim_list = [mesh_dim_0]
        new_sharding_strategy = _construct_batch_sharding_strategies(mesh_dim_list, new_name)
        self.strategies_vector.append(new_sharding_strategy)

        # shard on mesh_dim_1
        new_name = f'S{mesh_dim_1}R = RR x R'
        mesh_dim_list = [mesh_dim_1]
        new_sharding_strategy = _construct_batch_sharding_strategies(mesh_dim_list, new_name)
        self.strategies_vector.append(new_sharding_strategy)

        # shard on mesh_dim_0, mesh_dim_1
        new_name = f'S{mesh_dim_0}{mesh_dim_1}R = RR x R'
        mesh_dim_list = [mesh_dim_0, mesh_dim_1]
        new_sharding_strategy = _construct_batch_sharding_strategies(mesh_dim_list, new_name)
        self.strategies_vector.append(new_sharding_strategy)

    @ignore_sharding_exception
    def split_input_batch(self, mesh_dim_0):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}R x R WITH SYNC_BN'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = 1
        memory_cost, memory_cost_forward_activation, _ = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        # the all reduce communication will happen during the sync bn computing.
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost_forward_activation, mesh_dim_0)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_batch_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x R WITH SYNC_BN'

        dim_partition_dict_for_input = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // (self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1])
        channel_in = self.input_data.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_weight = 1
        memory_cost, memory_cost_forward_activation, _ = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        # the all reduce communication will happen during the sync bn computing.
        communication_cost = self.device_mesh.flatten_device_mesh.all_reduce_cost(memory_cost_forward_activation, 0)
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
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1} WITH SYNC_BN'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(bs, channel_in)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_1]
        memory_cost, memory_cost_forward_activation, _ = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        # the all reduce communication will happen during the sync bn computing.
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost_forward_activation, mesh_dim_0)
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

        # RS = RS x S and strategies based on it, such as
        # SS = RS x S
        self.split_input_channel(0, 1)
        self.split_input_channel(1, 0)

        # RR = RR x R and strategies based on it, such as
        # SR = SR x R
        self.non_split(0, 1)

        # RS01 = RS01 x S01
        self.split_input_channel_1d(0, 1)

        # SR = SR x R WITH SYNC_BN
        self.split_input_batch(0)
        self.split_input_batch(1)

        # SS = SS x S WITH SYNC_BN
        self.split_input_both_dim(0, 1)
        self.split_input_both_dim(1, 0)

        # S01R = S01R x R WITH SYNC_BN
        self.split_input_batch_1d(0, 1)

        return self.strategies_vector
