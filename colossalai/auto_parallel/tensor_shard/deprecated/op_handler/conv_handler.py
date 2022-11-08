import operator
import warnings
from functools import reduce

import torch

from colossalai.auto_parallel.tensor_shard.deprecated._utils import ignore_sharding_exception
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector

from .operator_handler import OperatorHandler

__all__ = ['ConvHandler']


class ConvHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of Convolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.output_data = self.node._meta_data
        self._sanity_check()

    def _sanity_check(self):
        '''
        In sanity check, we need make sure the input data having correct dimension size.
        For Conv1d, the dim of input data should be 3([N, C, L]).
        For Conv2d, the dim of input data should be 4([N, C, H, W]).
        For Conv3d, the dim of input data should be 5([N, C, H, W, D]).
        '''
        assert self.input_data.dim() in (3, 4,
                                         5), f'We suppose the dim of input fed into conv op should in range of [3, 5].'

    def _generate_compute_cost(self, bs, channel_in, channel_out):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be devided by TFLOPS, now it just shows the computation size.

        Argument:
            bs(int): Batch size of the input data.
            channel_in(int): The channel dimension of input data.
            channel_out(int): The out channel of the conv weight.

        Return:
            compute_cost(float): Computation cost per device with this specific strategy
        '''
        # TODO: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        # 1D: (L) * N * Cout * Cin * kernel
        # 2D: (H * W) * N * Cout * Cin * kernel
        # 3D: (H * W  * D) * N * Cout * Cin * kernel
        output_size = self.output_data.shape[2:]
        output_size_product = reduce(operator.mul, output_size, 1)
        input_size = self.input_data.shape[2:]
        input_size_product = reduce(operator.mul, input_size, 1)
        kernel_size = self.weight.shape[2:]
        kernel_size_product = reduce(operator.mul, kernel_size, 1)
        forward_compute_cost = output_size_product * bs * channel_in * channel_out * kernel_size_product
        backward_activation_cost = input_size_product * bs * channel_in * channel_out * kernel_size_product
        backward_weight_cost = output_size_product * bs * channel_in * channel_out * kernel_size_product
        compute_cost = forward_compute_cost + backward_activation_cost + backward_weight_cost
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
    def split_input_batch_weight_out_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_1]
        memory_cost, _, memory_cost_backward_activation, memory_cost_backward_weight = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # This strategy do not need to do all_reduce operation during forward
        communication_cost_forward = 0
        # compute the backward communication cost to all reduce the input activation grad
        communication_cost_backward_activation = self.device_mesh.all_reduce_cost(memory_cost_backward_activation,
                                                                                  mesh_dim_1)
        # compute the backward communication cost to all reduce the weight due to data parallel
        communication_cost_backward_weight = self.device_mesh.all_reduce_cost(memory_cost_backward_weight, mesh_dim_0)
        # total communication cost
        communication_cost = communication_cost_forward + communication_cost_backward_activation + communication_cost_backward_weight

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_batch(self, mesh_dim_0):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}R x RR'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = 1
        memory_cost, _, _, memory_cost_backward_weight = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        # This strategy do not need to do all_reduce operation in forward phase.
        communication_cost_forward = 0
        # compute the backward communication cost to all reduce the weight due to data parallel
        communication_cost_backward_weight = self.device_mesh.all_reduce_cost(memory_cost_backward_weight, mesh_dim_0)
        # compute the total cost
        communication_cost = communication_cost_forward + communication_cost_backward_weight
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_both_dim_weight_in_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_1]
        memory_cost, memory_cost_forward_activation, _, memory_cost_backward_weight = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # compute the communication cost of this strategy during forward phase
        communication_cost_forward = self.device_mesh.all_reduce_cost(memory_cost_forward_activation, mesh_dim_1)
        # This strategy do not need to do all_reduce operation to compute the input activation grad
        communication_cost_backward_activation = 0
        # compute the backward communication cost to all reduce the weight due to data parallel
        communication_cost_backward_weight = self.device_mesh.all_reduce_cost(memory_cost_backward_weight, mesh_dim_0)
        # compute total cost
        communication_cost = communication_cost_forward + communication_cost_backward_activation + communication_cost_backward_weight
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_input_in_channel_weight_both_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_0]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
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
    def split_input_in_channel_weight_in_channel(self, mesh_dim_0):
        name = f'RR = RS{mesh_dim_0} x S{mesh_dim_0}R'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_0]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = 1
        sharding_size_backward_activation = self.device_mesh.shape[mesh_dim_0]
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0]
        memory_cost, memory_cost_forward_activation, _, _ = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # compute the communication cost of this strategy during forward phase
        communication_cost_forward = self.device_mesh.all_reduce_cost(memory_cost_forward_activation, mesh_dim_0)
        # This strategy do NOT need all_reduce during forward phase
        communication_cost_backward = 0
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
    def split_weight_out_channel(self, mesh_dim_0):
        name = f'RS{mesh_dim_0} = RR x RS{mesh_dim_0}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_0]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.shape[mesh_dim_0]
        sharding_size_backward_activation = 1
        sharding_size_weight = self.device_mesh.shape[mesh_dim_0]
        memory_cost, _, memory_cost_backward_activation, _ = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # This strategy do not need to do all_reduce during forward phase
        communication_cost_forward = 0
        # compute the communication cost of this strategy during backward phase
        communication_cost_backward = self.device_mesh.all_reduce_cost(memory_cost_backward_activation, mesh_dim_0)
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
    def non_split(self):
        name = f'RR = RR x RR'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = 1
        sharding_size_backward_activation = 1
        sharding_size_weight = 1
        memory_cost, _, _, _ = self._generate_memory_cost(sharding_size_forward, sharding_size_backward_activation,
                                                          sharding_size_weight)

        # This strategy do not need to do all_reduce in both forward and backward phase
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
    def split_1d_parallel_on_input_batch(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x RR'

        dim_partition_dict_for_input = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // (self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1])
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = self.device_mesh.mesh_shape[mesh_dim_0] * self.device_mesh.mesh_shape[mesh_dim_1]
        sharding_size_backward_activation = self.device_mesh.mesh_shape[mesh_dim_0] * self.device_mesh.mesh_shape[
            mesh_dim_1]
        sharding_size_weight = 1
        memory_cost, _, _, memory_cost_backward_weight = self._generate_memory_cost(sharding_size_forward,
                                                                                    sharding_size_backward_activation,
                                                                                    sharding_size_weight)

        # This strategy do not need to do all_reduce in forward phase
        communication_cost_forward = 0
        # compute the backward communication cost to all reduce the weight due to data parallel
        communication_cost_backward_weight = self.device_mesh.flatten_device_mesh.all_reduce_cost(
            memory_cost_backward_weight, 0)
        # compute the total communication cost
        communication_cost = communication_cost_backward_weight + communication_cost_forward

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def split_1d_parallel_on_in_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'

        dim_partition_dict_for_input = {1: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0, mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input, sharding_spec_for_weight])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // (self.device_mesh.shape[mesh_dim_0] *
                                                  self.device_mesh.shape[mesh_dim_1])
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        sharding_size_forward = 1
        sharding_size_backward_activation = self.device_mesh.mesh_shape[mesh_dim_0] * self.device_mesh.mesh_shape[
            mesh_dim_1]
        sharding_size_weight = self.device_mesh.mesh_shape[mesh_dim_0] * self.device_mesh.mesh_shape[mesh_dim_1]
        memory_cost, memory_cost_forward_activation, _, _ = self._generate_memory_cost(
            sharding_size_forward, sharding_size_backward_activation, sharding_size_weight)

        # compute communication cost during forward phase
        communication_cost_forward = self.device_mesh.flatten_device_mesh.all_reduce_cost(
            memory_cost_forward_activation, 0)
        # This strategy do NOT need do all_reduce during backward phase
        communication_cost_backward = 0
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

        Example:
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1]
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
            shape_consistency_manager = ShapeConsistencyManager()

            model = ConvModel(16, 32)
            input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}
            # graph():
            #     %x : torch.Tensor [#users=1] = placeholder[target=x]
            #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
            #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
            #     return conv
            graph = tracer.trace(root=model, meta_args=input_sample)
            gm = GraphModule(model, graph, model.__class__.__name__)
            gm.recompile()
            # [x, mul, conv, output]
            nodes = [node for node in gm.graph.nodes]

            # strategies_for_input = [[R, R, R, R], [R, S0, R, R], [R, S1, R, R], [S0, R, R, R], [S0, S1, R, R], [S1, R, R, R], [S1, S0, R, R]]
            strategies_vector_for_input = StrategiesVector(node=nodes[0], in_nodes=[nodes[1], 2], strategies=strategies_for_input)
            setattr(nodes[1], 'strategies_vector', strategies_vector_for_input)

            strategies_vector = StrategiesVector(node=nodes[2], in_nodes=[nodes[1], ])
            conv_handler = ConvHandler(input_node=nodes[1], input_index=0, weight=dict(gm.named_modules())[nodes[2].name].weight, output_node=nodes[2],
                                    device_mesh=device_mesh, strategies_vector=strategies_vector, shape_consistency_manager=shape_consistency_manager)
            conv_handler.register_strategy_into_strategies_vector()
            for strategy in conv_handler.strategies_vector:
                print(f'{strategy.name}: compute_cost is {strategy.compute_cost}, communication_cost is {strategy.communication_cost}, memory_cost is {strategy.memory_cost}, resharding_costs is {strategy.resharding_costs}')

        Output:
            S0S1 = S0R x RS1: compute_cost is 8856576, communication_cost is 0, memory_cost is 492032.0, resharding_costs is {mul: [0, 32769.001, 131074.2, 0, 32769.1, 131074.2, 98307.201]}
            S1S0 = S1R x RS0: compute_cost is 8856576, communication_cost is 0, memory_cost is 492032.0, resharding_costs is {mul: [0, 131074.2, 32769.001, 131074.2, 98307.201, 0, 32769.1]}
            S0R = S0R x RR: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {mul: [0, 32769.001, 131074.2, 0, 32769.1, 131074.2, 98307.201]}
            S1R = S1R x RR: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {mul: [0, 131074.2, 32769.001, 131074.2, 98307.201, 0, 32769.1]}
            S0R = S0S1 x S1R: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {mul: [0, 65538.002, 0, 0, 0, 65538.002, 196614.402]}
            S1R = S1S0 x S0R: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {mul: [0, 0, 65538.002, 65538.002, 196614.402, 0, 0]}
            RS1 = RS0 x S0S1: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {mul: [0, 0, 131074.2, 32769.001, 98307.201, 131074.2, 32769.1]}
            RS0 = RS1 x S1S0: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {mul: [0, 131074.2, 0, 131074.2, 32769.1, 32769.001, 98307.201]}
            RR = RS0 x S0R: compute_cost is 17713152, communication_cost is 1968129.01, memory_cost is 1968128, resharding_costs is {mul: [0, 0, 131074.2, 32769.001, 98307.201, 131074.2, 32769.1]}
            RR = RS1 x S1R: compute_cost is 17713152, communication_cost is 1968129.01, memory_cost is 1968128, resharding_costs is {mul: [0, 131074.2, 0, 131074.2, 32769.1, 32769.001, 98307.201]}
            RS0 = RR x RS0: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {mul: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
            RS1 = RR x RS1: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {mul: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
            RR = RR x RR: compute_cost is 35426304, communication_cost is 0, memory_cost is 1968128, resharding_costs is {mul: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
            S01R = S01R x RR: compute_cost is 8856576, communication_cost is 0, memory_cost is 492032.0, resharding_costs is {mul: [0, 65538.002, 262148.4, 0, 16385.001, 262148.4, 196614.402]}
            RR = RS01 x S01R: compute_cost is 8856576, communication_cost is 0, memory_cost is 1968128, resharding_costs is {mul: [0, 0, 262148.4, 65538.002, 196614.402, 262148.4, 65538.2]}
        '''
        # SS = SR x RS
        self.split_input_batch_weight_out_channel(0, 1)
        self.split_input_batch_weight_out_channel(1, 0)

        # SR = SR x RR
        self.split_input_batch(0)
        self.split_input_batch(1)

        # SR = SS x SR
        self.split_input_both_dim_weight_in_channel(0, 1)
        self.split_input_both_dim_weight_in_channel(1, 0)

        # RS = RS x SS
        self.split_input_in_channel_weight_both_channel(0, 1)
        self.split_input_in_channel_weight_both_channel(1, 0)

        # RR = RS x SR
        self.split_input_in_channel_weight_in_channel(0)
        self.split_input_in_channel_weight_in_channel(1)

        # RS = RR x RS
        self.split_weight_out_channel(0)
        self.split_weight_out_channel(1)

        # RR= RR x RR
        self.non_split()

        # S01R = S01R x RR
        self.split_1d_parallel_on_input_batch(0, 1)

        # RR = RS01 x S01R
        self.split_1d_parallel_on_in_channel(0, 1)

        return self.strategies_vector


CONV_STRATEGIES_LIST = [
    'S0S1 = S0R x RS1', 'S1S0 = S1R x RS0', 'S0R = S0R x RR', 'S1R = S1R x RR', 'S0R = S0S1 x S1R', 'S1R = S1S0 x S0R',
    'RS1 = RS0 x S0S1', 'RS0 = RS1 x S1S0', 'RR = RS0 x S0R', 'RR = RS1 x S1R', 'RS0 = RR x RS0', 'RS1 = RR x RS1',
    'RR = RR x RR', 'S01R = S01R x RR', 'RR = RS01 x S01R'
]
