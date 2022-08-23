import operator
import torch
from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from .operator_handler import OperatorHandler
from functools import reduce


class DotHandler(OperatorHandler):
    """
    A OperatorHandler which deals with the sharding strategies of linear matrix multiplication.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.output_data = self.node._meta_data

    def _generate_compute_cost(self, input_shape, weight_shape):
        # TODO: consider bias addition
        compute_cost = reduce(operator.mul, input_shape) * weight_shape[0] * 2
        return compute_cost

    def split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        # handle case SS = SR x RS
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        # linear layer weight is transposed during init
        dim_partition_dict_for_weight = {0: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute computation cost
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost
        # no all-reduce required for this case
        communication_cost = 0

        # create and register strategy
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        # handle the case SR = SS x SR
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        # since weight of the linear layer is transposed
        # the actual dim to be sharded is 1
        dim_partition_dict_for_weight = {1: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_1]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def recompute_split_both_contract(self, mesh_dim):
        name = f'RR = RS{mesh_dim} x S{mesh_dim}R'

        dim_partition_dict_for_input = {1: [mesh_dim]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        memory_cost = numel * size_per_elem_bytes

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_rhs_space_only(self, mesh_dim):
        name = f'RS{mesh_dim} = RR x RS{mesh_dim}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(self.input_data.shape, self.weight.shape)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        '''
        Generate every possible strategies for a Conv node, and record all strategies into the strategies_vector.

        Output:

        '''
        # SS = SR x RS
        self.split_lhs_space_rhs_space(0, 1)
        self.split_lhs_space_rhs_space(1, 0)

        # SR = SS x SR
        self.split_lhs_space_both_contract(0, 1)
        self.split_lhs_space_both_contract(1, 0)

        # RS = RS x SS
        self.split_rhs_space_both_contract(0, 1)
        self.split_rhs_space_both_contract(1, 0)

        # RR= RS x SR
        self.recompute_split_both_contract(0)
        self.recompute_split_both_contract(1)

        # RS = RR x RS
        self.split_rhs_space_only(0)
        self.split_rhs_space_only(1)
        return self.strategies_vector
