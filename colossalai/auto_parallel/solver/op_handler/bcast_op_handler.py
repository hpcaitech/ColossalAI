import operator
from functools import reduce
import warnings
import torch
from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from .operator_handler import OperatorHandler
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec
from copy import deepcopy
from typing import Dict, List
from colossalai.auto_parallel.solver._utils import exception_handler

__all__ = ['BcastOpHandler']


class BcastOpHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of broadcast operators(such as operator.add).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.predecessor_node) == 2
        self.lhs_data = self.predecessor_node[0]._meta_data
        self.rhs_data = self.predecessor_node[1]._meta_data
        self.lhs = self.predecessor_node[0]
        self.rhs = self.predecessor_node[1]
        self.output_data = self.node._meta_data

    def _generate_sharding_spec(self, input_: torch.Tensor, dim_partition_dict: Dict[int, List[int]]) -> ShardingSpec:
        shape = list(input_.shape)

        # padding the shape to the same length as output_data
        while len(shape) < self.output_data.dim():
            shape.insert(0, 1)
        shape = torch.Size(shape)

        # if the sharding happens on a size one dimension, we should record it as R.
        processed_dim_partition_dict = deepcopy(dim_partition_dict)
        for dim_index, _ in dim_partition_dict.items():
            if shape[dim_index] == 1:
                processed_dim_partition_dict.pop(dim_index)
        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                     entire_shape=shape,
                                     dim_partition_dict=processed_dim_partition_dict)

        return sharding_spec

    def _generate_resharding_costs(self, sharding_specs):
        # The resharding_cost of weight is counted due to sharing weight cases.
        dtype = self.node._meta_data.dtype
        nodes = self.predecessor_node
        resharding_costs = {}
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()

        # shape consistency manager is a singleton class
        shape_consistency_manager = ShapeConsistencyManager()

        for input_node, input_spec in zip(nodes, sharding_specs):
            resharding_costs[input_node] = []
            for strategy in input_node.strategies_vector:
                input_sharding_spec = strategy.output_sharding_spec
                assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'
                # if the input shape is smaller than the target input, we will fill the input to the same length as target.
                # Then, use the padded input sharding spec to compute the resharding cost.
                if len(input_sharding_spec.entire_shape) < len(input_spec.entire_shape):
                    new_entire_shape = list(input_sharding_spec.entire_shape)
                    while len(new_entire_shape) < len(input_spec.entire_shape):
                        new_entire_shape.insert(0, 1)
                    new_entire_shape = torch.Size(new_entire_shape)
                    new_device_mesh = input_sharding_spec.device_mesh
                    new_dim_partition_dict = input_sharding_spec.dim_partition_dict
                    input_sharding_spec = ShardingSpec(device_mesh=new_device_mesh,
                                                       entire_shape=new_entire_shape,
                                                       dim_partition_dict=new_dim_partition_dict)

                # compute the resharding cost during forward phase
                _, _, resharding_cost_forward = shape_consistency_manager.shape_consistency(
                    input_sharding_spec, input_spec)

                _, _, resharding_cost_backward = shape_consistency_manager.shape_consistency(
                    input_spec, input_sharding_spec)
                total_resharding_cost = resharding_cost_forward + resharding_cost_backward

                # we need multiply the size of elem dtype to get correct communication cost
                resharding_cost = total_resharding_cost * size_per_elem_bytes
                resharding_costs[input_node].append(resharding_cost)

        return resharding_costs

    def _enumerate_all_possible_output(self, mesh_dim_0, mesh_dim_1):
        # use mesh_dim_0, mesh_dim_1 instead of constant 0, 1 in here for N-D device mesh scaliablity.

        output_sharding_spec_list = []
        output_dim_partition_list = []

        # enumerate all the 2D sharding cases
        for i in range(self.output_data.dim()):
            for j in range(i + 1, self.output_data.dim()):
                dim_partition_dict_0 = {i: [mesh_dim_0], j: [mesh_dim_1]}
                dim_partition_dict_1 = {i: [mesh_dim_1], j: [mesh_dim_0]}
                output_dim_partition_list.append(dim_partition_dict_0)
                output_dim_partition_list.append(dim_partition_dict_1)

        # enumerate all the 1D sharding cases
        for i in range(self.output_data.dim()):
            dim_partition_dict_0 = {i: [mesh_dim_0]}
            dim_partition_dict_1 = {i: [mesh_dim_1]}
            dim_partition_dict_flatten = {i: [mesh_dim_0, mesh_dim_1]}
            output_dim_partition_list.append(dim_partition_dict_0)
            output_dim_partition_list.append(dim_partition_dict_1)
            output_dim_partition_list.append(dim_partition_dict_flatten)

        # add empty dict for fully replicated case
        output_dim_partition_list.append({})
        check_duplicated_list = []
        for output_dim_partition_dict in output_dim_partition_list:
            output_sharding_spec = self._generate_sharding_spec(self.output_data, output_dim_partition_dict)
            sharding_seq = output_sharding_spec.sharding_sequence
            if sharding_seq not in check_duplicated_list:
                check_duplicated_list.append(sharding_seq)
                output_sharding_spec_list.append(output_sharding_spec)

        return output_sharding_spec_list

    def _generate_compute_cost(self, *args, **kwargs):
        return super()._generate_compute_cost(*args, **kwargs)

    @exception_handler
    def _register_strategy(self, output_sharding_spec):
        dim_partition_dict_for_input = output_sharding_spec.dim_partition_dict
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_input)
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_input)

        name = f'{output_sharding_spec.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'
        dim_partition_dict_for_output = output_sharding_spec.dim_partition_dict

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the computation cost of this strategy
        sharding_dims = []
        for mesh_dims in dim_partition_dict_for_output.values():
            for mesh_dim in mesh_dims:
                sharding_dims.append(self.device_mesh.shape[mesh_dim])
        sharding_size = reduce(operator.mul, sharding_dims, 1)
        memory_cost = self.output_data.numel() / sharding_size
        compute_cost = memory_cost
        communication_cost = 0

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=output_sharding_spec,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        output_sharding_specs = self._enumerate_all_possible_output(0, 1)
        for output_sharding_spec in output_sharding_specs:
            self._register_strategy(output_sharding_spec)
