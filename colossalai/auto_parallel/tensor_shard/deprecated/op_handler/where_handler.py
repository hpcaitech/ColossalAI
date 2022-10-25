import operator
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List

import torch

from colossalai.auto_parallel.tensor_shard.deprecated._utils import (enumerate_all_possible_1d_sharding,
                                                                     enumerate_all_possible_2d_sharding,
                                                                     ignore_sharding_exception)
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import (ShardingStrategy, StrategiesVector)
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from .operator_handler import OperatorHandler

__all__ = ['WhereHandler']


class WhereHandler(OperatorHandler):
    """
    An OperatorHandler which deals with the sharding strategies of torch.where.
    """

    def __init__(self, *args, **kwargs):
        # TODO: x or y could be scalar
        super().__init__(*args, **kwargs)
        assert len(self.predecessor_node) == 3
        self.condition_data = self.predecessor_node[0]._meta_data
        self.x_data = self.predecessor_node[1]._meta_data
        self.y_data = self.predecessor_node[2]._meta_data
        self.condition = self.predecessor_node[0]
        self.x = self.predecessor_node[1]
        self.y = self.predecessor_node[2]
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
        for dim_index, sharding_index_list in processed_dim_partition_dict.items():
            sharding_list = [self.device_mesh.mesh_shape[sharding_index] for sharding_index in sharding_index_list]
            sharding_size = reduce(operator.mul, sharding_list, 1)
            assert shape[
                dim_index] % sharding_size == 0, f'we cannot shard the {dim_index} dimension of tensor into {sharding_size} partitions.'
        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                     entire_shape=shape,
                                     dim_partition_dict=processed_dim_partition_dict)

        return sharding_spec

    def _generate_compute_cost(self, total_sharding_size):
        lhs_matrix_shape = self.lhs_data.shape[-2:]
        rhs_matrix_shape = self.rhs_data.shape[-2:]
        batch_dimensions_shape = self.output_data.shape[:-2]
        batch_dimensions_product = reduce(operator.mul, batch_dimensions_shape, 1)
        compute_cost = reduce(
            operator.mul, lhs_matrix_shape) * rhs_matrix_shape[0] * batch_dimensions_product * 2 / total_sharding_size
        return compute_cost

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

                # compute the resharding cost
                _, _, total_resharding_cost = shape_consistency_manager.shape_consistency(
                    input_sharding_spec, input_spec)
                total_resharding_cost = total_resharding_cost['total']
                # we need multiply the size of elem dtype to get correct communication cost
                resharding_cost = total_resharding_cost * size_per_elem_bytes
                resharding_costs[input_node].append(resharding_cost)

        return resharding_costs

    def _convert_partition_dict_to_sharding_spec(self, dim_partition_list):

        sharding_spec_list = []
        check_duplicated_list = []
        for output_dim_partition_dict in dim_partition_list:
            try:
                output_sharding_spec = self._generate_sharding_spec(self.output_data, output_dim_partition_dict)
            except AssertionError as e:
                warnings.warn(f'{e}')
                break
            sharding_seq = output_sharding_spec.sharding_sequence
            if sharding_seq not in check_duplicated_list:
                check_duplicated_list.append(sharding_seq)
                sharding_spec_list.append(output_sharding_spec)

        return sharding_spec_list

    def _enumerate_all_possible_output(self, mesh_dim_0, mesh_dim_1):
        # use mesh_dim_0, mesh_dim_1 instead of constant 0, 1 in here for N-D device mesh scaliablity.

        output_dim_partition_list = []
        dim_size = self.output_data.dim()
        # enumerate all the 2D sharding cases
        sharding_list_2d = enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, dim_size)
        output_dim_partition_list.extend(sharding_list_2d)

        # enumerate all the 1D sharding cases
        sharding_list_1d_on_dim_0 = enumerate_all_possible_1d_sharding(mesh_dim_0, dim_size)
        output_dim_partition_list.extend(sharding_list_1d_on_dim_0)
        sharding_list_1d_on_dim_1 = enumerate_all_possible_1d_sharding(mesh_dim_1, dim_size)
        output_dim_partition_list.extend(sharding_list_1d_on_dim_1)

        # add empty dict for fully replicated case
        output_dim_partition_list.append({})
        output_sharding_spec_list = self._convert_partition_dict_to_sharding_spec(output_dim_partition_list)

        return output_sharding_spec_list

    @ignore_sharding_exception
    def _register_strategy(self, output_sharding_spec):
        dim_partition_dict_for_input = output_sharding_spec.dim_partition_dict
        sharding_spec_for_condition = self._generate_sharding_spec(self.condition_data, dim_partition_dict_for_input)
        sharding_spec_for_x = self._generate_sharding_spec(self.x_data, dim_partition_dict_for_input)
        sharding_spec_for_y = self._generate_sharding_spec(self.y_data, dim_partition_dict_for_input)

        name = f'{output_sharding_spec.sharding_sequence} = {sharding_spec_for_condition.sharding_sequence} x {sharding_spec_for_x.sharding_sequence} x {sharding_spec_for_y.sharding_sequence}'
        dim_partition_dict_for_output = output_sharding_spec.dim_partition_dict

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs(
            [sharding_spec_for_condition, sharding_spec_for_x, sharding_spec_for_y])

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
                                               input_shardings=(sharding_spec_for_condition, sharding_spec_for_x,
                                                                sharding_spec_for_y))

        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        MESH_DIM_LIST = [0, 1]
        output_sharding_specs = self._enumerate_all_possible_output(MESH_DIM_LIST[0], MESH_DIM_LIST[1])
        for output_sharding_spec in output_sharding_specs:
            self._register_strategy(output_sharding_spec)
