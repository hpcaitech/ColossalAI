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

                # we need multiply the size of elem dtype to get correct communication cost
                resharding_cost = total_resharding_cost["total"] * size_per_elem_bytes
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

    ##############################################
    #used to generate strategies for torch.matmul#
    ##############################################
    @ignore_sharding_exception
    def _registry_no_split_strategies_for_matmul(self, dim_partition_dict_for_batch_dim):
        # this dim partition dict only describes the batch dimensions, but in this scenario,
        # matrix dimensions are fully replicated, so it do not need extra process.
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_batch_dim)
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_batch_dim)
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_batch_dim)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        batch_sharding_dims = []
        for mesh_dims in dim_partition_dict_for_batch_dim.values():
            for mesh_dim in mesh_dims:
                batch_sharding_dims.append(self.device_mesh.shape[mesh_dim])
        batch_sharding_size = reduce(operator.mul, batch_sharding_dims, 1)
        # in this case, total_sharding_size is equal to the batch sharding size
        memory_cost = self.output_data.numel() / batch_sharding_size

        # compute the computation cost of this strategy
        compute_cost = self._generate_compute_cost(batch_sharding_size)

        # in this case, no communication takes place.
        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        communication_cost = 0

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def _split_dim_i(self, dim_partition_dict_for_batch_dim, mesh_dim_on_matrix):
        # A batched matrix multiplication can be viewed as [b, i, k] x [b, k, j] -> [b, i, j]
        # this dim partition dict describe the batch dimensions, so we should append the matrix dimension sharding info on it.
        # In this scenario, matrix dimensions will be sharded on 'i' dimension.

        # in this case, the matrix dimensions of lhs is sharded on 'i' dimension.
        dim_partition_dict_for_lhs = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_lhs.update({-2: mesh_dim_on_matrix})

        # in this case, the matrix dimensions of rhs is fully replicated.
        dim_partition_dict_for_rhs = deepcopy(dim_partition_dict_for_batch_dim)

        # in this case, the matrix dimensions of output is sharded on 'i' dimension.

        dim_partition_dict_for_output = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_output.update({-2: mesh_dim_on_matrix})

        # generate sharding specs
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_dims = []

        # append batch sharding dims
        for mesh_dims in dim_partition_dict_for_batch_dim.values():
            for mesh_dim in mesh_dims:
                total_sharding_dims.append(self.device_mesh.shape[mesh_dim])

        # append the sharding dims on matrix dimension
        for mesh_dim in mesh_dim_on_matrix:
            total_sharding_dims.append(self.device_mesh.shape[mesh_dim])
        total_sharding_size = reduce(operator.mul, total_sharding_dims, 1)

        # in this case, output_data uses all the sharding dims.
        memory_cost = self.output_data.numel() / total_sharding_size
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        communication_cost = 0

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def _split_dim_k(self, dim_partition_dict_for_batch_dim, mesh_dim_on_matrix):
        # A batched matrix multiplication can be viewed as [b, i, k] x [b, k, j] -> [b, i, j]
        # this dim partition dict describe the batch dimensions, so we should append the matrix dimension sharding info on it.
        # In this scenario, matrix dimensions will be sharded on 'k' dimension.

        # in this case, the matrix dimensions of lhs is sharded on 'k' dimension.
        dim_partition_dict_for_lhs = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_lhs.update({-1: mesh_dim_on_matrix})

        # in this case, the matrix dimensions of rhs is sharded on 'k' dimension.
        dim_partition_dict_for_rhs = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_rhs.update({-2: mesh_dim_on_matrix})

        # in this case, the matrix dimensions of output is fully replicated.
        dim_partition_dict_for_output = deepcopy(dim_partition_dict_for_batch_dim)

        # generate sharding specs
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_dims = []
        batch_sharding_dims = []
        # append batch sharding dims
        for mesh_dims in dim_partition_dict_for_batch_dim.values():
            for mesh_dim in mesh_dims:
                total_sharding_dims.append(self.device_mesh.shape[mesh_dim])
                batch_sharding_dims.append(self.device_mesh.shape[mesh_dim])

        # append the sharding dims on matrix dimension
        for mesh_dim in mesh_dim_on_matrix:
            total_sharding_dims.append(self.device_mesh.shape[mesh_dim])
        batch_sharding_size = reduce(operator.mul, batch_sharding_dims, 1)
        total_sharding_size = reduce(operator.mul, total_sharding_dims, 1)

        # in this case, output_data is fully replicated on matrix dimensions.
        memory_cost = self.output_data.numel() / batch_sharding_size

        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        # The communication takes place during forward activation computation.
        if len(mesh_dim_on_matrix) == 1:
            communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_on_matrix[0])
        else:
            communication_cost = self.device_mesh.flatten_device_mesh.all_reduce_cost(memory_cost, 0)

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def _split_dim_j(self, dim_partition_dict_for_batch_dim, mesh_dim_on_matrix):
        # A batched matrix multiplication can be viewed as [b, i, k] x [b, k, j] -> [b, i, j]
        # this dim partition dict describe the batch dimensions, so we should append the matrix dimension sharding info on it.
        # In this scenario, matrix dimensions will be is sharded on 'j' dimension.

        # in this case, the matrix dimensions of lhs is fully replicated.
        dim_partition_dict_for_lhs = deepcopy(dim_partition_dict_for_batch_dim)

        # in this case, the matrix dimensions of rhs is sharded on 'j' dimension.
        dim_partition_dict_for_rhs = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_rhs.update({-1: mesh_dim_on_matrix})

        # in this case, the matrix dimensions of output is sharded on 'j' dimension.
        dim_partition_dict_for_output = deepcopy(dim_partition_dict_for_batch_dim)
        dim_partition_dict_for_output.update({-1: mesh_dim_on_matrix})

        # generate sharding specs
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_dims = []

        # append batch sharding dims
        for mesh_dims in dim_partition_dict_for_batch_dim.values():
            for mesh_dim in mesh_dims:
                total_sharding_dims.append(self.device_mesh.shape[mesh_dim])

        # append the sharding dims on matrix dimension
        for mesh_dim in mesh_dim_on_matrix:
            total_sharding_dims.append(self.device_mesh.shape[mesh_dim])
        total_sharding_size = reduce(operator.mul, total_sharding_dims, 1)

        # in this case, output_data uses all the sharding dims.
        memory_cost = self.output_data.numel() / total_sharding_size
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        # The communication takes place during backward activation computation.
        if len(mesh_dim_on_matrix) == 1:
            communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_on_matrix[0])
        else:
            communication_cost = self.device_mesh.flatten_device_mesh.all_reduce_cost(memory_cost, 0)

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    def _registry_1d_strategies_for_matmul(self, dim_partition_dict, mesh_dim_list):
        self._split_dim_i(dim_partition_dict, mesh_dim_list)
        self._split_dim_k(dim_partition_dict, mesh_dim_list)
        self._split_dim_j(dim_partition_dict, mesh_dim_list)

    @ignore_sharding_exception
    def _split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        dim_partition_dict_for_lhs = {-2: [mesh_dim_0], -1: [mesh_dim_1]}
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)

        dim_partition_dict_for_rhs = {-2: [mesh_dim_1]}
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)

        dim_partition_dict_for_output = {-2: [mesh_dim_0]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_size = reduce(operator.mul, self.device_mesh.shape, 1)
        output_sharding_size = reduce(operator.mul, self.output_data.shape, 1)
        # in this case, output_data uses all the sharding dims.
        memory_cost = self.output_data.numel() / output_sharding_size
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        # The communication takes place during forward activation computation.
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def _split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        dim_partition_dict_for_lhs = {-1: [mesh_dim_0]}
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)

        dim_partition_dict_for_rhs = {-2: [mesh_dim_0], -1: [mesh_dim_1]}
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)

        dim_partition_dict_for_output = {-1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_size = reduce(operator.mul, self.device_mesh.shape, 1)
        output_sharding_size = reduce(operator.mul, self.output_data.shape, 1)
        # in this case, output_data uses all the sharding dims.
        memory_cost = self.output_data.numel() / output_sharding_size
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        # The communication takes place during forward and backward activation computation.
        communication_cost_forward_activation = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_0)
        communication_cost_backward_activation = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        communication_cost = communication_cost_backward_activation + communication_cost_forward_activation

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    @ignore_sharding_exception
    def _split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        dim_partition_dict_for_lhs = {-2: [mesh_dim_0]}
        sharding_spec_for_lhs = self._generate_sharding_spec(self.lhs_data, dim_partition_dict_for_lhs)

        dim_partition_dict_for_rhs = {-1: [mesh_dim_1]}
        sharding_spec_for_rhs = self._generate_sharding_spec(self.rhs_data, dim_partition_dict_for_rhs)

        dim_partition_dict_for_output = {-2: [mesh_dim_0], -1: [mesh_dim_1]}
        sharding_spec_for_output = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        name = f'{sharding_spec_for_output.sharding_sequence} = {sharding_spec_for_lhs.sharding_sequence} x {sharding_spec_for_rhs.sharding_sequence}'

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_lhs, sharding_spec_for_rhs])

        # compute the memory cost of this strategy
        total_sharding_size = reduce(operator.mul, self.device_mesh.shape, 1)
        output_sharding_size = reduce(operator.mul, self.output_data.shape, 1)
        # in this case, output_data uses all the sharding dims.
        memory_cost = self.output_data.numel() / output_sharding_size
        compute_cost = self._generate_compute_cost(total_sharding_size)

        # TODO: add all-reduce cost if lhs or rhs is type of Parameters.
        # The communication takes place during backward activation computation.
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_output,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_lhs, sharding_spec_for_rhs))

        self.strategies_vector.append(sharding_strategies)

    def _registry_2d_strategies_for_matmul(self):
        self._split_lhs_space_both_contract(0, 1)
        self._split_lhs_space_both_contract(1, 0)
        self._split_rhs_space_both_contract(0, 1)
        self._split_rhs_space_both_contract(1, 0)
        self._split_lhs_space_rhs_space(0, 1)
        self._split_lhs_space_rhs_space(1, 0)

    def register_strategy(self) -> StrategiesVector:
        MESH_DIM_LIST = [0, 1]
        if self.node.target != torch.matmul:
            output_sharding_specs = self._enumerate_all_possible_output(MESH_DIM_LIST[0], MESH_DIM_LIST[1])
            for output_sharding_spec in output_sharding_specs:
                self._register_strategy(output_sharding_spec)
        else:
            # we only care about the non-computing dimensions,
            # therefore, we omit the last two dimensions.
            dim_size = self.output_data.dim() - 2

            # Both device mesh axises are uesd on batch dimensions
            dim_partition_dicts_2d = enumerate_all_possible_2d_sharding(MESH_DIM_LIST[0], MESH_DIM_LIST[1], dim_size)
            for dim_partition_dict in dim_partition_dicts_2d:
                self._registry_no_split_strategies_for_matmul(dim_partition_dict)

            # Only one device mesh axis is uesd on batch dimensions
            for mesh_dim_index in [0, 1]:
                dim_partition_dicts_1d = enumerate_all_possible_1d_sharding(MESH_DIM_LIST[mesh_dim_index], dim_size)
                for dim_partition_dict in dim_partition_dicts_1d:
                    self._registry_no_split_strategies_for_matmul(dim_partition_dict)
                    self._registry_1d_strategies_for_matmul(dim_partition_dict, [MESH_DIM_LIST[mesh_dim_index - 1]])

            # No device mesh axis is uesd on batch dimensions
            dim_partition_dict_on_batch_dim = {}
            self._registry_no_split_strategies_for_matmul(dim_partition_dict_on_batch_dim)
            self._registry_1d_strategies_for_matmul(dim_partition_dict_on_batch_dim, MESH_DIM_LIST)
            self._registry_2d_strategies_for_matmul()
