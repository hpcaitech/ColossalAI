import copy
import operator
from functools import reduce
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.auto_parallel.tensor_shard.utils import (
    enumerate_all_possible_1d_sharding,
    enumerate_all_possible_2d_sharding,
    ignore_sharding_exception,
)
from colossalai.tensor.shape_consistency import CollectiveCommPattern

from .strategy_generator import StrategyGenerator

__all__ = ['LayerNormGenerator']


class LayerNormGenerator(StrategyGenerator):
    """
    LayerNormGenerator is a generic class to generate strategies for LayerNorm operation.
    The operation data is defined as `output = input x other + bias`.
    """

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        '''
        # TODO: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        # TODO: a constant coefficient need to be added.

        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        sharded_weight_shape = strategy.sharding_specs[self.op_data['other']].get_sharded_shape_per_device()
        if self.has_bias:
            # bias add is an element wise operation, so the cost is equal to product of output shape.
            bias_compute_cost = reduce(operator.mul, sharded_weight_shape)
        # in LayerNorm context, batch dimensions mean all the dimensions do not join the normalization.
        input_batch_shape = sharded_input_shape[:-len(sharded_weight_shape)]
        input_batch_product = reduce(operator.mul, input_batch_shape, 1)
        norm_kernel_product = reduce(operator.mul, sharded_weight_shape, 1)
        forward_compute_cost = input_batch_product * norm_kernel_product
        backward_activation_compute_cost = input_batch_product * norm_kernel_product
        # To compute gradient of on norm kernel element requires input_batch_product times computation, so
        # the total cost is input_batch_product * norm_kernel_product
        backward_weight_compute_cost = input_batch_product * norm_kernel_product
        backward_compute_cost = backward_activation_compute_cost + backward_weight_compute_cost
        if self.has_bias:
            forward_compute_cost += bias_compute_cost
            backward_compute_cost += bias_compute_cost
        total_compute_cost = forward_compute_cost + backward_compute_cost
        compute_cost = TrainCycleItem(fwd=forward_compute_cost, bwd=backward_compute_cost, total=total_compute_cost)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        '''
        Compute the memory cost per device with this specific strategy.
        '''
        forward_size_mapping = {
            'input': self._compute_size_in_bytes(strategy, "input"),
            'other': self._compute_size_in_bytes(strategy, "other"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        if self.has_bias:
            bias_size = self._compute_size_in_bytes(strategy, "bias")
            forward_size_mapping['bias'] = bias_size

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + other + bias + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items() if not self.is_param(k)])
        fwd_parameter_cost = sum([v for k, v in forward_size_mapping.items() if self.is_param(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad + other_grad + bias_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items() if not self.is_param(k)])
        bwd_parameter_cost = sum([v for k, v in backward_size_mapping.items() if self.is_param(k)])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=bwd_parameter_cost)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost,
                                    parameter=fwd_parameter_cost + bwd_parameter_cost)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def _generate_strategy_with_dim_partition(self, dim_partition):
        dim_partition_dict_mapping = {
            "input": dim_partition,
            "other": {},
            "output": dim_partition,
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["input"].sharding_sequence} x {sharding_spec_mapping["other"].sharding_sequence}'
        total_mesh_dim_list = []
        for mesh_dim_list in dim_partition.values():
            total_mesh_dim_list.extend(mesh_dim_list)
        # if there is only one sharding dimension, we should use the value instead of list as logical_process_axis.
        if len(total_mesh_dim_list) == 1:
            total_mesh_dim_list = total_mesh_dim_list[0]
        communication_action_mapping = {}

        other_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["other"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=total_mesh_dim_list,
            comm_type=CommType.HOOK)
        communication_action_mapping["other"] = other_comm_action

        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping["bias"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=total_mesh_dim_list,
                comm_type=CommType.HOOK)
            communication_action_mapping["bias"] = bias_comm_action

        strategy = self.get_sharding_strategy(name=name,
                                              sharding_spec_mapping=sharding_spec_mapping,
                                              communication_action_mapping=communication_action_mapping)

        return strategy

    def split_input_batch_single_mesh_dim(self, mesh_dim_0, batch_dimension_length):
        strategy_list = []
        dim_partition_list = enumerate_all_possible_1d_sharding(mesh_dim_0, batch_dimension_length)
        for dim_partition in dim_partition_list:
            strategy = self._generate_strategy_with_dim_partition(dim_partition)
            strategy_list.append(strategy)
        return strategy_list

    def split_input_batch_both_mesh_dim(self, mesh_dim_0, mesh_dim_1, batch_dimension_length):
        strategy_list = []
        dim_partition_list = enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, batch_dimension_length)
        for dim_partition in dim_partition_list:
            strategy = self._generate_strategy_with_dim_partition(dim_partition)
            strategy_list.append(strategy)
        return strategy_list

    @ignore_sharding_exception
    def non_split(self):
        name = f'RR = RR x R'
        dim_partition_dict_mapping = {
            "input": {},
            "other": {},
            "output": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        communication_action_mapping = {}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def collate_strategies(self) -> List[ShardingStrategy]:
        '''
        Generate every possible strategies for a LayerNorm node, and record all strategies into the strategies_vector.
        '''
        strategy_list = []
        input_data_dim = len(self.op_data["input"].logical_shape)
        weight_data_dim = len(self.op_data["other"].logical_shape)
        # in LayerNorm context, batch dimensions mean all the dimensions do not join the normalization.
        batch_dimension_length = input_data_dim - weight_data_dim

        # SR = SR x R with single mesh dim on batch dimensions
        strategy_list.extend(self.split_input_batch_single_mesh_dim(0, batch_dimension_length))
        strategy_list.extend(self.split_input_batch_single_mesh_dim(1, batch_dimension_length))

        # SR = SR x R with both mesh dims on batch dimensions
        strategy_list.extend(self.split_input_batch_both_mesh_dim(0, 1, batch_dimension_length))

        # RR = RR x R
        strategy_list.append(self.non_split())

        return strategy_list
