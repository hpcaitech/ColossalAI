import operator
from functools import reduce
from ..sharding_strategy import ShardingStrategy_V2, TrainCycleItem, MemoryCost
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from .strategy_generator import StrategyGenerator_V2, FollowingStrategyGenerator
from typing import List
from .._utils import exception_handler, enumerate_all_possible_1d_sharding, enumerate_all_possible_2d_sharding
import copy

__all__ = ['WhereGenerator']


class WhereGenerator(StrategyGenerator_V2):
    """
    WhereGenerator is a generic class to generate strategies for Where operation.
    """

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy_V2):
        compute_cost = TrainCycleItem(fwd=10, bwd=10, total=20)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy_V2):
        '''
        Compute the memory cost per device with this specific strategy.
        '''
        forward_size_mapping = {
            'condition': self._compute_size_in_bytes(strategy, "condition"),
            'x': self._compute_size_in_bytes(strategy, "x"),
            'y': self._compute_size_in_bytes(strategy, "y"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = condition + x + y + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items()])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)

        # compute bwd cost incurred
        # bwd_cost = condition_grad + x_grad + y_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items()])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def _generate_strategy_with_dim_partition(self, dim_partition):
        dim_partition_dict_mapping = {
            "condition": dim_partition,
            "x": dim_partition,
            "y": dim_partition,
            "output": dim_partition
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["condition"].sharding_sequence} x {sharding_spec_mapping["x"].sharding_sequence} x {sharding_spec_mapping["y"].sharding_sequence}'
        communication_action_mapping = {}

        strategy = self.get_sharding_strategy(name=name,
                                              sharding_spec_mapping=sharding_spec_mapping,
                                              communication_action_mapping=communication_action_mapping)

        return strategy

    def enumerate_all_possible_output_spec(self, mesh_dim_0, mesh_dim_1, dimension_length):
        dim_partition_list = []
        dim_partition_list.extend(enumerate_all_possible_1d_sharding(mesh_dim_0, dimension_length))
        dim_partition_list.extend(enumerate_all_possible_1d_sharding(mesh_dim_1, dimension_length))
        dim_partition_list.extend(enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, dimension_length))
        # append {} for non_split case
        dim_partition_list.append({})

        return dim_partition_list

    def generate(self):
        '''
        Generate every possible strategies for a where node, and record all strategies into the strategies_vector.
        '''
        strategy_list = []

        dimension_length = len(self.op_data["output"].logical_shape)
        dim_partition_list = self.enumerate_all_possible_output_spec(0, 1, dimension_length)
        for dim_partition in dim_partition_list:
            strategy = self._generate_strategy_with_dim_partition(dim_partition)
            strategy_list.append(strategy)

        for strategy in strategy_list:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategy_list
