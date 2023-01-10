from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, ShardingStrategy, TrainCycleItem
from colossalai.auto_parallel.tensor_shard.utils import (
    enumerate_all_possible_1d_sharding,
    enumerate_all_possible_2d_sharding,
    ignore_sharding_exception,
)
from colossalai.tensor.sharding_spec import ShardingSpecException

from .strategy_generator import StrategyGenerator

__all__ = ['GetattrGenerator']


class GetattrGenerator(StrategyGenerator):
    """
    PlaceholderGenerator is a generic class to generate strategies for placeholder node.
    """

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        compute_cost = TrainCycleItem(fwd=10, bwd=10, total=20)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        '''
        Compute the memory cost per device with this specific strategy.
        '''
        forward_size_mapping = {'output': self._compute_size_in_bytes(strategy, "output")}

        # compute fwd cost incurred
        # fwd_cost = output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items()])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)

        bwd_mem_cost = MemoryCost(activation=0, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def enumerate_all_possible_output(self, mesh_dim_0, mesh_dim_1):
        # we check for the output logical shape to get the number of dimensions
        dim_partition_list = []
        dim_size = len(self.op_data['output'].logical_shape)

        # enumerate all the 2D sharding cases
        sharding_list_2d = enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, dim_size)
        dim_partition_list.extend(sharding_list_2d)

        # enumerate all the 1D sharding cases
        sharding_list_1d_on_dim_0 = enumerate_all_possible_1d_sharding(mesh_dim_0, dim_size)
        dim_partition_list.extend(sharding_list_1d_on_dim_0)
        sharding_list_1d_on_dim_1 = enumerate_all_possible_1d_sharding(mesh_dim_1, dim_size)
        dim_partition_list.extend(sharding_list_1d_on_dim_1)

        # add empty dict for fully replicated case
        dim_partition_list.append({})

        # sharding strategy bookkeeping
        strategy_list = []

        # convert these dim partition dict to sharding strategy
        for dim_partition_dict in dim_partition_list:
            dim_partition_dict_mapping = dict(output=dim_partition_dict)

            try:
                sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
                communication_action_mapping = {}

                # get name
                name = f"get_attr {sharding_spec_mapping['output'].sharding_sequence}"
                sharding_strategy = self.get_sharding_strategy(
                    name=name,
                    sharding_spec_mapping=sharding_spec_mapping,
                    communication_action_mapping=communication_action_mapping)
                strategy_list.append(sharding_strategy)
            except ShardingSpecException:
                continue

        return strategy_list

    def collate_strategies(self) -> List[ShardingStrategy]:
        return self.enumerate_all_possible_output(0, 1)
