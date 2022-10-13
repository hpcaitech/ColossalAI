import operator
from functools import reduce
from ..sharding_strategy import ShardingStrategy_V2, TrainCycleItem, MemoryCost
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from .strategy_generator import StrategyGenerator_V2
from typing import List
from .._utils import exception_handler
import copy

__all__ = ['PlaceholderGenerator']


class PlaceholderGenerator(StrategyGenerator_V2):
    """
    PlaceholderGenerator is a generic class to generate strategies for placeholder node.
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

    def generate(self):
        dim_partition_dict_mapping = {
            "output": {},
        }
        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = f'Replica Placeholder'

        strategy = self.get_sharding_strategy(name=name,
                                              sharding_spec_mapping=sharding_spec_mapping,
                                              communication_action_mapping=communication_action_mapping)

        self.update_communication_cost(strategy)
        self.update_compute_cost(strategy)
        self.update_memory_cost(strategy)

        return [strategy]
