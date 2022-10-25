from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (MemoryCost, ShardingStrategy, TrainCycleItem)

from .strategy_generator import OutputStrategyGenerator

__all__ = ['OutputGenerator']


class OutputGenerator(OutputStrategyGenerator):
    """
    OutputGenerator is a generic class to generate strategies for Output Node.
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
        fwd_mem_cost = MemoryCost(activation=0, parameter=0)

        bwd_mem_cost = MemoryCost(activation=0, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=0, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def collate_strategies(self) -> List[ShardingStrategy]:
        dim_partition_dict_mapping = {
            "output": {},
        }
        for index, _ in enumerate(self.predecessor_nodes):
            mapping_name = f"input_{index}"
            dim_partition_dict_mapping[mapping_name] = {}

        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = 'Replica Output'

        strategy = self.get_sharding_strategy(name=name,
                                              sharding_spec_mapping=sharding_spec_mapping,
                                              communication_action_mapping=communication_action_mapping)

        return [strategy]
