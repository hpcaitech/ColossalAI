import copy
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (MemoryCost, ShardingStrategy, TrainCycleItem)

from .strategy_generator import FollowingStrategyGenerator

__all__ = ['UnaryElementwiseGenerator']


class UnaryElementwiseGenerator(FollowingStrategyGenerator):
    """
    UnaryElementwiseGenerator which deals with the sharding strategies of UnaryElementwiseOp.
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
        forward_size_mapping = {
            'input': self._compute_size_in_bytes(strategy, "input"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items() if not self.is_param(k)])
        fwd_parameter_cost = sum([v for k, v in forward_size_mapping.items() if self.is_param(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items() if not self.is_param(k)])
        bwd_parameter_cost = sum([v for k, v in backward_size_mapping.items() if self.is_param(k)])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=bwd_parameter_cost)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost,
                                    parameter=fwd_parameter_cost + bwd_parameter_cost)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        # For element-wise function, we keep the sharding spec of output node same as
        # the input. Therefore, the different strategies of input node with same
        # output sharding spec will generate same strategy for element-wise function.
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_input = input_sharding_spec.dim_partition_dict
            dim_partition_dict_for_output = copy.deepcopy(dim_partition_dict_for_input)
            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{sharding_spec_mapping["input"].sharding_sequence} -> {sharding_spec_mapping["output"].sharding_sequence}_{index}'
            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)
            strategy_list.append(strategy)

        return strategy_list
