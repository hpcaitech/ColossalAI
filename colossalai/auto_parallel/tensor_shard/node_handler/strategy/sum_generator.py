import copy
import operator
from functools import reduce
from typing import List

from colossalai.auto_parallel.tensor_shard.node_handler.strategy.strategy_generator import FollowingStrategyGenerator
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.auto_parallel.tensor_shard.utils import (
    check_keep_sharding_status,
    detect_reshape_mapping,
    infer_output_dim_partition_dict,
)
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from colossalai.tensor.sharding_spec import ShardingSpec

__all__ = ['SumGenerator']


class SumGenerator(FollowingStrategyGenerator):
    """
    SumGenerator deals with the sharding strategies of torch.sum op.
    """

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        sharded_output_shape = strategy.sharding_specs[self.op_data['output']].get_sharded_shape_per_device()
        input_size_product = reduce(operator.mul, sharded_input_shape)
        output_size_product = reduce(operator.mul, sharded_output_shape)

        compute_cost = TrainCycleItem(fwd=input_size_product,
                                      bwd=output_size_product,
                                      total=input_size_product + output_size_product)

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
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_input = copy.deepcopy(input_sharding_spec.dim_partition_dict)
            sum_dims, sum_mapping_dict = self.op_data['sum_info'].data

            # TODO: a better way to handle the distributed sum is sum all the data on chip and then do all reduce
            # among all the shard groups
            recover_dims = []
            dim_partition_dict_for_output = {}
            for dim in dim_partition_dict_for_input:
                if dim in sum_dims:
                    recover_dims.append(dim)
                elif dim in sum_mapping_dict:
                    dim_partition_dict_for_output[sum_mapping_dict[dim]] = dim_partition_dict_for_input[dim]
                else:
                    raise RuntimeError(f'dim {dim} is not in sum_mapping_dict or sum_dims')

            for dim in recover_dims:
                dim_partition_dict_for_input.pop(dim)

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
