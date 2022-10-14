import operator
from functools import reduce
from ..sharding_strategy import ShardingStrategy, TrainCycleItem, MemoryCost
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from .strategy_generator import FollowingStrategyGenerator
from typing import List
from .._utils import exception_handler
import copy

__all__ = ['GetItemStrategyGenerator', 'TensorStrategyGenerator', 'TensorTupleStrategyGenerator']


class GetItemStrategyGenerator(FollowingStrategyGenerator):
    """
    GetItemStrategyGenerator is a generic class to generate strategies for operator.getitem.
    The operation data is defined as `output = input[other]`.

    There are mainly three use cases:
        1. args_0._meta_data: torch.Tensor, args_1._meta_data: int
        2. args_0._meta_data: torch.Tensor, args_1._meta_data: slice
        3. args_0._meta_data: Tuple[torch.Tensor], args_1._meta_data: int
    """

    @property
    def has_bias(self):
        return 'bias' in self.op_data

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


class TensorStrategyGenerator(GetItemStrategyGenerator):
    '''
    Deal with case 1 and 2.
    '''

    def generate(self):
        strategy_list = []
        for strategy in self.predecessor_node.strategies_vector:
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            dim_partition_dict_for_input = strategy.output_sharding_specs[self.op_data["input"]].dim_partition_dict
            dim_partition_dict_for_output = copy.deepcopy(dim_partition_dict_for_input)
            gather_input = 0 in dim_partition_dict_for_input
            if gather_input:
                logical_process_axis = dim_partition_dict_for_output.pop(0)

            shift_dim_partition_dict_for_output = {}
            for dim, mesh_dim_list in dim_partition_dict_for_output.items():
                shift_dim_partition_dict_for_output[dim - 1] = mesh_dim_list
            dim_partition_dict_for_output = shift_dim_partition_dict_for_output
            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
            if gather_input:
                input_communication_spec = self.get_communication_spec(
                    sharding_spec_mapping["input"],
                    communication_pattern=CollectiveCommPattern.GATHER_FWD_SPLIT_BWD,
                    logical_process_axis=logical_process_axis)
                communication_action_mapping["input"] = input_communication_spec

            name = f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["input"].sharding_sequence}'

            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)

            strategy_list.append(strategy)

        for strategy in strategy_list:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategy_list


class TensorTupleStrategyGenerator(GetItemStrategyGenerator):
    '''
    Deal with case 3.
    '''

    def generate(self):
        strategy_list = []
        index = self.op_data["index"].data

        for strategy in self.predecessor_node.strategies_vector:
            # the sharding spec for input in this case is a tuple of ShardingSpec.
            sharding_spec_for_input = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_output = sharding_spec_for_input[index].dim_partition_dict
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            dim_partition_dict_mapping = {
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
            sharding_spec_mapping["input"] = sharding_spec_for_input

            name = f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["input"].sharding_sequence}'

            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)

            strategy_list.append(strategy)

        for strategy in strategy_list:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategy_list
