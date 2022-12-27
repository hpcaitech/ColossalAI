import copy
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.logging import get_dist_logger
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from colossalai.tensor.sharding_spec import ShardingSpecException

from .strategy_generator import FollowingStrategyGenerator

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

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        getitem_index = self.op_data['index'].data
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            try:
                logger = get_dist_logger()
                dim_partition_dict_mapping = {}
                communication_action_mapping = {}
                dim_partition_dict_for_input = copy.deepcopy(
                    strategy.output_sharding_specs[self.op_data["input"]].dim_partition_dict)

                int_index = False
                if isinstance(getitem_index, int):
                    int_index = True
                    getitem_dims = [
                        0,
                    ]
                    shift_length = 1
                elif isinstance(getitem_index, slice):
                    getitem_dims = [
                        0,
                    ]
                else:
                    getitem_dims = [i for i in range(len(getitem_index))]
                    if isinstance(getitem_index[0], int):
                        int_index = True
                        shift_length = len(getitem_index)

                gather_dims = []
                for dim in getitem_dims:
                    if dim in dim_partition_dict_for_input:
                        gather_dims.append(dim)

                for dim in gather_dims:
                    dim_partition_dict_for_input.pop(dim)
                dim_partition_dict_for_output = copy.deepcopy(dim_partition_dict_for_input)

                if int_index:
                    shift_dim_partition_dict_for_output = {}
                    for dim, mesh_dim_list in dim_partition_dict_for_output.items():
                        shift_dim_partition_dict_for_output[dim - shift_length] = mesh_dim_list
                    dim_partition_dict_for_output = shift_dim_partition_dict_for_output

                dim_partition_dict_mapping = {
                    "input": dim_partition_dict_for_input,
                    "output": dim_partition_dict_for_output,
                }
                sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

                name = f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["input"].sharding_sequence}_{index}'

                strategy = self.get_sharding_strategy(name=name,
                                                      sharding_spec_mapping=sharding_spec_mapping,
                                                      communication_action_mapping=communication_action_mapping)
            except ShardingSpecException as e:
                logger.debug(e)
                continue
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

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        index = self.op_data["index"].data

        for strategy_index, strategy in enumerate(self.predecessor_node.strategies_vector):
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
            input_sharding_info = f"get the {index} element from ("
            for sharding_spec in sharding_spec_for_input:
                input_sharding_info += f'{sharding_spec.sharding_sequence}, '
            input_sharding_info += ")"
            name = f'{sharding_spec_mapping["output"].sharding_sequence} = {input_sharding_info}_{strategy_index}'

            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)

            strategy_list.append(strategy)

        return strategy_list
