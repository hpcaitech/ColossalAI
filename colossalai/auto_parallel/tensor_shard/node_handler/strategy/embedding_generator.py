import copy
import operator
import warnings
from functools import reduce
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.auto_parallel.tensor_shard.utils import ignore_sharding_exception
from colossalai.tensor.shape_consistency import CollectiveCommPattern

from .strategy_generator import StrategyGenerator


class EmbeddingStrategyGenerator(StrategyGenerator):
    """
    EmbeddingStrategyGenerator is a generic class to generate strategies for nn.Embedding or F.embedding.
    The operation data is defined as `output = input x other`.
    """

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: The computation cost for the embedding handler is estimated as dense computing now.
              It may not be accurate.
        '''
        # TODO: estimate the embedding computation cost as sparse operation
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        sharded_other_shape = strategy.sharding_specs[self.op_data['other']].get_sharded_shape_per_device()
        sharded_output_shape = strategy.sharding_specs[self.op_data['output']].get_sharded_shape_per_device()

        input_size_product = reduce(operator.mul, sharded_input_shape)
        other_size_product = reduce(operator.mul, sharded_other_shape)
        output_size_product = reduce(operator.mul, sharded_output_shape)

        forward_compute_cost = input_size_product * other_size_product

        backward_activation_cost = other_size_product * output_size_product / sharded_output_shape[-1]
        backward_weight_cost = input_size_product * other_size_product
        backward_compute_cost = backward_weight_cost + backward_activation_cost

        total_compute_cost = forward_compute_cost + backward_compute_cost

        compute_cost = TrainCycleItem(fwd=forward_compute_cost, bwd=backward_compute_cost, total=total_compute_cost)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        forward_size_mapping = {
            'input': self._compute_size_in_bytes(strategy, "input"),
            'other': self._compute_size_in_bytes(strategy, "other"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + other + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items() if not self.is_param(k)])
        fwd_parameter_cost = sum([v for k, v in forward_size_mapping.items() if self.is_param(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad + other_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items() if not self.is_param(k)])
        bwd_parameter_cost = sum([v for k, v in backward_size_mapping.items() if self.is_param(k)])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=bwd_parameter_cost)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost,
                                    parameter=fwd_parameter_cost + bwd_parameter_cost)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def non_split(self):
        name = f'RR = R x RR'

        dim_partition_dict_mapping = {
            "input": {},
            "other": {},
            "output": {},
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping={})

    @ignore_sharding_exception
    def split_input(self, mesh_dim_0):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0} x RR'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {},
            "output": {
                0: [mesh_dim_0],
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        communication_action_mapping = {}
        if self.is_param("other"):
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.HOOK)

        else:
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.BEFORE,
                arg_index=1)

        communication_action_mapping["other"] = other_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_input_and_embedding_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0} x RS{mesh_dim_1}'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0],
            },
            "other": {
                1: [mesh_dim_1],
            },
            "output": {
                0: [mesh_dim_0],
                1: [mesh_dim_1],
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        input_comm_action = self.get_communication_action(
            sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=0)
        communication_action_mapping = {"input": input_comm_action}

        if self.is_param("other"):
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.HOOK)

        else:
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.BEFORE,
                arg_index=1)

        communication_action_mapping["other"] = other_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_1d_parallel_on_input(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1} x RR'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "other": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1],
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        communication_action_mapping = {}

        if self.is_param("other"):
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.HOOK)

        else:
            other_comm_action = self.get_communication_action(
                sharding_spec_mapping["other"],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.BEFORE,
                arg_index=1)

        communication_action_mapping["other"] = other_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_embedding_dim(self, mesh_dim_0):
        name = f'RS{mesh_dim_0} = R x RS{mesh_dim_0}'

        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                1: [mesh_dim_0],
            },
            "output": {
                1: [mesh_dim_0],
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        input_comm_action = self.get_communication_action(
            sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_0,
            comm_type=CommType.BEFORE,
            arg_index=0)

        communication_action_mapping = {"input": input_comm_action}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_1d_parallel_on_embedding_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = R x RS{mesh_dim_0}{mesh_dim_1}'

        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                1: [mesh_dim_0, mesh_dim_1],
            },
            "output": {
                1: [mesh_dim_0, mesh_dim_1],
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        input_comm_action = self.get_communication_action(
            sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1],
            comm_type=CommType.BEFORE,
            arg_index=0)

        communication_action_mapping = {"input": input_comm_action}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategies = []

        # RR= R x RR
        strategies.append(self.non_split())

        # SR = S x RR
        strategies.append(self.split_input(0))
        strategies.append(self.split_input(1))

        # SS = S x RS
        strategies.append(self.split_input_and_embedding_dim(0, 1))
        strategies.append(self.split_input_and_embedding_dim(1, 0))

        # S01R = S01 x RR
        strategies.append(self.split_1d_parallel_on_input(0, 1))

        # RS = R x RS
        strategies.append(self.split_embedding_dim(0))
        strategies.append(self.split_embedding_dim(1))

        # RS01 = R x RS01
        strategies.append(self.split_1d_parallel_on_embedding_dim(0, 1))

        return strategies
