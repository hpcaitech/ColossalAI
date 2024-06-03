import operator
from functools import reduce
from typing import List

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, ShardingStrategy, TrainCycleItem
from colossalai.auto_parallel.tensor_shard.utils import (
    enumerate_all_possible_1d_sharding,
    enumerate_all_possible_2d_sharding,
    ignore_sharding_exception,
)
from colossalai.tensor.sharding_spec import ShardingSpecException

from .strategy_generator import StrategyGenerator

__all__ = ["BinaryElementwiseStrategyGenerator"]


class BinaryElementwiseStrategyGenerator(StrategyGenerator):
    """
    An BinaryElementwiseStrategyGenerator is a node handler which deals with elementwise operations
    which have two operands and broadcasting occurs such as torch.add.

    The logical shape for this operation will be `input <op> other`.
    """

    def validate(self) -> bool:
        assert (
            len(self.op_data) == 3
        ), f"BinaryElementwiseStrategyGenerator only accepts three operation data (input, other and output), but got {len(self.op_data)}"
        for name, op_data in self.op_data.items():
            if not isinstance(op_data.data, (torch.Tensor, int, float)):
                raise TypeError(f"The operation data {name} is not a torch.Tensor/int/float.")

    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        shape = strategy.sharding_specs[self.op_data["input"]].get_sharded_shape_per_device()

        # since elementwise ops are not compute-intensive,
        # we approximate the backward compute cost
        # to be twice the fwd compute cost
        fwd_compute_cost = reduce(operator.mul, shape)
        bwd_compute_cost = fwd_compute_cost * 2
        compute_cost = TrainCycleItem(
            fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost
        )
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        # all input, output and outputs have the same shape
        strategy.sharding_specs[self.op_data["input"]].get_sharded_shape_per_device()

        # compute fwd memory cost in bytes
        # as the elementwise ops are not memory-intensive
        # we approximate the fwd memory cost to be the output
        # and the backward memory cost to be grad of input and other
        input_bytes = self._compute_size_in_bytes(strategy, "input")
        other_bytes = self._compute_size_in_bytes(strategy, "other")
        output_bytes = self._compute_size_in_bytes(strategy, "output")
        fwd_memory_cost = MemoryCost(activation=output_bytes)
        bwd_memory_cost = MemoryCost(activation=input_bytes + other_bytes)
        total_memory_cost = MemoryCost(activation=input_bytes + other_bytes + output_bytes)
        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_memory_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def enumerate_all_possible_output(self, mesh_dim_0, mesh_dim_1):
        # we check for the output logical shape to get the number of dimensions
        dim_partition_list = []
        dim_size = len(self.op_data["output"].logical_shape)

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
            dim_partition_dict_mapping = dict(
                input=dim_partition_dict, other=dim_partition_dict, output=dim_partition_dict
            )

            try:
                sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
                communication_action_mapping = {}

                # get name
                sharding_seq = sharding_spec_mapping["input"].sharding_sequence
                name = f"{sharding_seq} = {sharding_seq} <binary-elementwise-op> {sharding_seq}"
                sharding_strategy = self.get_sharding_strategy(
                    name=name,
                    sharding_spec_mapping=sharding_spec_mapping,
                    communication_action_mapping=communication_action_mapping,
                )
                strategy_list.append(sharding_strategy)
            except ShardingSpecException:
                continue
        return strategy_list

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = self.enumerate_all_possible_output(0, 1)
        return strategy_list
