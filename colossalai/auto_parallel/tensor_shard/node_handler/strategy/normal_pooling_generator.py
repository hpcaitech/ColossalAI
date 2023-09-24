import copy
import operator
from functools import reduce
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, ShardingStrategy, TrainCycleItem
from colossalai.auto_parallel.tensor_shard.utils import (
    enumerate_all_possible_1d_sharding,
    enumerate_all_possible_2d_sharding,
    ignore_sharding_exception,
)

from .strategy_generator import StrategyGenerator


class NormalPoolStrategyGenerator(StrategyGenerator):
    """
    NormalPoolStrategyGenerator is a generic class to generate strategies for pool operation like MaxPoolxd.
    The reason we call this normal pool is AvgPoolxd and MaxPoolxd are taking the kernel size element from image,
    and reduce them depending on the operation type.
    """

    def validate(self) -> bool:
        """
        In sanity check, we need make sure the input data having correct dimension size.
        For Pool1d, the dim of input data should be 3([N, C, L]).
        For Pool2d, the dim of input data should be 4([N, C, H, W]).
        For Pool3d, the dim of input data should be 5([N, C, H, W, D]).
        """
        input_op_data = self.op_data["input"]
        assert input_op_data.data.dim() in (
            3,
            4,
            5,
        ), f"We suppose the dim of input fed into Pool op should in range of [3, 5]."

    def update_compute_cost(self, strategy: ShardingStrategy) -> TrainCycleItem:
        """
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be divided by TFLOPS, now it just shows the computation size.
        """
        # TODO: compute_cost need to be divided by TFLOPS, now it just shows the computation size.
        # 1D: (Lout) * N * C * kernel
        # 2D: (H * W) * N * Cout * Cin * kernel
        # 3D: (H * W  * D) * N * Cout * Cin * kernel
        sharded_output_shape = strategy.sharding_specs[self.op_data["output"]].get_sharded_shape_per_device()
        sharded_input_shape = strategy.sharding_specs[self.op_data["input"]].get_sharded_shape_per_device()

        kernel_size = self.op_data["other"].data
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(sharded_output_shape) - 2)
        kernel_size_product = reduce(operator.mul, kernel_size)
        output_size_product = reduce(operator.mul, sharded_output_shape)
        input_size_product = reduce(operator.mul, sharded_input_shape)

        forward_compute_cost = output_size_product * kernel_size_product
        backward_compute_cost = input_size_product * kernel_size_product

        total_compute_cost = forward_compute_cost + backward_compute_cost

        compute_cost = TrainCycleItem(fwd=forward_compute_cost, bwd=backward_compute_cost, total=total_compute_cost)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        forward_size_mapping = {
            "input": self._compute_size_in_bytes(strategy, "input"),
            "output": self._compute_size_in_bytes(strategy, "output"),
        }

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items()])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)

        # compute bwd cost incurred
        # bwd_cost = input_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items()])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def _generate_strategy_with_dim_partition(self, dim_partition):
        dim_partition_dict_mapping = {"input": dim_partition, "output": dim_partition}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = (
            f'{sharding_spec_mapping["output"].sharding_sequence} = {sharding_spec_mapping["input"].sharding_sequence}'
        )
        communication_action_mapping = {}

        strategy = self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

        return strategy

    def enumerate_all_possible_batch_dimensions_dim_partition(self, mesh_dim_0, mesh_dim_1):
        dim_partition_list = []
        dim_partition_list.extend(enumerate_all_possible_1d_sharding(mesh_dim_0, 2))
        dim_partition_list.extend(enumerate_all_possible_1d_sharding(mesh_dim_1, 2))
        dim_partition_list.extend(enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, 2))
        # append {} for non_split case
        dim_partition_list.append({})

        return dim_partition_list

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []

        dim_partition_list = self.enumerate_all_possible_batch_dimensions_dim_partition(0, 1)
        for dim_partition in dim_partition_list:
            strategy = self._generate_strategy_with_dim_partition(dim_partition)
            strategy_list.append(strategy)

        return strategy_list
