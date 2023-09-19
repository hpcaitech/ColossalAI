import copy
import operator
from functools import reduce
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.auto_parallel.tensor_shard.utils import ignore_sharding_exception
from colossalai.tensor.shape_consistency import CollectiveCommPattern

from .strategy_generator import StrategyGenerator

__all__ = ["BatchNormStrategyGenerator"]


class BatchNormStrategyGenerator(StrategyGenerator):
    """
    A StrategyGenerator which deals with the sharding strategies of batch normalization.

    To keep the math consistency, there are two way to do BatchNorm if the input
    shards on batch dimension:
    1. We gather the input partitions through batch dimension, then do the normal BatchNorm.
    2. We do the SyncBatchNorm on the each input partition separately, the SyncBN op will help
       us to keep the computing correctness.
    In this generator, both methods will be considered.
    """

    def validate(self) -> bool:
        """
        In sanity check, we need make sure the input data having correct dimension size.
        For BatchNorm1d, the dim of input data should be 3([N, C, L]).
        For BatchNorm2d, the dim of input data should be 4([N, C, H, W]).
        For BatchNorm3d, the dim of input data should be 5([N, C, H, W, D]).
        """
        input_op_data = self.op_data["input"]
        assert input_op_data.data.dim() in (
            3,
            4,
            5,
        ), f"We suppose the dim of input fed into conv op should in range of [3, 5]."

    def update_compute_cost(self, strategy: ShardingStrategy):
        """
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be divided by TFLOPS, now it just shows the computation size.
        """
        # TODO: a constant coefficient need to be added.
        # 1D: (L) * N * Cin
        # 2D: (H * W) * N  * Cin
        # 3D: (H * W  * D) * N  * Cin
        sharded_input_shape = strategy.sharding_specs[self.op_data["input"]].get_sharded_shape_per_device()
        sharded_output_shape = strategy.sharding_specs[self.op_data["output"]].get_sharded_shape_per_device()
        if self.has_bias:
            # bias add is an element wise operation, so the cost is equal to product of output shape.
            bias_compute_cost = reduce(operator.mul, sharded_output_shape)
        input_product = reduce(operator.mul, sharded_input_shape, 1)
        forward_compute_cost = input_product
        backward_activation_compute_cost = input_product
        backward_weight_compute_cost = input_product
        backward_compute_cost = backward_weight_compute_cost + backward_activation_compute_cost
        if self.has_bias:
            forward_compute_cost += bias_compute_cost
            backward_compute_cost += bias_compute_cost
        total_compute_cost = forward_compute_cost + backward_compute_cost
        compute_cost = TrainCycleItem(fwd=forward_compute_cost, bwd=backward_compute_cost, total=total_compute_cost)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        forward_size_mapping = {
            "input": self._compute_size_in_bytes(strategy, "input"),
            "other": self._compute_size_in_bytes(strategy, "other"),
            "output": self._compute_size_in_bytes(strategy, "output"),
            "running_mean": self._compute_size_in_bytes(strategy, "running_mean"),
            "running_var": self._compute_size_in_bytes(strategy, "running_var"),
        }

        if self.has_bias:
            bias_size = self._compute_size_in_bytes(strategy, "bias")
            forward_size_mapping["bias"] = bias_size

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + other + bias + output
        fwd_activation_cost = sum(
            [v for k, v in forward_size_mapping.items() if not self.is_param(k) and not self.is_buffer(k)]
        )
        fwd_parameter_cost = sum([v for k, v in forward_size_mapping.items() if self.is_param(k)])
        fwd_buffer_cost = sum([v for k, v in forward_size_mapping.items() if self.is_buffer(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost, buffer=fwd_buffer_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad + other_grad + bias_grad
        bwd_activation_cost = sum(
            [v for k, v in backward_size_mapping.items() if not self.is_param(k) and not self.is_buffer(k)]
        )
        bwd_parameter_cost = sum([v for k, v in backward_size_mapping.items() if self.is_param(k)])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=bwd_parameter_cost)

        # compute total cost
        total_mem_cost = MemoryCost(
            activation=fwd_activation_cost + bwd_activation_cost,
            parameter=fwd_parameter_cost + bwd_parameter_cost,
            buffer=fwd_buffer_cost,
        )
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def split_input_channel(self, mesh_dim_0):
        name = f"RS{mesh_dim_0} = RS{mesh_dim_0} x S{mesh_dim_0}"
        dim_partition_dict_mapping = {
            "input": {1: [mesh_dim_0]},
            "other": {0: [mesh_dim_0]},
            "output": {1: [mesh_dim_0]},
            "running_mean": {0: [mesh_dim_0]},
            "running_var": {0: [mesh_dim_0]},
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {0: [mesh_dim_0]}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    @ignore_sharding_exception
    def split_input_channel_1d(self, mesh_dim_0, mesh_dim_1):
        name = f"RS{mesh_dim_0}{mesh_dim_1} = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}"
        dim_partition_dict_mapping = {
            "input": {1: [mesh_dim_0, mesh_dim_1]},
            "other": {0: [mesh_dim_0, mesh_dim_1]},
            "output": {1: [mesh_dim_0, mesh_dim_1]},
            "running_mean": {0: [mesh_dim_0, mesh_dim_1]},
            "running_var": {0: [mesh_dim_0, mesh_dim_1]},
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {0: [mesh_dim_0, mesh_dim_1]}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    @ignore_sharding_exception
    def non_split(self):
        name = f"RR = RR x R"
        dim_partition_dict_mapping = {
            "input": {},
            "other": {},
            "output": {},
            "running_mean": {},
            "running_var": {},
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    @ignore_sharding_exception
    def split_input_batch(self, mesh_dim_0):
        name = f"S{mesh_dim_0}R = S{mesh_dim_0}R x R WITH SYNC_BN"
        dim_partition_dict_mapping = {
            "input": {0: [mesh_dim_0]},
            "other": {},
            "output": {0: [mesh_dim_0]},
            "running_mean": {},
            "running_var": {},
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        # For SyncBN case, we don't need to do communication for weight and bias.
        # TODO: the communication happens internally at SyncBN operation. We need to replace the BN operation
        # to SyncBN operation instead of inserting a communication node.
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_0,
            comm_type=CommType.IMPLICIT,
        )

        # TODO: Temporary solution has no communication cost,
        # above action should be added after the SyncBN replace pass completed.
        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    @ignore_sharding_exception
    def split_input_batch_1d(self, mesh_dim_0, mesh_dim_1):
        name = f"S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x R WITH SYNC_BN"
        dim_partition_dict_mapping = {
            "input": {0: [mesh_dim_0, mesh_dim_1]},
            "other": {},
            "output": {0: [mesh_dim_0, mesh_dim_1]},
            "running_mean": {},
            "running_var": {},
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        # For SyncBN case, we don't need to do communication for gradients of weight and bias.
        # TODO: the communication happens internally at SyncBN operation. We need to replace the BN operation
        # to SyncBN operation instead of inserting a communication node.
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1],
            comm_type=CommType.IMPLICIT,
        )

        # TODO: Temporary solution has no communication cost,
        # above action should be added after the SyncBN replace pass completed.
        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    @ignore_sharding_exception
    def split_input_both_dim(self, mesh_dim_0, mesh_dim_1):
        name = f"S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1} WITH SYNC_BN"
        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0],
                1: [mesh_dim_1],
            },
            "other": {
                0: [mesh_dim_1],
            },
            "output": {
                0: [mesh_dim_0],
                1: [mesh_dim_1],
            },
            "running_mean": {
                0: [mesh_dim_1],
            },
            "running_var": {
                0: [mesh_dim_1],
            },
            "num_batches_tracked": {},
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {
                0: [mesh_dim_1],
            }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        # For SyncBN case, we don't need to do communication for gradients of weight and bias.
        # TODO: the communication happens internally at SyncBN operation. We need to replace the BN operation
        # to SyncBN operation instead of inserting a communication node.
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=[mesh_dim_0],
            comm_type=CommType.IMPLICIT,
        )

        # TODO: Temporary solution has no communication cost,
        # above action should be added after the SyncBN replace pass completed.
        communication_action_mapping = {}

        return self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

    def collate_strategies(self) -> List[ShardingStrategy]:
        """
        Generate every possible strategies for a BatchNorm node, and record all strategies into the strategies_vector.
        """

        strategy_list = []
        # RS = RS x S
        strategy_list.append(self.split_input_channel(0))
        strategy_list.append(self.split_input_channel(1))

        # RR = RR x R
        strategy_list.append(self.non_split())

        # RS01 = RS01 x S01
        strategy_list.append(self.split_input_channel_1d(0, 1))

        # The strategies with SYNC_BN are temporarily commented,
        # because it requires some additional passes to keep runtime
        # computation correctness.

        # TODO: The strategies below should be uncommented after runtime
        # passes ready.
        # SR = SR x R WITH SYNC_BN
        strategy_list.append(self.split_input_batch(0))
        strategy_list.append(self.split_input_batch(1))

        # SS = SS x S WITH SYNC_BN
        strategy_list.append(self.split_input_both_dim(0, 1))
        strategy_list.append(self.split_input_both_dim(1, 0))

        # S01R = S01R x R WITH SYNC_BN
        strategy_list.append(self.split_input_batch_1d(0, 1))

        return strategy_list
