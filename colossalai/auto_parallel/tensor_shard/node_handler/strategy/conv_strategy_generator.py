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


class ConvStrategyGenerator(StrategyGenerator):
    """
    ConvStrategyGenerator is a generic class to generate strategies.
    The operation data is defined as `output = input x other + bias`.
    """

    def validate(self) -> bool:
        '''
        In sanity check, we need make sure the input data having correct dimension size.
        For Conv1d, the dim of input data should be 3([N, C, L]).
        For Conv2d, the dim of input data should be 4([N, C, H, W]).
        For Conv3d, the dim of input data should be 5([N, C, H, W, D]).
        '''
        input_op_data = self.op_data['input']
        assert input_op_data.data.dim() in (
            3, 4, 5), f'We suppose the dim of input fed into conv op should in range of [3, 5].'

    def update_compute_cost(self, strategy: ShardingStrategy):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        '''
        # TODO: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        # 1D: (L) * N * Cout * Cin * kernel
        # 2D: (H * W) * N * Cout * Cin * kernel
        # 3D: (H * W  * D) * N * Cout * Cin * kernel
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        sharded_other_shape = strategy.sharding_specs[self.op_data['other']].get_sharded_shape_per_device()
        sharded_output_shape = strategy.sharding_specs[self.op_data['output']].get_sharded_shape_per_device()
        if self.has_bias:
            # bias add is an element wise operation, so the cost is equal to product of output shape.
            bias_compute_cost = reduce(operator.mul, sharded_output_shape)

        output_size = sharded_output_shape[2:]
        output_size_product = reduce(operator.mul, output_size)
        input_size = sharded_input_shape[2:]
        input_size_product = reduce(operator.mul, input_size, 1)
        kernel_size = sharded_other_shape[2:]
        kernel_size_product = reduce(operator.mul, kernel_size, 1)
        batch_size = sharded_input_shape[0]
        channel_in = sharded_input_shape[1]
        channel_out = sharded_other_shape[1]

        forward_compute_cost = output_size_product * batch_size * channel_in * channel_out * kernel_size_product

        backward_activation_cost = input_size_product * batch_size * channel_in * channel_out * kernel_size_product
        backward_weight_cost = output_size_product * batch_size * channel_in * channel_out * kernel_size_product
        backward_compute_cost = backward_weight_cost + backward_activation_cost
        if self.has_bias:
            forward_compute_cost += bias_compute_cost
            backward_compute_cost += bias_compute_cost
        total_compute_cost = forward_compute_cost + backward_compute_cost

        compute_cost = TrainCycleItem(fwd=forward_compute_cost, bwd=backward_compute_cost, total=total_compute_cost)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        forward_size_mapping = {
            'input': self._compute_size_in_bytes(strategy, "input"),
            'other': self._compute_size_in_bytes(strategy, "other"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        if self.has_bias:
            bias_size = self._compute_size_in_bytes(strategy, "bias")
            forward_size_mapping['bias'] = bias_size

        backward_size_mapping = copy.deepcopy(forward_size_mapping)
        backward_size_mapping.pop("output")
        # compute fwd cost incurred
        # fwd_cost = input + other + bias + output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items() if not self.is_param(k)])
        fwd_parameter_cost = sum([v for k, v in forward_size_mapping.items() if self.is_param(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad + other_grad + bias_grad
        bwd_activation_cost = sum([v for k, v in backward_size_mapping.items() if not self.is_param(k)])
        bwd_parameter_cost = sum([v for k, v in backward_size_mapping.items() if self.is_param(k)])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=bwd_parameter_cost)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost,
                                    parameter=fwd_parameter_cost + bwd_parameter_cost)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    @ignore_sharding_exception
    def split_input_batch_weight_out_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                1: [mesh_dim_1]
            },
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {0: [mesh_dim_1]}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["input"],
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

        if self.has_bias:
            if self.is_param('bias'):
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.BEFORE,
                    key_for_kwarg='bias')
            communication_action_mapping["bias"] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_input_batch(self, mesh_dim_0):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}R x RR'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {},
            "output": {
                0: [mesh_dim_0],
            },
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

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

        if self.has_bias:
            if self.is_param('bias'):
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.BEFORE,
                    key_for_kwarg='bias')
            communication_action_mapping["bias"] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_input_both_dim_weight_in_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0],
                1: [mesh_dim_1],
            },
            "other": {
                0: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
            },
        }
        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        output_comm_action = self.get_communication_action(
            sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.AFTER)

        communication_action_mapping = {"output": output_comm_action}

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
        if self.has_bias:
            if self.is_param("bias"):
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim_0,
                    comm_type=CommType.BEFORE,
                    key_for_kwarg='bias')
            communication_action_mapping["bias"] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_input_in_channel_weight_both_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_mapping = {
            "input": {
                1: [mesh_dim_0],
            },
            "other": {
                0: [mesh_dim_0],
                1: [mesh_dim_1],
            },
            "output": {
                1: [mesh_dim_1],
            },
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {
                0: [mesh_dim_1],
            }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        output_comm_action = self.get_communication_action(
            sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_0,
            comm_type=CommType.AFTER)
        input_comm_action = self.get_communication_action(
            sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=0)

        communication_action_mapping = {"output": output_comm_action, "input": input_comm_action}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_input_in_channel_weight_in_channel(self, mesh_dim_0):
        name = f'RR = RS{mesh_dim_0} x S{mesh_dim_0}R'

        dim_partition_dict_mapping = {
            "input": {
                1: [mesh_dim_0],
            },
            "other": {
                0: [mesh_dim_0],
            },
            "output": {},
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        output_comm_action = self.get_communication_action(
            sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_0,
            comm_type=CommType.AFTER)

        communication_action_mapping = {"output": output_comm_action}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_weight_out_channel(self, mesh_dim_0):
        name = f'RS{mesh_dim_0} = RR x RS{mesh_dim_0}'

        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                1: [mesh_dim_0],
            },
            "output": {
                1: [mesh_dim_0],
            },
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {
                0: [mesh_dim_0],
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
    def non_split(self):
        name = f'RR = RR x RR'

        dim_partition_dict_mapping = {
            "input": {},
            "other": {},
            "output": {},
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping={})

    @ignore_sharding_exception
    def split_1d_parallel_on_input_batch(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x RR'

        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1],
            },
            "other": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1],
            },
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

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

        if self.has_bias:
            if self.is_param("bias"):
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=[mesh_dim_0, mesh_dim_1],
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec_mapping["bias"],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=[mesh_dim_0, mesh_dim_1],
                    comm_type=CommType.BEFORE,
                    key_for_kwarg='bias')
            communication_action_mapping["bias"] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_1d_parallel_on_in_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'
        dim_partition_dict_mapping = {
            "input": {
                1: [mesh_dim_0, mesh_dim_1],
            },
            "other": {
                0: [mesh_dim_0, mesh_dim_1],
            },
            "output": {},
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {}

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        output_comm_action = self.get_communication_action(
            sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1],
            comm_type=CommType.AFTER)

        communication_action_mapping = {"output": output_comm_action}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_1d_parallel_on_out_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RR x RS{mesh_dim_0}{mesh_dim_1}'
        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                1: [mesh_dim_0, mesh_dim_1],
            },
            "output": {
                1: [mesh_dim_0, mesh_dim_1],
            },
        }

        if self.has_bias:
            dim_partition_dict_mapping["bias"] = {
                0: [mesh_dim_0, mesh_dim_1],
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
        # SS = SR x RS
        strategies.append(self.split_input_batch_weight_out_channel(0, 1))
        strategies.append(self.split_input_batch_weight_out_channel(1, 0))

        # SR = SR x RR
        strategies.append(self.split_input_batch(0))
        strategies.append(self.split_input_batch(1))

        # SR = SS x SR
        strategies.append(self.split_input_both_dim_weight_in_channel(0, 1))
        strategies.append(self.split_input_both_dim_weight_in_channel(1, 0))

        # RS = RS x SS
        strategies.append(self.split_input_in_channel_weight_both_channel(0, 1))
        strategies.append(self.split_input_in_channel_weight_both_channel(1, 0))

        # RR = RS x SR
        strategies.append(self.split_input_in_channel_weight_in_channel(0))
        strategies.append(self.split_input_in_channel_weight_in_channel(1))

        # RS = RR x RS
        strategies.append(self.split_weight_out_channel(0))
        strategies.append(self.split_weight_out_channel(1))

        # RR= RR x RR
        strategies.append(self.non_split())

        # S01R = S01R x RR
        strategies.append(self.split_1d_parallel_on_input_batch(0, 1))

        # RR = RS01 x S01R
        strategies.append(self.split_1d_parallel_on_in_channel(0, 1))

        # RS01 = RR x RS01
        strategies.append(self.split_1d_parallel_on_out_channel(0, 1))

        return strategies
