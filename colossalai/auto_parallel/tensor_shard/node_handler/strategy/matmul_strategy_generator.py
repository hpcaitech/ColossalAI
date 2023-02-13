import operator
from ast import arg
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


class MatMulStrategyGenerator(StrategyGenerator):
    """
    MatMulStrategyGenerator is a generic class to cover all matrix multiplication cases.
    The operation data is defined as `output = input x other + bias`.
    """

    def update_memory_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        size_mapping = {
            'input': self._compute_size_in_bytes(strategy, "input"),
            'other': self._compute_size_in_bytes(strategy, "other"),
            'output': self._compute_size_in_bytes(strategy, "output")
        }

        if self.has_bias:
            bias_size = self._compute_size_in_bytes(strategy, "bias")
            size_mapping['bias'] = bias_size

        # compute fwd cost incurred
        # fwd_cost = input + other + bias + output
        fwd_activation_cost = sum([v for k, v in size_mapping.items() if not self.is_param(k)])
        fwd_parameter_cost = sum([v for k, v in size_mapping.items() if self.is_param(k)])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=fwd_parameter_cost)

        # compute bwd cost incurred
        # bwd_cost = input_grad + bias_grad
        bwd_activation_cost = sum([v for k, v in size_mapping.items() if k in ['input', 'other', 'bias']])
        bwd_mem_cost = MemoryCost(activation=bwd_activation_cost, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost + bwd_activation_cost,
                                    parameter=fwd_parameter_cost + 0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost


class DotProductStrategyGenerator(MatMulStrategyGenerator):

    def validate(self) -> bool:
        input_op_data = self.op_data['input']
        other_op_data = self.op_data['other']
        assert input_op_data.data.dim() == 1 and other_op_data.data.dim() == 1

    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        fwd_compute_cost = sharded_input_shape[0]
        bwd_compute_cost = fwd_compute_cost * 2
        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)
        return compute_cost

    @ignore_sharding_exception
    def no_split(self):
        name = f'R = R dot R'
        dim_partition_dict = {"input": {}, "other": {}, "output": {}, 'bias': {}}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)
        communication_action_mapping = {}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_one_dim(self, mesh_dim):
        name = f'R = S{mesh_dim} dot S{mesh_dim}'

        # get sharding spec
        dim_partition_dict = {"input": {0: [mesh_dim]}, "other": {0: [mesh_dim]}, "output": {}, "bias": {0: [mesh_dim]}}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication action
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim,
            comm_type=CommType.AFTER)
        communication_action_mapping = {"output": output_comm_action}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []

        # do not split dimensions for dot product
        # R = R dot R
        strategy_list.append(self.no_split())

        # split two tensors in the same dimensions
        # S = S dot S
        strategy_list.append(self.split_one_dim(0))
        strategy_list.append(self.split_one_dim(1))

        return strategy_list


class MatVecStrategyGenerator(MatMulStrategyGenerator):

    def validate(self) -> bool:
        input_op_data = self.op_data['input']
        other_op_data = self.op_data['other']
        assert input_op_data.data.dim() == 2 and other_op_data.data.dim() == 1

    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        fwd_compute_cost = sharded_input_shape[0]
        bwd_compute_cost = fwd_compute_cost * 2
        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)
        return compute_cost

    @ignore_sharding_exception
    def no_split(self):
        name = "R = R x R"
        dim_partition_dict = {"input": {}, "other": {}, "output": {}}

        if self.has_bias:
            dim_partition_dict['bias'] = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping={})

    @ignore_sharding_exception
    def split_input_batch(self, mesh_dim):
        name = f'S{mesh_dim}R = S{mesh_dim}R x R'

        # get sharding spec
        dim_partition_dict = {
            "input": {
                0: [mesh_dim]
            },
            "other": {},
            "output": {
                0: [mesh_dim]
            },
        }

        if self.has_bias:
            dim_partition_dict['bias'] = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication action
        communication_action_mapping = {}
        if self.is_param('other'):
            other_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['other'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim,
                comm_type=CommType.HOOK)
        else:
            other_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['other'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim,
                comm_type=CommType.BEFORE,
                arg_index=1)
        communication_action_mapping['other'] = other_comm_action

        if self.has_bias:
            if self.is_param('bias'):
                bias_comm_action = self.get_communication_action(
                    sharding_spec=sharding_spec_mapping['bias'],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim,
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec=sharding_spec_mapping['bias'],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=mesh_dim,
                    comm_type=CommType.BEFORE,
                    arg_index=2)
            communication_action_mapping['bias'] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []

        # no split
        strategy_list.append(self.no_split())

        # split the batch dim for the first tensor only
        strategy_list.append(self.split_input_batch(0))
        strategy_list.append(self.split_input_batch(1))

        return strategy_list


class LinearProjectionStrategyGenerator(MatMulStrategyGenerator):

    def __init__(self, operation_data_mapping, device_mesh, linear_projection_type='linear'):
        super().__init__(operation_data_mapping, device_mesh)
        self.linear_projection_type = linear_projection_type

    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        # C = AB
        # C: [M, N], A: [M, P], B: [P, N]
        # fwd cost = MNP (only count mul)
        # bwd: 2 x fwd_cost
        sharded_input_shape = strategy.sharding_specs[self.op_data['input']].get_sharded_shape_per_device()
        sharded_other_shape = strategy.sharding_specs[self.op_data['other']].get_sharded_shape_per_device()
        dim_m_val = reduce(operator.mul, sharded_input_shape[:-1])
        dim_n_val = sharded_other_shape[-1]
        dim_p_val = sharded_other_shape[0]

        fwd_compute_cost = dim_m_val * dim_n_val * dim_p_val
        bwd_compute_cost = fwd_compute_cost * 2
        compute_cost = TrainCycleItem(fwd=bwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)
        strategy.compute_cost = compute_cost

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategies = []

        # SS = SR x RS
        strategies.append(self.split_lhs_space_rhs_space(0, 1))
        strategies.append(self.split_lhs_space_rhs_space(1, 0))

        # SR = SS x SR
        strategies.append(self.split_lhs_space_both_contract(0, 1))
        strategies.append(self.split_lhs_space_both_contract(1, 0))

        # RS = RS x SS
        strategies.append(self.split_rhs_space_both_contract(0, 1))
        strategies.append(self.split_rhs_space_both_contract(1, 0))

        # RR= RS x SR
        strategies.append(self.recompute_split_both_contract(0))
        strategies.append(self.recompute_split_both_contract(1))

        # RS = RR x RS
        strategies.append(self.split_rhs_space_only(0))
        strategies.append(self.split_rhs_space_only(1))

        # S01R = S01R x RR
        strategies.append(self.split_lhs_1st_dim_1d(0, 1))

        # RR = RS01 x S01R
        strategies.append(self.split_lhs_2nd_dim_1d(0, 1))

        # RS01 = RR x RS01
        strategies.append(self.split_rhs_2nd_dim_1d(0, 1))

        # RR = RR x RR
        strategies.append(self.non_split())

        return strategies

    @ignore_sharding_exception
    def split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        # handle case SS = SR x RS
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'
        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                -1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
        }

        # linear bias only has one dimension, but addmm bias has same dimensions
        # as the output logically.
        if self.linear_projection_type == 'linear':
            dim_partition_dict_mapping['bias'] = {-1: [mesh_dim_1]}
        elif self.linear_projection_type == 'addmm':
            dim_partition_dict_mapping['bias'] = {0: [mesh_dim_0], -1: [mesh_dim_1]}
        else:
            raise ('Unsupported linear projection type')

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        communication_action_mapping = {}
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=0)

        if self.is_param('other'):
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

        communication_action_mapping['input'] = input_comm_action
        communication_action_mapping['other'] = other_comm_action

        # we only add allreduce comm action for linear bias, because
        # allreduce comm action for addmm bias will be considered in post processing
        if self.has_bias and self.linear_projection_type == 'linear':
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
            communication_action_mapping['bias'] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_lhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        # handle the case SR = SS x SR
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        # get sharding spec mapping
        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0]
            },
        }

        # linear bias only has one dimension, but addmm bias has same dimensions
        # as the output logically.
        if self.linear_projection_type == 'linear':
            dim_partition_dict_mapping['bias'] = {}
        elif self.linear_projection_type == 'addmm':
            dim_partition_dict_mapping['bias'] = {0: [mesh_dim_0]}
        else:
            raise ('Unsupported linear projection type')

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action mapping
        communication_action_mapping = {}

        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.AFTER)

        if self.is_param('other'):
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

        communication_action_mapping['other'] = other_comm_action
        communication_action_mapping['output'] = output_comm_action

        # we only add allreduce comm action for linear bias, because
        # allreduce comm action for addmm bias will be considered in post processing
        if self.has_bias and self.linear_projection_type == 'linear':
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
            communication_action_mapping['bias'] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        # get sharding specs
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim_0]
            },
            "other": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_1]
            },
        }

        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication actions
        communication_action_mapping = {}
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_0,
            comm_type=CommType.AFTER)
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=0)
        communication_action_mapping["input"] = input_comm_action
        communication_action_mapping['output'] = output_comm_action
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def recompute_split_both_contract(self, mesh_dim):
        name = f'RR = RS{mesh_dim} x S{mesh_dim}R'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim]
            },
            "other": {
                0: [mesh_dim]
            },
            "bias": {},
            "output": {},
        }
        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        communication_action_mapping = {}
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim,
            comm_type=CommType.AFTER)

        communication_action_mapping['output'] = output_comm_action
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_rhs_space_only(self, mesh_dim):
        name = f'RS{mesh_dim} = RR x RS{mesh_dim}'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                -1: [mesh_dim]
            },
            "bias": {
                -1: [mesh_dim]
            },
            "output": {
                -1: [mesh_dim]
            },
        }
        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication actions
        communication_action_mapping = {}
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim,
            comm_type=CommType.BEFORE,
            arg_index=0)

        communication_action_mapping['input'] = input_comm_action
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_lhs_1st_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}{mesh_dim_1}R = S{mesh_dim_0}{mesh_dim_1}R x RR'
        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "other": {},
            "bias": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1]
            },
        }

        # linear bias only has one dimension, but addmm bias has same dimensions
        # as the output logically.
        if self.linear_projection_type == 'linear':
            dim_partition_dict_mapping['bias'] = {}
        elif self.linear_projection_type == 'addmm':
            dim_partition_dict_mapping['bias'] = {0: [mesh_dim_0, mesh_dim_1]}
        else:
            raise ('Unsupported linear projection type')

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        communication_action_mapping = {}
        if self.is_param('other'):
            other_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['other'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.HOOK)
        else:
            other_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['other'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.BEFORE,
                arg_index=1)
        communication_action_mapping['other'] = other_comm_action

        # we only add allreduce comm action for linear bias, because
        # allreduce comm action for addmm bias will be considered in post processing
        if self.has_bias and self.linear_projection_type == 'linear':
            if self.is_param('bias'):
                bias_comm_action = self.get_communication_action(
                    sharding_spec=sharding_spec_mapping['bias'],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=[mesh_dim_0, mesh_dim_1],
                    comm_type=CommType.HOOK)
            else:
                bias_comm_action = self.get_communication_action(
                    sharding_spec=sharding_spec_mapping['bias'],
                    communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                    logical_process_axis=[mesh_dim_0, mesh_dim_1],
                    comm_type=CommType.BEFORE,
                    key_for_kwarg='bias')
            communication_action_mapping['bias'] = bias_comm_action
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_lhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {},
            "output": {},
        }

        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        communication_action_mapping = {}
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1],
            comm_type=CommType.AFTER)
        communication_action_mapping['output'] = output_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_rhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RR x RS{mesh_dim_0}{mesh_dim_1}'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
        }

        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        communication_action_mapping = {}
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1],
            comm_type=CommType.BEFORE,
            arg_index=0)
        communication_action_mapping['input'] = input_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def non_split(self):
        name = f'RR = RR x RR'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {},
            "other": {},
            "bias": {},
            "output": {},
        }

        # We don't have to do anything special for bias here, because
        # the bias is already the same sharding spec as the output.
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        communication_action_mapping = {}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def validate(self) -> bool:
        assert "input" in self.op_data
        assert "other" in self.op_data

        # make sure the other has 2 dim
        input_data = self.op_data['input']
        other_data = self.op_data['other']
        assert input_data.data.dim() > 0 and other_data.data.dim() == 2
        assert other_data.logical_shape[0] == input_data.logical_shape[-1]

        if self.has_bias:
            bias_data = self.op_data['bias']
            assert bias_data.logical_shape[-1] == other_data.logical_shape[-1]


class BatchedMatMulStrategyGenerator(MatMulStrategyGenerator):
    """
    Generate sharding strategies for the batched matrix multiplication.

    A batched matrix multiplication can be viewed as
    [b, i, k] x [b, k, j] -> [b, i, j]

    The bias term is considered to have a 2D logical shape.

    Note: This class will be used to generate strategies for torch.bmm
    and torch.addbmm. However, the result of torch.addbmm is not correct,
    some extra runtime apply actions are required to keep numerical correctness.
    """

    # TODO: torch.addbmm correctness issue need to be fixed.
    def __init__(self, *args, **kwargs):
        self.squeeze_batch_dim = False
        super().__init__(*args, **kwargs)

    def _pop_batch_dim_sharding_for_output(self, dim_partition_dict):
        # remove partition dict for dim 0
        dim_partition_dict['output'].pop(0, None)

        # decrease the remaining dim index by 1
        temp_dim_partition = {}
        keys = list(dim_partition_dict['output'].keys())
        for key in keys:
            val = dim_partition_dict['output'].pop(key)
            temp_dim_partition[key - 1] = val
        dim_partition_dict['output'].update(temp_dim_partition)

    def validate(self) -> bool:
        input_op_data = self.op_data['input']
        other_op_data = self.op_data['other']
        assert len(input_op_data.logical_shape) == 3 or len(other_op_data.logical_shape) == 3

        if 'bias' in self.op_data:
            bias_op_data = self.op_data['bias']
            assert bias_op_data.data.dim() < 3 and len(bias_op_data.logical_shape) == 2

        if self.op_data['output'].data.dim() == 2:
            # addbmm will shrink the first batch dim
            self.squeeze_batch_dim = True

    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        fwd_compute_cost = self.op_data['input'].data.shape[-1] * reduce(operator.mul,
                                                                         self.op_data['output'].data.shape)
        bwd_compute_cost = fwd_compute_cost * 2
        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)
        strategy.compute_cost = compute_cost

    @ignore_sharding_exception
    def split_one_batch_dim(self, mesh_dim):
        name = f'Sb{mesh_dim} = Sb{mesh_dim} x Sb{mesh_dim}'

        # get sharding_spec
        dim_partition_dict = {"input": {0: [mesh_dim]}, "other": {0: [mesh_dim]}, "bias": {}, "output": {0: [mesh_dim]}}
        if self.squeeze_batch_dim:
            self._pop_batch_dim_sharding_for_output(dim_partition_dict)
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication actions
        communication_action_mapping = {}
        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['bias'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim,
                comm_type=CommType.BEFORE,
                arg_index=0)
            communication_action_mapping['bias'] = bias_comm_action
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_two_batch_dim(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}{mesh_dim_1} = Sb{mesh_dim_0}{mesh_dim_1} x Sb{mesh_dim_0}{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0, mesh_dim_1]
            }
        }
        if self.squeeze_batch_dim:
            self._pop_batch_dim_sharding_for_output(dim_partition_dict)
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication actions
        communication_action_mapping = {}
        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['bias'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.BEFORE,
                arg_index=0)
            communication_action_mapping['bias'] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_batch_dim_lhs_space(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}Si{mesh_dim_1} = Sb{mesh_dim_0}Si{mesh_dim_1} x Sb{mesh_dim_0}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0],
                1: [mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0]
            },
            "bias": {
                0: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                1: [mesh_dim_1]
            }
        }
        if self.squeeze_batch_dim:
            self._pop_batch_dim_sharding_for_output(dim_partition_dict)
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication actions
        communication_action_mapping = {}
        other_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['other'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=1)
        communication_action_mapping['other'] = other_comm_action

        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['bias'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=[mesh_dim_0, mesh_dim_1],
                comm_type=CommType.BEFORE,
                arg_index=0)
            communication_action_mapping['bias'] = bias_comm_action
            # for addbmm case, other is the third argument instead of second.
            communication_action_mapping['other'].arg_index += 1

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_batch_dim_rhs_space(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}Sj{mesh_dim_1} = Sb{mesh_dim_0}R x Sb{mesh_dim_0}Sj{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                0: [mesh_dim_0],
                2: [mesh_dim_1]
            },
            "bias": {
                1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                2: [mesh_dim_1]
            }
        }
        if self.squeeze_batch_dim:
            self._pop_batch_dim_sharding_for_output(dim_partition_dict)
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication actions
        communication_action_mapping = {}
        input_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.BEFORE,
            arg_index=0)
        communication_action_mapping['input'] = input_comm_action

        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['bias'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.BEFORE)
            communication_action_mapping['bias'] = bias_comm_action
            # for addbmm case, other is the second argument instead of first.
            communication_action_mapping['input'].arg_index += 1

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    @ignore_sharding_exception
    def split_batch_dim_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'Sb{mesh_dim_0}R = Sb{mesh_dim_0}Sk{mesh_dim_1} x Sb{mesh_dim_0}Sk{mesh_dim_1}'
        dim_partition_dict = {
            "input": {
                0: [mesh_dim_0],
                2: [mesh_dim_1]
            },
            "other": {
                0: [mesh_dim_0],
                1: [mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0],
            }
        }
        if self.squeeze_batch_dim:
            self._pop_batch_dim_sharding_for_output(dim_partition_dict)
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict)

        # get communication actions
        communication_action_mapping = {}
        output_comm_action = self.get_communication_action(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_1,
            comm_type=CommType.AFTER)
        communication_action_mapping['output'] = output_comm_action

        if self.has_bias:
            bias_comm_action = self.get_communication_action(
                sharding_spec=sharding_spec_mapping['bias'],
                communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
                logical_process_axis=mesh_dim_0,
                comm_type=CommType.BEFORE,
                arg_index=0)
            communication_action_mapping['bias'] = bias_comm_action

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        device_mesh_is_1d = True
        if len(self.device_mesh.mesh_shape) == 2 and 1 not in self.device_mesh.mesh_shape:
            device_mesh_is_1d = False

        if device_mesh_is_1d:
            # split only the batch dimension
            # Sb = Sb x Sb
            # can be None as it is only for 1D device mesh
            # only for 1D device mesh
            if len(self.device_mesh.mesh_shape) == 1:
                mesh_dim = 0
            else:
                mesh_dim = self.device_mesh.mesh_shape.index(1)
            strategy_list.append(self.split_one_batch_dim(mesh_dim))
        else:
            # for 2D device mesh
            # split batch dim of two inputs and the i dim of the first tensor
            # SbSi = SbSi x Sb
            strategy_list.append(self.split_batch_dim_lhs_space(0, 1))
            strategy_list.append(self.split_batch_dim_lhs_space(1, 0))

            # split batch dim of two inputs and the j of the second tensor
            # SbSj = Sb x SbSj
            strategy_list.append(self.split_batch_dim_rhs_space(0, 1))
            strategy_list.append(self.split_batch_dim_rhs_space(1, 0))

            # split batch dim of two inputs and the k dim of two inputs
            # Sb = SbSk x SbSk, need to all-reduce by k dim
            strategy_list.append(self.split_batch_dim_both_contract(0, 1))
            strategy_list.append(self.split_batch_dim_both_contract(1, 0))

            # split two batch dim
            strategy_list.append(self.split_two_batch_dim(0, 1))

        return strategy_list
