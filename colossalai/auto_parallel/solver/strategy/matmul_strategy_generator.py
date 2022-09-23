from cmath import log
from distutils.log import Log
import operator
import torch
from functools import reduce
from ..sharding_strategy import ShardingStrategy_V2, TrainCycleItem, MemoryCost
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from .strategy_generator import StrategyGenerator_V2
from typing import List


class DotProductStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass


class MatVecStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass


class LinearProjectionStrategyGenerator(StrategyGenerator_V2):

    def update_compute_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
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

    def update_memory_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        input_size = self._compute_size_in_bytes(strategy, "input")
        other_size = self._compute_size_in_bytes(strategy, "input")

        if "bias" in self.op_data:
            bias_size = self._compute_size_in_bytes(strategy, "bias")
        else:
            bias_size = 0
        output_size = self._compute_size_in_bytes(strategy, "output")

        fwd_mem_cost = MemoryCost(activation=output_size, parameter=other_size + bias_size)
        bwd_mem_cost = MemoryCost(activation=input_size + other_size + bias_size, parameter=other_size)
        total_mem_cost = MemoryCost(activation=input_size + 2 * output_size + bias_size,
                                    parameter=other_size + bias_size)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def generate(self) -> List[ShardingStrategy_V2]:
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

        # update mete info on cost
        for strategy in strategies:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategies

    def split_lhs_space_rhs_space(self, mesh_dim_0, mesh_dim_1):
        # handle case SS = SR x RS
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'
        dim_partition_dict_mapping = {
            "input": {
                0: [mesh_dim_0]
            },
            "other": {
                self.dim_q: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                0: [mesh_dim_0],
                -1: [mesh_dim_1]
            },
        }
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # set communication action
        input_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1)
        other_comm_spec = self.get_communication_spec(
            sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_0)

        communication_action_mapping = {"input": input_comm_spec, "other": other_comm_spec}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

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
                self.dim_p: [mesh_dim_1]
            },
            "bias": {},
            "output": {
                0: [mesh_dim_0]
            },
        }
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action mapping
        input_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping["input"],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_0)
        output_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping["output"],
            communication_pattern=CollectiveCommPattern.REDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_1)

        communication_action_mapping = {"input": input_comm_spec, 'output': output_comm_spec}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def split_rhs_space_both_contract(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        # get sharding specs
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim_0]
            },
            "other": {
                self.dim_p: [mesh_dim_0],
                self.dim_q: [mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_1]
            },
        }
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication actions
        output_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.REDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim_0)
        input_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim_1)
        communication_action_mapping = {"output": output_comm_spec, "input": input_comm_spec}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def recompute_split_both_contract(self, mesh_dim):
        name = f'RR = RS{mesh_dim} x S{mesh_dim}R'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim]
            },
            "other": {
                self.dim_p: [mesh_dim]
            },
            "bias": {},
            "output": {},
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        output_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.REDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=mesh_dim)
        communication_action_mapping = {'output': output_comm_spec}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def split_rhs_space_only(self, mesh_dim):
        name = f'RS{mesh_dim} = RR x RS{mesh_dim}'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                self.dim_q: [mesh_dim]
            },
            "bias": {
                -1: [mesh_dim]
            },
            "output": {
                -1: [mesh_dim]
            },
        }

        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication actions
        input_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=mesh_dim)
        communication_action_mapping = {'input': input_comm_spec}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

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
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        other_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['other'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1])

        communcation_action_mapping = {"other": other_comm_spec}
        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communcation_action_mapping)

    def split_lhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RR = RS{mesh_dim_0}{mesh_dim_1} x S{mesh_dim_0}{mesh_dim_1}R'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "other": {
                self.dim_p: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {},
            "output": {},
        }
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        output_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['output'],
            communication_pattern=CollectiveCommPattern.REDUCE_FWD_IDENTITY_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1])
        communication_action_mapping = {'output': output_comm_spec}

        return self.get_sharding_strategy(name=name,
                                          sharding_spec_mapping=sharding_spec_mapping,
                                          communication_action_mapping=communication_action_mapping)

    def split_rhs_2nd_dim_1d(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_0}{mesh_dim_1} = RR x RS{mesh_dim_0}{mesh_dim_1}'

        # get sharding spec
        dim_partition_dict_mapping = {
            "input": {},
            "other": {
                self.dim_q: [mesh_dim_0, mesh_dim_1]
            },
            "bias": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
            "output": {
                -1: [mesh_dim_0, mesh_dim_1]
            },
        }
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        # get communication action
        input_comm_spec = self.get_communication_spec(
            sharding_spec=sharding_spec_mapping['input'],
            communication_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
            logical_process_axis=[mesh_dim_0, mesh_dim_1])
        communication_action_mapping = {'input': input_comm_spec}

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

        # check if bias has the same a valid dim
        has_bias = "bias" in self.op_data

        if has_bias:
            bias_data = self.op_data['bias']
            assert bias_data.logical_shape[-1] == other_data.logical_shape[-1]


class BatchedMatMulStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass
