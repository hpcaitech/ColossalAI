import copy
from typing import List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from colossalai.tensor.sharding_spec import ShardingSpec

from .strategy_generator import FollowingStrategyGenerator

__all__ = ['ReshapeGenerator']


class ReshapeGenerator(FollowingStrategyGenerator):
    """
    ReshapeGenerator which deals with the sharding strategies of Reshape Op, such as torch.Tensor.permute.
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

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        # For reshape function, to keep the computing correctness we keep the sharding
        # spec of input is fully replicated. In addition, we will keep the output in
        # replica status and let the successor node choose the way to resharding the
        # output node. Therefore, the different strategies of input node with same
        # output sharding spec will generate same strategy for reshape function.
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_input = input_sharding_spec.dim_partition_dict
            dim_partition_dict_for_output = {}
            if isinstance(self.op_data["output"].data, tuple):
                dim_partition_dict_for_output = [{} for _ in range(len(self.op_data["output"].data))]
            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{sharding_spec_mapping["input"].sharding_sequence} -> FULLY REPLICATED_{index}'

            total_mesh_dim_list = []
            for mesh_dim_list in dim_partition_dict_for_input.values():
                total_mesh_dim_list.extend(mesh_dim_list)
            # if there is only one sharding dimension, we should use the value instead of list as logical_process_axis.
            if len(total_mesh_dim_list) == 1:
                total_mesh_dim_list = total_mesh_dim_list[0]
                input_comm_action = self.get_communication_action(
                    sharding_spec=sharding_spec_mapping["input"],
                    communication_pattern=CollectiveCommPattern.GATHER_FWD_SPLIT_BWD,
                    logical_process_axis=total_mesh_dim_list,
                    comm_type=CommType.BEFORE,
                    arg_index=0)
                input_comm_action.comm_spec.gather_dim = total_mesh_dim_list

            elif len(total_mesh_dim_list) >= 2:
                source_spec = sharding_spec_mapping["input"]
                target_spec = ShardingSpec(device_mesh=self.device_mesh,
                                           entire_shape=source_spec.entire_shape,
                                           dim_partition_dict={})
                comm_spec = {'src_spec': source_spec, 'tgt_spec': target_spec}
                input_comm_action = CommAction(comm_spec=comm_spec, comm_type=CommType.BEFORE, arg_index=0)

            else:
                input_comm_action = None

            if input_comm_action is not None:
                communication_action_mapping["input"] = input_comm_action
            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)
            strategy_list.append(strategy)

        for strategy in strategy_list:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategy_list
