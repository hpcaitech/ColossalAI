import copy
from typing import List

from colossalai.auto_parallel.tensor_shard.node_handler.strategy.strategy_generator import FollowingStrategyGenerator
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    MemoryCost,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.auto_parallel.tensor_shard.utils import (
    check_keep_sharding_status,
    detect_reshape_mapping,
    infer_output_dim_partition_dict,
)
from colossalai.tensor.shape_consistency import CollectiveCommPattern
from colossalai.tensor.sharding_spec import ShardingSpec

__all__ = ['ReshapeGenerator', 'ViewGenerator', 'PermuteGenerator', 'TransposeGenerator', 'SplitGenerator']


class ReshapeGenerator(FollowingStrategyGenerator):
    """
    ReshapeGenerator is the base class for all the reshape operation.
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
        return super().collate_strategies()


class ViewGenerator(ReshapeGenerator):
    """
    ViewGenerator deals with the sharding strategies of view op.
    """

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]

            origin_shape = self.op_data['input'].data.shape
            tgt_shape = self.op_data['tgt_shape'].data

            reshape_mapping_dict = detect_reshape_mapping(origin_shape, tgt_shape)

            dim_partition_dict_for_input = input_sharding_spec.dim_partition_dict
            keep_sharding_status = check_keep_sharding_status(dim_partition_dict_for_input, reshape_mapping_dict)

            if keep_sharding_status:
                dim_partition_dict_for_output = infer_output_dim_partition_dict(dim_partition_dict_for_input,
                                                                                reshape_mapping_dict)
            else:
                dim_partition_dict_for_output = {}

            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            if keep_sharding_status:
                name = f'{sharding_spec_mapping["input"].sharding_sequence} -> {sharding_spec_mapping["output"].sharding_sequence}_{index}'
            else:
                name = f'{sharding_spec_mapping["input"].sharding_sequence} -> FULLY REPLICATED_{index}'

                # add comm action for converting input to fully replicated
                total_mesh_dim_list = []
                for mesh_dim_list in dim_partition_dict_for_input.values():
                    total_mesh_dim_list.extend(mesh_dim_list)
                # if there is only one sharding dimension, we should use the value instead of list as logical_process_axis.
                if len(total_mesh_dim_list) == 1:
                    total_mesh_dim_list = total_mesh_dim_list[0]
                    # the total mesh dim list only has one element, so the shard dim has only one element as well.
                    shard_dim = list(dim_partition_dict_for_input.keys())[0]
                    input_comm_action = self.get_communication_action(
                        sharding_spec=sharding_spec_mapping["input"],
                        communication_pattern=CollectiveCommPattern.GATHER_FWD_SPLIT_BWD,
                        logical_process_axis=total_mesh_dim_list,
                        comm_type=CommType.BEFORE,
                        arg_index=0)
                    # it will gather the input through gather_dim during forward phase.
                    input_comm_action.comm_spec.gather_dim = shard_dim
                    # it will split the input activation grad through shard_dim during backward phase.
                    input_comm_action.comm_spec.shard_dim = shard_dim

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

        return strategy_list


class PermuteGenerator(ReshapeGenerator):
    """
    PermuteGenerator deals with the sharding strategies of permute op.
    """

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]

            permute_dims = self.op_data['permute_dims'].data
            dim_partition_dict_for_input = input_sharding_spec.dim_partition_dict
            dim_partition_dict_for_output = {}
            for dim_index, permute_dim in enumerate(permute_dims):
                if permute_dim in dim_partition_dict_for_input:
                    dim_partition_dict_for_output[dim_index] = dim_partition_dict_for_input[permute_dim]

            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{sharding_spec_mapping["input"].sharding_sequence} -> {sharding_spec_mapping["output"].sharding_sequence}_{index}'

            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)
            strategy_list.append(strategy)

        return strategy_list


class TransposeGenerator(ReshapeGenerator):
    """
    TransposeGenerator deals with the sharding strategies of permute op.
    """

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_input = input_sharding_spec.dim_partition_dict
            dim_partition_dict_for_output = {}

            transpose_dims = self.op_data['transpose_dims'].data
            dim_0 = transpose_dims[0]
            dim_1 = transpose_dims[1]
            for dim, sharded_dims in dim_partition_dict_for_input.items():
                if dim == dim_0:
                    dim_partition_dict_for_output[dim_1] = dim_partition_dict_for_input[dim_0]
                elif dim == dim_1:
                    dim_partition_dict_for_output[dim_0] = dim_partition_dict_for_input[dim_1]
                else:
                    dim_partition_dict_for_output[dim] = sharded_dims

            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{sharding_spec_mapping["input"].sharding_sequence} -> {sharding_spec_mapping["output"].sharding_sequence}_{index}'

            strategy = self.get_sharding_strategy(name=name,
                                                  sharding_spec_mapping=sharding_spec_mapping,
                                                  communication_action_mapping=communication_action_mapping)
            strategy_list.append(strategy)

        return strategy_list


class SplitGenerator(ReshapeGenerator):
    """
    SplitGenerator deals with the sharding strategies of split op.
    """

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        for index, strategy in enumerate(self.predecessor_node.strategies_vector):
            recover_dims = None
            dim_partition_dict_mapping = {}
            communication_action_mapping = {}
            input_sharding_spec = strategy.output_sharding_specs[self.op_data["input"]]
            dim_partition_dict_for_input = copy.deepcopy(input_sharding_spec.dim_partition_dict)
            split_size, split_dim = self.op_data['split_info'].data

            if split_dim in dim_partition_dict_for_input:
                recover_dims = dim_partition_dict_for_input.pop(split_dim)

            dim_partition_dict_for_output = [
                copy.deepcopy(dim_partition_dict_for_input) for _ in range(len(self.op_data["output"].data))
            ]
            assert len(dim_partition_dict_for_output) >= 2
            dim_partition_dict_mapping = {
                "input": dim_partition_dict_for_input,
                "output": dim_partition_dict_for_output,
            }
            sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)
            # add index into name to pass the duplicated check
            # we keep same strategies with different name for node merging, and it will not increase the searching space,
            # because in solver, this node will be merged into other nodes, and solver will not create a new variable for this node.
            name = f'{sharding_spec_mapping["input"].sharding_sequence}_{index}'

            # add comm action if the input need to be recovered to replica in the split dimension.
            if recover_dims:
                # if there is only one sharding dimension, we should use the value instead of list as logical_process_axis.
                if len(recover_dims) == 1:
                    recover_dims = recover_dims[0]
                    input_comm_action = self.get_communication_action(
                        sharding_spec=sharding_spec_mapping["input"],
                        communication_pattern=CollectiveCommPattern.GATHER_FWD_SPLIT_BWD,
                        logical_process_axis=recover_dims,
                        comm_type=CommType.BEFORE,
                        arg_index=0)
                    # it will gather the input through gather_dim during forward phase.
                    input_comm_action.comm_spec.gather_dim = split_dim
                    # it will split the input activation grad through split_dim during backward phase.
                    input_comm_action.comm_spec.shard_dim = split_dim

                elif len(recover_dims) >= 2:
                    # original sharding spec
                    source_spec = input_sharding_spec
                    # target sharding spec
                    target_spec = sharding_spec_mapping["input"]
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

        return strategy_list
