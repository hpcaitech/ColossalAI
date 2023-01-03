import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, TrainCycleItem
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.tensor.sharding_spec import ShardingSpec, ShardingSpecException
from colossalai.tensor.utils import all_gather_simulator, all_to_all_simulator, mix_gather_simulator, shard_simulator

from .comm_spec import *

__all__ = ['ShapeConsistencyManager', 'ShapeConsistencyOptions', 'set_shape_consistency_options']


@dataclass
class ShapeConsistencyOptions:
    """
    ShapeConsistencyOptions is a dataclass which specifies the preferences for shape consistency.
    """
    # TODO: shape consistency option is not implemented yet
    pass


def to_global(distributed_tensor: torch.Tensor, sharding_spec: ShardingSpec) -> torch.Tensor:
    shape_consistency_manager = ShapeConsistencyManager()
    global_sharding_spec = ShardingSpec(sharding_spec.device_mesh, sharding_spec.entire_shape, {})
    with torch.no_grad():
        global_tensor = shape_consistency_manager.apply_for_autoparallel_runtime(distributed_tensor, sharding_spec,
                                                                                 global_sharding_spec)
    return global_tensor


def set_shape_consistency_options(options: ShapeConsistencyOptions):
    """
    Configure the shape consistency manager via function call.
    """
    manager = ShapeConsistencyManager()
    manager.options = options


class ShapeConsistencyManager(metaclass=SingletonMeta):

    def __init__(self):
        self._options = None
        self._forward_only = False
        self.total_communication_cost = 0
        self.total_transform_steps = 0
        self.cached_spec_pairs_transform_path = {}

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options_: ShapeConsistencyOptions):
        assert isinstance(options_, ShapeConsistencyOptions)
        self._options = options_

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, value):
        assert isinstance(value, bool)
        self._forward_only = value

    def get_all_all_gather_spec(self, source_spec: ShardingSpec,
                                orig_cost_dict: Dict[str, float]) -> Dict[ShardingSpec, float]:
        '''
        Get all valid sharding specs from source_spec with single all-gather operation, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the all-gather operation, we just care about the S dimension.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(Dict[str, float]): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-gather operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_gather_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: R,S1,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,R
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.GATHER_FWD_SPLIT_BWD
        for target_pair in source_spec.dim_partition_dict.items():
            shard_list = all_gather_simulator(target_pair)
            index = target_pair[0]
            new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)

            # We won't add empty list into dim_partition_dict
            # The key will be popped if the related shard_list is empty
            if shard_list:
                new_dim_partition_dict[index] = shard_list
            else:
                new_dim_partition_dict.pop(index)

            # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
            gather_dim = index
            logical_process_axis = target_pair[1][-1]
            comm_spec = CommSpec(
                comm_pattern,
                sharding_spec=source_spec,
                gather_dim=gather_dim,
            # shard_dim will be used during backward
                shard_dim=gather_dim,
                logical_process_axis=logical_process_axis,
                forward_only=self.forward_only)

            # compute the communication cost with CommSpec
            cost_dict = comm_spec.get_comm_cost()

            # generate new sharding spec
            try:
                new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                 source_spec.entire_shape,
                                                 dim_partition_dict=new_dim_partition_dict)
                for phase, cost in cost_dict.items():
                    cost_dict[phase] = cost + orig_cost_dict[phase]
                valid_spec_dict[new_sharding_spec] = (comm_spec, cost_dict)
            except ShardingSpecException:
                pass
        return valid_spec_dict

    def get_all_all_to_all_spec(self, source_spec: ShardingSpec,
                                orig_cost_dict: Dict[str, float]) -> Dict[ShardingSpec, float]:
        '''
        Get all valid sharding specs from source_spec with single all-to-all operation, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the all-to-all operation, we just care about the pairs containing S dimension.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(Dict[str, float]): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0], 1: [1]}
            # DistSpec:
            #     shard_sequence: S0,S1,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_all_to_all_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: S01,R,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: R,S1,S0
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,S1
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                # skip (R, R) cases
                if f_index not in source_spec.dim_partition_dict and b_index not in source_spec.dim_partition_dict:
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        # skip (S01, R) -> (R, S01) is NOT allowed
                        if len(source_spec.dim_partition_dict[f_index]) >= 2:
                            continue
                        f_target_pair = (f_index, deepcopy(source_spec.dim_partition_dict[f_index]))
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        # skip (R, S01) -> (S01, R) is NOT allowed
                        if len(source_spec.dim_partition_dict[b_index]) >= 2:
                            continue
                        b_target_pair = (b_index, deepcopy(source_spec.dim_partition_dict[b_index]))
                    else:
                        b_target_pair = (b_index, [])

                # skip (S1, S0) -> S10
                if f_target_pair[1] and b_target_pair[1] and f_target_pair[1][0] >= b_target_pair[1][0]:
                    continue
                f_shard_list, b_shard_list = all_to_all_simulator(f_target_pair, b_target_pair)
                f_index = f_target_pair[0]
                b_index = b_target_pair[0]

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                if len(f_shard_list) < len(f_target_pair[1]):
                    gather_dim = f_index
                    shard_dim = b_index
                    logical_process_axis = f_target_pair[1][-1]
                else:
                    gather_dim = b_index
                    shard_dim = f_index
                    logical_process_axis = b_target_pair[1][-1]
                comm_spec = CommSpec(comm_pattern,
                                     sharding_spec=source_spec,
                                     gather_dim=gather_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis,
                                     forward_only=self.forward_only)

                # compute the communication cost with CommSpec
                cost_dict = comm_spec.get_comm_cost()
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)

                # We won't add empty list into dim_partition_dict
                # The key will be popped if the related shard_list is empty
                if f_shard_list:
                    new_dim_partition_dict[f_index] = f_shard_list
                else:
                    new_dim_partition_dict.pop(f_index)
                if b_shard_list:
                    new_dim_partition_dict[b_index] = b_shard_list
                else:
                    new_dim_partition_dict.pop(b_index)

                # generate new sharding spec
                try:
                    new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                     source_spec.entire_shape,
                                                     dim_partition_dict=new_dim_partition_dict)
                    for phase, cost in cost_dict.items():
                        cost_dict[phase] = cost + orig_cost_dict[phase]
                    valid_spec_dict[new_sharding_spec] = (comm_spec, cost_dict)
                except ShardingSpecException:
                    pass

        return valid_spec_dict

    def get_all_shard_spec(self, source_spec: ShardingSpec, orig_cost_dict):
        '''
        Get all valid sharding specs from source_spec with single shard operation, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        For the sharding operation, we just care about legal sharding dimensions.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.

        Example:
            dim_partition_dict = {0: [0]}
            # DistSpec:
            #     shard_sequence: S0,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
            shape_consistency_manager = ShapeConsistencyManager()
            rst_dict = shape_consistency_manager.get_all_shard_spec(sharding_spec, {'forward': 0, 'backward': 0, 'total': 0})
            print(rst_dict)

        Output:
            {DistSpec:
            shard_sequence: S01,R,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,S1,R
            device_mesh_shape: (4, 4): 0, DistSpec:
            shard_sequence: S0,R,S1
            device_mesh_shape: (4, 4): 0}
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.SPLIT_FWD_GATHER_BWD

        # legal sharding dims means the mesh_id is still available to use.
        legal_sharding_dims = [i for i in range(len(source_spec.device_mesh.mesh_shape))]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)
        if len(legal_sharding_dims) == 0:
            return valid_spec_dict

        tensor_dims = len(source_spec.entire_shape)

        for index in range(tensor_dims):
            if index not in source_spec.dim_partition_dict:
                shard_list_list = shard_simulator((index, []), legal_sharding_dims)
            else:
                shard_list_list = shard_simulator((index, source_spec.dim_partition_dict[index]), legal_sharding_dims)
            if not shard_list_list:
                continue
            for shard_list in shard_list_list:
                new_dim_partition_dict = deepcopy(source_spec.dim_partition_dict)
                new_dim_partition_dict[index] = shard_list

                # generate the CommSpec to record the action of source_sharding_spec->new_sharding_spec
                shard_dim = index
                logical_process_axis = shard_list[-1]
                comm_spec = CommSpec(comm_pattern,
                                     sharding_spec=source_spec,
                                     gather_dim=shard_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis,
                                     forward_only=self.forward_only)

                # compute the communication cost with CommSpec
                cost_dict = comm_spec.get_comm_cost()

                # generate new sharding spec
                try:
                    new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                     source_spec.entire_shape,
                                                     dim_partition_dict=new_dim_partition_dict)
                    for phase, cost in cost_dict.items():
                        cost_dict[phase] = cost + orig_cost_dict[phase]
                    valid_spec_dict[new_sharding_spec] = (comm_spec, cost_dict)
                except ShardingSpecException:
                    pass
        return valid_spec_dict

    def get_all_mix_gather_spec(self, source_spec: ShardingSpec,
                                orig_cost_dict: Dict[str, float]) -> Dict[ShardingSpec, float]:
        '''
        S0S1 -> RR
        S1S0 -> RR
        S01R -> RR
        RS01 -> RR
        '''
        valid_spec_dict = {}
        comm_pathern = CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD
        tensor_dims = len(source_spec.entire_shape)
        for f_index in range(tensor_dims - 1):
            for b_index in range(f_index + 1, tensor_dims):
                if (f_index not in source_spec.dim_partition_dict) and (b_index not in source_spec.dim_partition_dict):
                    continue
                else:
                    if f_index in source_spec.dim_partition_dict:
                        # skip (S10, R) -> (R, R)
                        if len(f_target_pair[1]) == 2 and f_target_pair[1][0] >= f_target_pair[1][1]:
                            continue
                        f_target_pair = (f_index, deepcopy(source_spec.dim_partition_dict[f_index]))
                    else:
                        f_target_pair = (f_index, [])
                    if b_index in source_spec.dim_partition_dict:
                        # skip (R, S10) -> (R, R)
                        if len(b_target_pair[1]) == 2 and b_target_pair[1][0] >= b_target_pair[1][1]:
                            continue
                        b_target_pair = (b_index, deepcopy(source_spec.dim_partition_dict[b_index]))
                    else:
                        b_target_pair = (b_index, [])

                gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
                comm_spec = CommSpec(comm_pathern,
                                     sharding_spec=source_spec,
                                     gather_dim=gather_dim,
                                     logical_process_axis=logical_process_axes,
                                     forward_only=self.forward_only,
                                     mix_gather=True)
                cost_dict = comm_spec.get_comm_cost()
                new_dim_partition_dict = {}
                # generate new sharding spec
                try:
                    new_sharding_spec = ShardingSpec(source_spec.device_mesh,
                                                     source_spec.entire_shape,
                                                     dim_partition_dict=new_dim_partition_dict)
                    for phase, cost in cost_dict.items():
                        cost_dict[phase] = cost + orig_cost_dict[phase]
                    valid_spec_dict[new_sharding_spec] = (comm_spec, cost_dict)
                except ShardingSpecException:
                    pass

        return valid_spec_dict

    def get_all_one_step_transform_spec(self, source_spec: ShardingSpec, orig_cost_dict) -> Dict[ShardingSpec, float]:
        '''
        Get all valid sharding specs from source_spec with one step transform, and
        accumulate commucation cost on origin cost which will finally be used in auto sharding solver.
        Note:
            all-gather will eliminate a sharding dimension, all-to-all will keep sharding dimension same as before,
            and shard will add a sharding dimension. Therefore, the result of above operations are mutual exclusive,
            we could safely put them together.

        Argument:
            source_spec(ShardingSpec): the ShardingSpec of the source_spec.
            orig_cost(float): the original communication cost before this operation.

        Return:
            valid_spec_dict(Dict[ShardingSpec, float]): all valid sharding specs from source_spec with single all-to-all operation.
        '''
        valid_spec_dict = {}
        valid_spec_dict.update(self.get_all_all_gather_spec(source_spec, orig_cost_dict))
        valid_spec_dict.update(self.get_all_all_to_all_spec(source_spec, orig_cost_dict))
        valid_spec_dict.update(self.get_all_shard_spec(source_spec, orig_cost_dict))
        return valid_spec_dict

    def mem_cost(self, comm_action_sequence: List[CommSpec]) -> TrainCycleItem:
        """memory cost of the communication action sequence

        Args:
            comm_action_sequence (List[CommSpec]): list of communication actions

        Returns:
            TrainCycleItem: memory (numel) cost of such comm_action_sequence
        """

        def compute_shape(sharding_spec: ShardingSpec):
            shape = sharding_spec.entire_shape
            new_shape = []
            for dim, shard in sharding_spec.dim_partition_dict.items():
                new_shape.append(shape[dim] // len(shard))
            return new_shape

        def gather_analysis(comm_spec: CommSpec, discard_input: bool, alloc_numel: int, peak_numel: int):
            """analyze all_gather memory footprint
            all_gather will allocate memory for the output tensor, and there will be temp memory for
            all_gather operation, which is twice the size of output tensor

            Args:
                comm_spec (CommSpec): input CommSpec
                discard_input (bool): whether to discard the input tensor
                alloc_numel (int): current allocated numel
                peak_numel (int): current peak numel
            """
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = np.prod(input_shape)
            output_numel = input_numel * comm_spec.device_mesh.mesh_shape[comm_spec.logical_process_axis]
            peak_numel = max(peak_numel, alloc_numel + output_numel * 2)
            alloc_numel += output_numel
            if discard_input:
                alloc_numel -= input_numel

            return alloc_numel, peak_numel

        def split_analysis(comm_spec: CommSpec, discard_input: bool, alloc_numel: int, peak_numel: int):
            """analyze split memory footprint
            split will allocate memory for the output tensor if we don't apply shard on the first dimension of
            the input tensor. If we apply shard on the first dimension, the `torch.tensor.contiguous()` will not
            generate new tensor in this case, so no memory will be allocated.

            Args:
                comm_spec (CommSpec): input CommSpec
                discard_input (bool): whether to discard the input tensor
                alloc_numel (int): current allocated numel
                peak_numel (int): current peak numel
            """
            shard_dim = comm_spec.shard_dim
            if shard_dim != 0:
                # if we don't shard the tensor on the first dimension, the split action will
                # generate a new tensor
                input_shape = compute_shape(comm_spec.sharding_spec)
                input_numel = np.prod(input_shape)
                output_numel = input_numel // comm_spec.device_mesh.mesh_shape[comm_spec.logical_process_axis]
                alloc_numel += output_numel
                peak_numel = max(peak_numel, alloc_numel)
                if discard_input:
                    alloc_numel -= input_numel
            else:
                # if we shard the tensor on the first dimension, the split action will not generate
                # a new tensor, and as it will preserve a reference to the input tensor, we could
                # override the discard_input option here
                # NOTE: this special case might fail in some weird cases, e.g. if we have three split
                # actions in the comm actions sequence, the first split action operate on the second dimension,
                # the second split action operate on the first dimension, and the third split action operate, again,
                # on the second dimension. Therefore, after the first two actions in the sequence, we will allocate
                # memory the same size as the output of first split action. However, the third split action will discard
                # the input tensor, and it actually should discard the tensor generated by the first split action, so in
                # the current memory estimation framework, we will overestimate the memory usage. But the above case is
                # kind of weird, and I think we could ignore it for now.
                pass

            return alloc_numel, peak_numel

        def reduce_analysis(comm_spec: CommSpec, discard_input: bool, alloc_numel: int, peak_numel: int):
            """
            a dummy function for reduce memory footprint analysis, as the reduce action doesn't allocate extra memory
            """
            return alloc_numel, peak_numel

        def all2all_analysis(comm_spec: CommSpec, discard_input: bool, alloc_numel: int, peak_numel: int):
            """analyze all_to_all memory footprint
            all_to_all will allocate memory for the output tensor, and temp memory of all_to_all action
            is twice the size of output tensor if we shard input tensor on the first dimension, otherwise
            the temp memory is three times the size of output tensor

            Args:
                comm_spec (CommSpec): input CommSpec
                discard_input (bool): whether to discard the input tensor
                alloc_numel (int): current allocated numel
                peak_numel (int): current peak numel
            """
            input_shape = compute_shape(comm_spec.sharding_spec)
            input_numel = np.prod(input_shape)
            output_numel = input_numel
            shard_dim = comm_spec.shard_dim
            if shard_dim != 0:
                peak_numel = max(peak_numel, alloc_numel + output_numel * 3)
            else:
                peak_numel = max(peak_numel, alloc_numel + output_numel * 2)
            alloc_numel += output_numel
            if discard_input:
                alloc_numel -= input_numel

            return alloc_numel, peak_numel

        def identity_analysis(comm_spec: CommSpec, discard_input: bool, alloc_numel: int, peak_numel: int):
            """
            a dummy function for identity memory footprint analysis, as the identity action doesn't allocate extra memory
            """
            return alloc_numel, peak_numel

        pattern_to_func_dict = {
            CollectiveCommPattern.GATHER_FWD_SPLIT_BWD: [gather_analysis, split_analysis],
            CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD: [all2all_analysis, all2all_analysis],
            CollectiveCommPattern.SPLIT_FWD_GATHER_BWD: [split_analysis, gather_analysis],
            CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD: [reduce_analysis, identity_analysis],
            CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD: [identity_analysis, reduce_analysis],
            CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD: [],
        }

        fwd_actions = []
        bwd_actions = []

        # construct forward and backward comm actions sequence
        for comm_spec in comm_action_sequence:
            comm_spec: CommSpec
            fwd_action, bwd_action = pattern_to_func_dict[comm_spec.comm_pattern]
            fwd_actions.append(fwd_action)
            bwd_actions.append(bwd_action)

        # analyze memory footprint of forward comm actions sequence
        fwd_alloc_numel = 0
        fwd_peak_numel = 0
        for idx, action_spec_pair in enumerate(zip(fwd_actions, comm_action_sequence)):
            # the first forward comm action will not discard input
            fwd_action, comm_spec = action_spec_pair
            fwd_alloc_numel, fwd_peak_numel = fwd_action(comm_spec, False, fwd_alloc_numel,
                                                         fwd_peak_numel) if idx == 0 else fwd_action(
                                                             comm_spec, True, fwd_alloc_numel, fwd_peak_numel)

        # analyze memory footprint for backward comm actions sequence
        bwd_alloc_numel = 0
        bwd_peak_numel = 0
        for idx, action_spec_pair in enumerate(zip(reversed(bwd_actions), reversed(comm_action_sequence))):
            bwd_action, comm_spec = action_spec_pair
            bwd_alloc_numel, bwd_peak_numel = bwd_action(comm_spec, False, bwd_alloc_numel,
                                                         bwd_peak_numel) if idx == 0 else bwd_action(
                                                             comm_spec, True, bwd_alloc_numel, bwd_peak_numel)

        fwd_mem = MemoryCost(activation=fwd_alloc_numel, temp=fwd_peak_numel - fwd_alloc_numel)
        bwd_mem = MemoryCost(activation=bwd_alloc_numel, temp=bwd_peak_numel - bwd_alloc_numel)
        total_mem = MemoryCost(activation=fwd_alloc_numel + bwd_alloc_numel)

        return TrainCycleItem(fwd_mem, bwd_mem, total_mem)

    def shape_consistency(self, source_spec: ShardingSpec,
                          target_spec: ShardingSpec) -> Tuple[List[ShardingSpec], List[CommSpec], float]:
        '''
        This method will find a path to transform source_spec to target_spec with
        a greedy algorithm.
        The basic idea is:
        Step1:
            Generate all one-step transform sequences from source_spec.
        Step2:
            Pick the 'best' sharding spec following the heuristic function.
        Step3:
            Repeat above steps until the source spec transform to target spec.

        During finding the transform path, commucation cost will be accumulated, and it
        will be finally used in auto parallel solver.

        Additionally, to avoid repeating the path search in runtime, we cached all solved path
        in auto parallel strategy building time, which could handle most of cases in runtime.

        Argument:
            source_spec(ShardingSpec): ShardingSpec of the source activation.
            target_spec(ShardingSpec): ShardingSpec of the target activation.

        Return:
            transform_path(List[ShardingSpec]): The transform path from source_spec to target_spec,
                                                it contains the source_spec and target_spec.
            comm_action_sequence(List[CommSpec]): Keep the communication operations to complete the shape consistency in order.
            total_cost(float): total cost to complete shape consistency transform.

        Example:
            dim_partition_source = {1: [0, 1]}
            dim_partition_target = {0: [0, 1]}
            # DistSpec:
            #     shard_sequence: R,S01,R
            #     device_mesh_shape: (4, 4)
            sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)
            # DistSpec:
            #     shard_sequence: S01,R,R
            #     device_mesh_shape: (4, 4)
            sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)
            transform_path, comm_action_sequence, total_cost = shape_consistency_manager.shape_consistency(sharding_spec_source, sharding_spec_target)
            print(f'transform_path: {transform_path}')
            print(f'comm_action_sequence: {comm_action_sequence}')
            print(f'total_cost: {total_cost}')

        output:
            transform_path: [DistSpec:
                    shard_sequence: R,S01,R
                    device_mesh_shape: (4, 4), DistSpec:
                    shard_sequence: R,S0,R
                    device_mesh_shape: (4, 4), DistSpec:
                    shard_sequence: S0,R,R
                    device_mesh_shape: (4, 4), DistSpec:
                    shard_sequence: S01,R,R
                    device_mesh_shape: (4, 4)]
            comm_action_sequence: [CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1),
                                   CommSpec:(comm_pattern:all2all, gather_dim:1, shard_dim:0, logical_process_axis: 0),
                                   CommSpec:(comm_pattern:shard, shard_dim:0, logical_process_axis:1)]
            total_cost: 12294.402000000002
        '''
        MAX_TRANSFORM_STEPS = 20
        total_cost_dict = {'forward': 0, 'backward': 0, 'total': 0}
        total_steps = 0
        transform_path = []
        comm_action_sequence = []
        spec_pairs = (str(source_spec.sharding_sequence), str(target_spec.sharding_sequence))
        self.cached_spec_pairs_transform_path[spec_pairs] = (None, None)

        # We do nothing if the sharding spec is all the same.
        if source_spec.sharding_sequence_difference(target_spec) == 0:
            self.cached_spec_pairs_transform_path[spec_pairs] = (transform_path, comm_action_sequence)
            return (transform_path, comm_action_sequence, total_cost_dict)

        temp_sharding_spec = source_spec

        transform_path.append(temp_sharding_spec)
        # To avoid dead loop, the loop will break after MAX_TRANSFORM_STEPS transforms
        while total_steps <= MAX_TRANSFORM_STEPS:
            valid_transform_spec_dict = self.get_all_one_step_transform_spec(temp_sharding_spec, total_cost_dict)
            best_difference_score = math.inf

            for sharding_spec, info_pairs in valid_transform_spec_dict.items():
                comm_spec, cost_dict = info_pairs
                spec_difference = sharding_spec.sharding_sequence_difference(target_spec)

                if spec_difference == 0:
                    for phase, cost in total_cost_dict.items():
                        total_cost_dict[phase] = cost + cost_dict[phase]
                    transform_path.append(sharding_spec)
                    comm_action_sequence.append(comm_spec)
                    self.cached_spec_pairs_transform_path[spec_pairs] = (transform_path, comm_action_sequence)
                    return (transform_path, comm_action_sequence, total_cost_dict)

                if spec_difference < best_difference_score:
                    temp_sharding_spec = sharding_spec
                    temp_cost_dict = cost_dict
                    temp_comm_spec = comm_spec
                    best_difference_score = spec_difference

            transform_path.append(temp_sharding_spec)
            comm_action_sequence.append(temp_comm_spec)
            for phase, cost in total_cost_dict.items():
                total_cost_dict[phase] = cost + temp_cost_dict[phase]
            total_steps += 1

        raise RuntimeError(f"Could not find a valid transform path with in {MAX_TRANSFORM_STEPS} steps.")

    def apply(self, tensor_with_sharding_spec: torch.Tensor, target_spec: ShardingSpec) -> torch.Tensor:
        '''
        Apply target_spec to tensor with source sharding spec, the transform path is generated by the
        shape_consistency method.

        Argument:
            tensor_with_sharding_spec (torch.Tensor): a tensor with source sharding spec to be transformed to the target spec.
            target_spec (ShardingSpec): The tensor transform processes will be directed by the target_spec.

        Example:
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            entire_shape = torch.Size((4, 2))
            shape_consistency_manager = ShapeConsistencyManager()
            dim_partition_source = {0: [0]}
            dim_partition_target = {1: [0]}

            # DistSpec:
            #     shard_sequence: S0,R
            #     device_mesh_shape: (2, 2)
            sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)

            # DistSpec:
            #     shard_sequence: R,S0
            #     device_mesh_shape: (2, 2)
            sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)

            if rank in (0, 1):
                sharded_tensor_0 = torch.zeros(2, 1)
                sharded_tensor_1 = torch.ones(2, 1)
                # tensor([[0., 1.],
                #         [0., 1.]])
                tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()
            if rank in (2, 3):
                sharded_tensor_0 = torch.ones(2, 1) * 2
                sharded_tensor_1 = torch.ones(2, 1) * 3
                # tensor([[2., 3.],
                #         [2., 3.]])
                tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()

            tensor_to_comm.sharding_spec = sharding_spec_source
            shape_consistency_manager.apply(tensor_to_comm, sharding_spec_target)
            print(tensor_to_comm)

        Output in rank0 and rank2:
            tensor([[0.],
                    [0.],
                    [2.],
                    [2.]])

        Output in rank1 and rank3:
            tensor([[1.],
                    [1.],
                    [3.],
                    [3.]])
        '''
        _, comm_action_sequence, _ = self.shape_consistency(tensor_with_sharding_spec.sharding_spec, target_spec)
        for comm_spec in comm_action_sequence:
            tensor_with_sharding_spec = comm_spec.covert_spec_to_action(tensor_with_sharding_spec)
        tensor_with_sharding_spec.sharding_spec = target_spec
        return tensor_with_sharding_spec

    def apply_for_autoparallel_runtime(self, tensor, source_spec, target_spec):
        _, comm_action_sequence, _ = self.shape_consistency(source_spec, target_spec)
        for comm_spec in comm_action_sequence:
            tensor = comm_spec.covert_spec_to_action(tensor)
        tensor.sharding_spec = target_spec
        return tensor
