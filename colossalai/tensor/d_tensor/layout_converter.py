import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.tensor.d_tensor.comm_spec import *
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.misc import LayoutException
from colossalai.tensor.utils import all_gather_simulator, all_to_all_simulator, shard_simulator

from .sharding_spec import ShardingSpec
from .utils import get_comm_cost

__all__ = ['LayoutConverter', 'LayoutConverterOptions', 'set_layout_converting_options']


@dataclass
class LayoutConverterOptions:
    """
    LayoutConverterOptions is a dataclass which specifies the preferences for layout converting.
    """
    # TODO: layout converter option is not implemented yet
    pass


def set_layout_converting_options(options: LayoutConverterOptions):
    """
    Configure the shape consistency manager via function call.
    """
    manager = LayoutConverter()
    manager.options = options


class LayoutConverter(metaclass=SingletonMeta):
    """
    LayoutConverter is a singleton class which converts the layout of a distributed tensor.
    """

    def __init__(self):
        self._options = None
        self._forward_only = False
        self.cached_solution = {}

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options_: LayoutConverterOptions):
        assert isinstance(options_, LayoutConverterOptions)
        self._options = options_

    @property
    def forward_only(self):
        return self._forward_only

    @forward_only.setter
    def forward_only(self, value):
        assert isinstance(value, bool)
        self._forward_only = value

    def all_gather_transform_layouts(self, source_layout: Layout) -> Dict[Layout, CommSpec]:
        '''
        Get all valid layouts from source_layout with single all-gather operation.
        For the all-gather operation, we just care about the S dimension.

        Argument:
            source_layout: the layout to be transformed.

        Return:
            valid_spec_dict(Dict[Layout, CommSpec]): all valid layouts from source_layout with single all-gather operation.

        Example:
            layout_converter = LayoutConverter()
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            global_shape = (4, 4, 4)
            dim_partition_dict = {0: [0], 1: [1]}

            # [S0,S1,R]
            sharding_spec = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_dict)
            layout = Layout(device_mesh=device_mesh,
                            sharding_spec=sharding_spec,
                            global_shape=global_shape)

            rst_dict = layout_converter.all_gather_transform_layouts(layout)
            for layout, comm_spec in rst_dict.items():
                print(f'{layout.sharding_spec.sharding_sequence}: {comm_spec}')

        Output:
            [R, S1, R]: CommSpec:(comm_pattern:GATHER_FWD_SPLIT_BWD, gather_dim:0, shard_dim:0, logical_process_axis:0)
            [S0, R, R]: CommSpec:(comm_pattern:GATHER_FWD_SPLIT_BWD, gather_dim:1, shard_dim:1, logical_process_axis:1)
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.GATHER_FWD_SPLIT_BWD
        source_spec = source_layout.sharding_spec

        # the key of the dict is the axis
        # the value is the process group
        current_rank = source_layout.device_mesh._global_rank_of_current_process
        process_group_dict = source_layout.device_mesh._process_group_dict[current_rank]

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
                process_group_dict=process_group_dict,
                gather_dim=gather_dim,
            # shard_dim will be used during backward
                shard_dim=gather_dim,
                logical_process_axis=logical_process_axis)

            # generate new sharding spec
            try:
                new_sharding_spec = ShardingSpec(source_spec.dims, dim_partition_dict=new_dim_partition_dict)
                new_layout = Layout(device_mesh=source_layout.device_mesh,
                                    sharding_spec=new_sharding_spec,
                                    global_shape=source_layout.global_shape)

                valid_spec_dict[new_layout] = comm_spec
            except LayoutException:
                pass
        return valid_spec_dict

    def all_to_all_transform_layout(self, source_layout: Layout) -> Dict[Layout, CommSpec]:
        '''
        Get all valid layouts from source_layout with single all-to-all operation.
        For the all-to-all operation, we just care about the pairs containing S dimension.

        Argument:
            source_layout(Layout): the layout to be transformed.

        Return:
            valid_spec_dict(Dict[Layout, CommSpec]): all valid layouts from source_layout with single all-to-all operation.

        Example:
            layout_converter = LayoutConverter()
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            global_shape = (4, 4, 4)
            dim_partition_dict = {0: [0], 1: [1]}

            # [S0,S1,R]
            sharding_spec = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_dict)
            layout = Layout(device_mesh=device_mesh,
                                    sharding_spec=sharding_spec,
                                    global_shape=global_shape)
            rst_dict = layout_converter.all_to_all_transform_layout(layout)

            for layout, comm_spec in rst_dict.items():
                print(f'{layout.sharding_spec.sharding_sequence}: {comm_spec}')

        Output:
            [S01, R, R]: CommSpec:(comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, gather_dim:1, shard_dim:0, logical_process_axis: 1)
            [R, S1, S0]: CommSpec:(comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, gather_dim:0, shard_dim:2, logical_process_axis: 0)
            [S0, R, S1]: CommSpec:(comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, gather_dim:1, shard_dim:2, logical_process_axis: 1)
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD

        # the key of the dict is the axis
        # the value is the process group
        current_rank = source_layout.device_mesh._global_rank_of_current_process
        process_group_dict = source_layout.device_mesh._process_group_dict[current_rank]

        source_spec = source_layout.sharding_spec
        tensor_dims = source_spec.dims
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
                                     process_group_dict=process_group_dict,
                                     gather_dim=gather_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis)

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
                    new_sharding_spec = ShardingSpec(source_spec.dims, dim_partition_dict=new_dim_partition_dict)
                    new_layout = Layout(device_mesh=source_layout.device_mesh,
                                        sharding_spec=new_sharding_spec,
                                        global_shape=source_layout.global_shape)
                    valid_spec_dict[new_layout] = comm_spec
                except LayoutException:
                    pass

        return valid_spec_dict

    def shard_transform_layout(self, source_layout: Layout) -> Dict[Layout, CommSpec]:
        '''
        Get all valid layouts from source_layout with single shard operation.
        For the sharding operation, we just care about legal sharding dimensions.

        Argument:
            source_layout(Layout): the layout to be transformed.

        Return:
            valid_spec_dict(Dict[Layout, CommSpec]): all valid layouts from source_layout with single shard operation.

        Example:
            layout_converter = LayoutConverter()
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            global_shape = (4, 4, 4)

            dim_partition_dict = {0: [0]}

            # [S0,R,R]
            sharding_spec = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_dict)
            layout = Layout(device_mesh=device_mesh,
                          sharding_spec=sharding_spec,
                          global_shape=global_shape)
            rst_dict = layout_converter.shard_transform_layout(layout)

            for layout, comm_spec in rst_dict.items():
                print(f'{layout.sharding_spec.sharding_sequence}: {comm_spec}')

        Output:
            [S01, R, R]: CommSpec:(comm_pattern:SPLIT_FWD_GATHER_BWD, gather_dim:0, shard_dim:0, logical_process_axis:1)
            [S0, S1, R]: CommSpec:(comm_pattern:SPLIT_FWD_GATHER_BWD, gather_dim:1, shard_dim:1, logical_process_axis:1)
            [S0, R, S1]: CommSpec:(comm_pattern:SPLIT_FWD_GATHER_BWD, gather_dim:2, shard_dim:2, logical_process_axis:1)
        '''
        valid_spec_dict = {}
        comm_pattern = CollectiveCommPattern.SPLIT_FWD_GATHER_BWD
        source_spec = source_layout.sharding_spec

        # the key of the dict is the axis
        # the value is the process group
        current_rank = source_layout.device_mesh._global_rank_of_current_process
        process_group_dict = source_layout.device_mesh._process_group_dict[current_rank]

        # legal sharding dims means the mesh_id is still available to use.
        legal_sharding_dims = [i for i in range(len(source_layout.device_mesh.shape))]
        for dim, shard_list in source_spec.dim_partition_dict.items():
            for element in shard_list:
                legal_sharding_dims.remove(element)

        if len(legal_sharding_dims) == 0:
            return valid_spec_dict

        tensor_dims = source_spec.dims

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
                                     process_group_dict=process_group_dict,
                                     gather_dim=shard_dim,
                                     shard_dim=shard_dim,
                                     logical_process_axis=logical_process_axis)

                # generate new sharding spec
                try:
                    new_sharding_spec = ShardingSpec(dim_size=source_spec.dims,
                                                     dim_partition_dict=new_dim_partition_dict)
                    new_layout = Layout(device_mesh=source_layout.device_mesh,
                                        sharding_spec=new_sharding_spec,
                                        global_shape=source_layout.global_shape)
                    valid_spec_dict[new_layout] = comm_spec
                except LayoutException:
                    pass
        return valid_spec_dict

    def get_all_one_step_transform_spec(self, source_layout: Layout) -> Dict[Layout, CommSpec]:
        '''
        Get all valid layouts from source_layout with one step transform.

        Note:
            all-gather will eliminate a sharding dimension, all-to-all will keep sharding dimension same as before,
            and shard will add a sharding dimension. Therefore, the result of above operations are mutual exclusive,
            we could safely put them together.

        Argument:
            source_layout(Layout): the layout to be transformer.

        Return:
            valid_spec_dict(Dict[Layout, CommSpec]): all valid layouts from source_layout with one step transform.
        '''
        valid_spec_dict = {}
        valid_spec_dict.update(self.all_gather_transform_layouts(source_layout))
        valid_spec_dict.update(self.all_to_all_transform_layout(source_layout))
        valid_spec_dict.update(self.shard_transform_layout(source_layout))
        return valid_spec_dict

    def layout_converting(self, source_layout: Layout,
                          target_layout: Layout) -> Tuple[List[Layout], List[CommSpec], float]:
        '''
        This method will find a path to transform source_layout to target_layout with
        a greedy algorithm.
        The basic idea is:
        Step1:
            Generate all one-step transform sequences from source_layout.
        Step2:
            Pick the 'best' layout following the heuristic function.
        Step3:
            Repeat above steps until the source layout transform to target layout.

        Additionally, to avoid repeating the path search in runtime, we cached all solved path
        in auto parallel strategy building time, which could handle most of cases in runtime.

        Args:
            source_layout(Layout): the layout to be transformed.
            target_layout(Layout): the layout to be achieved after a serious of transforms.

        Return:
            transform_path(List[Layout]): The transform path from source_layout to target_layout,
                                                it contains the source_layout and target_layout.
            comm_action_sequence(List[CommSpec]): Keep the communication operations to complete the layout converting in order.

        Example:
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            global_shape = (4, 4, 4)

            dim_partition_source = {1: [0, 1]}
            dim_partition_target = {0: [0, 1]}

            # [R,S01,R]
            sharding_spec_source = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_source)
            source_layout = Layout(device_mesh=device_mesh,
                                sharding_spec=sharding_spec_source,
                                global_shape=global_shape)

            # [S01,R,R]
            sharding_spec_target = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_target)
            target_layout = Layout(device_mesh=device_mesh,
                                sharding_spec=sharding_spec_target,
                                global_shape=global_shape)

            transform_path, comm_action_sequence = layout_converter.layout_converting(source_layout, target_layout)
            transform_path_str = '->'.join([str(layout.sharding_spec.sharding_sequence) for layout in transform_path])
            print(transform_path_str)

        output:
            [R, S01, R]->[R, S0, R]->[S0, R, R]->[S01, R, R]
        '''
        source_spec = source_layout.sharding_spec
        target_spec = target_layout.sharding_spec
        MAX_TRANSFORM_STEPS = 20
        total_steps = 0
        transform_path = []
        comm_action_sequence = []
        spec_pairs = (str(source_spec.sharding_sequence), str(target_spec.sharding_sequence))

        if spec_pairs in self.cached_solution:
            return self.cached_solution[spec_pairs]

        # We do nothing if the sharding spec is all the same.
        if source_spec.spec_diff(target_spec) == 0:
            self.cached_solution[spec_pairs] = (transform_path, comm_action_sequence)
            return (
                transform_path,
                comm_action_sequence,
            )

        temp_sharding_layout = source_layout

        transform_path.append(temp_sharding_layout)
        # To avoid dead loop, the loop will break after MAX_TRANSFORM_STEPS transforms
        while total_steps <= MAX_TRANSFORM_STEPS:
            valid_transform_spec_dict = self.get_all_one_step_transform_spec(temp_sharding_layout)
            best_difference_score = math.inf

            for layout, comm_spec in valid_transform_spec_dict.items():
                sharding_spec = layout.sharding_spec
                spec_difference = sharding_spec.spec_diff(target_spec)

                if spec_difference == 0:
                    transform_path.append(layout)
                    comm_action_sequence.append(comm_spec)
                    self.cached_solution[spec_pairs] = (transform_path, comm_action_sequence)
                    return (transform_path, comm_action_sequence)

                if spec_difference < best_difference_score:
                    temp_sharding_layout = layout
                    temp_comm_spec = comm_spec
                    best_difference_score = spec_difference

            transform_path.append(temp_sharding_layout)
            comm_action_sequence.append(temp_comm_spec)

            total_steps += 1

        raise RuntimeError(f"Could not find a valid transform path with in {MAX_TRANSFORM_STEPS} steps.")

    def get_total_comm_cost(self, source_layout: Layout, target_layout: Layout) -> Dict[str, float]:
        '''
        Get the total communication cost of the layout converting process.
        '''
        transform_path, comm_action_sequence = self.layout_converting(source_layout, target_layout)
        total_cost = {'forward': 0.0, 'backward': 0.0, 'total': 0.0}
        for layout, comm_spec in zip(transform_path, comm_action_sequence):
            cost_dict = get_comm_cost(layout, comm_spec, self.forward_only)
            for key in total_cost:
                total_cost[key] += cost_dict[key]
        return total_cost

    def apply(self, tensor: torch.Tensor, source_layout: Layout, target_layout: Layout) -> torch.Tensor:
        '''
        Apply target_layout to tensor with source layout, the transform path is generated by the
        layout_converting method.

        Argument:
            tensor (torch.Tensor): The tensor to be redistributed.
            source_layout(Layout): The source layout of the tensor.
            target_layout (Layout): The tensor will be redistributed to the target_layout.

        Example:
            layout_converter = LayoutConverter()
            dim_partition_source = {0: [0]}
            dim_partition_target = {1: [0]}
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1,
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
            global_shape = (4, 4, 4)

            # [S0,R,R]
            sharding_spec_source = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_source)
            source_layout = Layout(device_mesh=device_mesh,
                                sharding_spec=sharding_spec_source,
                                global_shape=global_shape)

            # [R,S0,R]
            sharding_spec_target = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_target)
            target_layout = Layout(device_mesh=device_mesh,
                                sharding_spec=sharding_spec_target,
                                global_shape=global_shape)

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

            # converted_tensor: [R, S0, R]
            converted_tensor = layout_converter.apply(tensor_to_comm, source_layout, target_layout)
            print(converted_tensor)

        Output in rank0 and rank1:
            tensor([[0.],
                    [0.],
                    [2.],
                    [2.]])

        Output in rank2 and rank3:
            tensor([[1.],
                    [1.],
                    [3.],
                    [3.]])
        '''
        _, comm_action_sequence = self.layout_converting(source_layout, target_layout)
        for comm_spec in comm_action_sequence:
            tensor = comm_spec.covert_spec_to_action(tensor)
        tensor.dist_layout = target_layout
        return tensor
