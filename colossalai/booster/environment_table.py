from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.context import ParallelMode
from colossalai.device.device_mesh import DeviceMesh

from .utils import initialize_device_mesh

__all__ = ['EnvironmentTable']


@dataclass
class DistInfo:
    mesh_shapes: List[Union[torch.Size, List[int], Tuple[int]]] = None
    num_meshes: int = -1
    world_size: int = -1
    local_world_size: int = -1
    strategy_axis_mapping: Dict[ParallelMode, int] = {}
    auto_generate: bool = False

    def __post_init__(self):
        if self.num_meshes == -1:
            self.num_meshes = len(self.mesh_shapes)
        if self.world_size == -1:
            self.world_size = dist.get_world_size()

        local_world_size = 0
        for mesh_shape in self.mesh_shapes:
            local_world_size += torch.Size(mesh_shape).numel()

        if self.local_world_size == -1:
            self.local_world_size = local_world_size

        assert self.num_meshes == len(
            self.mesh_shapes
        ), f'num_meshes({self.num_meshes}) should be equal to len(mesh_shapes)({len(self.mesh_shapes)})'
        assert self.local_world_size == local_world_size, f'the sum of numel of all elements in mesh_shapes should be equal to world size, but got {local_world_size} != {self.local_world_size}'


class EnvironmentTable:

    def __init__(self, physical_ids: List[int], dist_info: DistInfo):
        self.physical_ids = physical_ids
        self.global_rank = dist.get_rank()
        self.local_rank = self.physical_ids.index(self.global_rank)
        self._init_with_dist_info(dist_info)

    @property
    def is_master(self) -> bool:
        return self.local_rank == 0

    def _init_with_dist_info(self, dist_info: DistInfo):
        self.global_world_size = dist_info.world_size
        self.local_world_size = dist_info.local_world_size
        self.num_meshes = dist_info.num_meshes
        self.auto_generate = dist_info.auto_generate
        self.mesh_shapes = dist_info.mesh_shapes
        self.device_mesh_pool: Dict[int, DeviceMesh] = {}
        self.strategy_axis_mapping = dist_info.strategy_axis_mapping
        self.process_group_pool = {}

        if self.auto_generate:
            self._auto_init_environment()
        else:
            self._init_with_mesh_shapes()

            if self.strategy_axis_mapping is not None:
                self._init_process_group_pool()

    def _auto_init_environment(self):
        device_mesh = initialize_device_mesh(self.local_world_size, self.physical_ids)
        # We may support multiple device meshes generation in future, after our automatic pipeline-parallelism feature is ready.
        self.device_mesh_pool = {0: device_mesh}

    def _init_with_mesh_shapes(self):
        assert self.mesh_shapes is not None, 'mesh_shapes should not be None in non auto-generate mode.'
        self.physical_id_for_meshes = {}
        start = 0
        for index, mesh_shape in enumerate(self.mesh_shapes):
            step = torch.Size(mesh_shape).numel()
            self.physical_id_for_meshes[index] = self.physical_ids[start:start + step]
            start += step

        for i, (mesh_shape,
                physical_id_for_mesh) in enumerate(zip(self.mesh_shapes, list(self.physical_id_for_meshes.values()))):
            device_mesh = initialize_device_mesh(self.local_world_size, physical_id_for_mesh, mesh_shape)
            self.device_mesh_pool[i] = device_mesh

    def _init_process_group_pool(self):
        for i in range(self.num_meshes):
            if i != 0:
                assert hash(torch.Size(self.mesh_shapes[i])) == hash(
                    torch.Size(self.mesh_shapes[i - 1])
                ), f'We only support equal mesh shapes if the strategy axis mapping is not None, but got {self.mesh_shapes[i]} != {self.mesh_shapes[i-1]}'

        for strategy, axis in self.strategy_axis_mapping.items():
            assert strategy in ParallelMode, f'{strategy} is not supported yet.'
            assert axis < len(
                self.mesh_shapes[0]), f'axis({axis}) should be less than the dim of mesh({len(self.mesh_shapes[0])})'

            process_groups: Dict[int, ProcessGroup] = {}
            for i in range(self.num_meshes):
                for rank in self.physical_id_for_meshes[i]:
                    process_groups_list = self.device_mesh_pool[i].process_groups_dict[axis]
                    for process_group_tuple in process_groups_list:
                        if rank in process_group_tuple[0]:
                            process_groups[rank] = process_group_tuple[1]

            assert len(
                process_groups
            ) == self.local_world_size, f'process_groups should have {self.local_world_size} elements, but got {len(process_groups)}'

            self.process_group_pool[strategy] = process_groups

    # TODO: implement more utility methods as given in
    # https://github.com/hpcaitech/ColossalAI/issues/3051
