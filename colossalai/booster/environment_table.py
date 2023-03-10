from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist

from colossalai.context import ParallelMode

__all__ = ['EnvironmentTable']


@dataclass
class DistInfo:
    mesh_shapes: List[Union[torch.Size, List[int], Tuple[int]]]
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

    def __init__(self, physical_ids: List[int], intra_op_world_sizes: List[int]):
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        self.physical_ids = physical_ids

        self.local_rank = self.physical_ids.index(self.global_rank)
        self.local_world_size = len(self.physical_ids)
        # intra_op_world_sizes is a list of world sizes for each intra-op parallelism group.
        # For example, if we have 32 devices, and we want to use pipeline parallelism with 4 stages,
        # which means we have 4 intra-op parallelism groups, the intra_op_world_sizes could be [8, 8, 8, 8]
        # or [16, 8, 4, 4]
        self.intra_op_world_sizes = intra_op_world_sizes

    @property
    def is_master(self) -> bool:
        # TODO: implement this method
        pass

    # TODO: implement more utility methods as given in
    # https://github.com/hpcaitech/ColossalAI/issues/3051
