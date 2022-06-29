from enum import Enum
from colossalai.tensor import ProcessGroup
from typing import Optional, List
from numpy import prod

__all__ = ['replicate', 'shard']


class DistPlacementPattern(Enum):
    REPLICATE = 'r'
    SHARD = 's'


class _DistSpec:

    def __init__(self,
                 dist_placement_pattern: DistPlacementPattern,
                 process_group: Optional[ProcessGroup] = None,
                 **meta_info):
        """_DistSpec, Distributed Specification

        Args:
            dist_placement_pattern (DistPlacementPattern): the pattern describing how tensors are distributed among processes.
                                                    The dist_placement_pattern is picked from a limited set, now including two patterns: replicate and shard.
            process_group (Optional[ProcessGroup], optional): the process group contains processes. Defaults to None.
        """
        self.placement = dist_placement_pattern
        self.process_group = process_group
        for k, v in meta_info.items():
            setattr(self, k, v)

    def __eq__(self, other: "_DistSpec") -> bool:
        if dir(self) != dir(other):
            return False
        for attr in dir(self):
            if not attr.startswith('__') and getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __repr__(self) -> str:
        res = "\nDistSpec:\n\t"
        for attr in dir(self):
            if not attr.startswith('__'):
                res += f'{attr}: {str(getattr(self, attr))}\n\t'
        return res


def replicate(process_group: Optional[ProcessGroup] = None) -> _DistSpec:
    # process_group=None means global process group
    return _DistSpec(DistPlacementPattern.REPLICATE, process_group)


def shard(process_group: ProcessGroup, dims: List[int], num_partitions: List[int]) -> _DistSpec:
    assert process_group is not None and isinstance(process_group, ProcessGroup)
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    assert prod(num_partitions) == process_group.tp_world_size(), f"{num_partitions} {process_group.tp_world_size()}"
    return _DistSpec(DistPlacementPattern.SHARD, process_group, dims=tuple(dims), num_partitions=tuple(num_partitions))
