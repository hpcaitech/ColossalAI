from enum import Enum
from torch.distributed import ProcessGroup
from typing import Optional, List

__all__ = ['replicate', 'shard']


class DistPlacementPattern(Enum):
    REPLICATE = 'r'
    SHARD = 's'


class _DistSpec:

    def __init__(self,
                 dist_placement_pattern: DistPlacementPattern,
                 process_group: Optional[ProcessGroup] = None,
                 **meta_info):
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


def replicate(process_group: Optional[ProcessGroup] = None) -> _DistSpec:
    # process_group=None means global process group
    return _DistSpec(DistPlacementPattern.REPLICATE, process_group)


def shard(process_group: ProcessGroup, dims: List[int], num_partitions: List[int]) -> _DistSpec:
    assert process_group is not None
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    return _DistSpec(DistPlacementPattern.SHARD, process_group, dims=tuple(dims), num_partitions=tuple(num_partitions))
