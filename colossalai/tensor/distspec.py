from enum import Enum
from typing import List

__all__ = ['replicate', 'shard']


class DistPlacementPattern(Enum):
    REPLICATE = 'r'
    SHARD = 's'


class _DistSpec:

    def __init__(self, dist_placement_pattern: DistPlacementPattern, **meta_info):
        """_DistSpec, Distributed Specification

        Args:
            dist_placement_pattern (DistPlacementPattern): the pattern describing how tensors are distributed among processes.
                                                    The dist_placement_pattern is picked from a limited set, now including two patterns: replicate and shard.
            process_group (Optional[ProcessGroup], optional): the process group contains processes. Defaults to None.
        """
        self.placement = dist_placement_pattern
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
        res_list = ["DistSpec:"]
        for attr in dir(self):
            if not attr.startswith('__'):
                res_list.append(f'\n\t{attr}: {str(getattr(self, attr))}')
        return ''.join(res_list)


def replicate() -> _DistSpec:
    return _DistSpec(DistPlacementPattern.REPLICATE)


def shard(dims: List[int], num_partitions: List[int]) -> _DistSpec:
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    return _DistSpec(DistPlacementPattern.SHARD, dims=tuple(dims), num_partitions=tuple(num_partitions))
