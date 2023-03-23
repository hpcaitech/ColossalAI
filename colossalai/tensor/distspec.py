from enum import Enum
from typing import List

__all__ = ['ReplicaSpec', 'ShardSpec']


class DistPlacementPattern(Enum):
    REPLICATE = 'r'
    SHARD = 's'


class _DistSpec:
    """_DistSpec

    A class indicates Distributed Specification.
    The DistSpec is only works for the tensor parallel process groups.
    Because the dist spec of data parallel process group can be automatically deduced.
    This is an internal data structrue.
    The API for users should be `ShardSpec` and `ReplicaSpec`.

    Args:
        dist_placement_pattern (DistPlacementPattern): the pattern describing how tensors are distributed among processes.
                                                The dist_placement_pattern is picked from a limited set, now including two patterns: replicate and shard.
        process_group (Optional[ProcessGroup], optional): the process group contains processes. Defaults to None.
    """

    def __init__(self, dist_placement_pattern: DistPlacementPattern, **meta_info):

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
        attr_list = []
        for attr in dir(self):
            if not attr.startswith('__'):
                attr_list.append(f'{attr}={str(getattr(self, attr))}')
        attr_str = ", ".join(attr_list)
        return "DistSpec(" + attr_str + ")"


def ReplicaSpec() -> _DistSpec:
    """ReplicaSpec

    A distributed specification represents the tensor is replicated among the tensor parallel process group.

    Returns:
        _DistSpec: an replicated dist spec instance.
    """
    return _DistSpec(DistPlacementPattern.REPLICATE)


def ShardSpec(dims: List[int], num_partitions: List[int]) -> _DistSpec:
    """ShardSpec

    A distributed specification represents the tensor is sharded among the tensor parallel process group.

    Note:
        Currently, only shard on one dimension is valid. In another word, dims should be of size 1.

    Args:
        dims (List[int]): a list of dimensions
        num_partitions (List[int]): a list of partition number of each dimensions.

    Returns:
        _DistSpec: an shard dist spec instance.
    """
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    return _DistSpec(DistPlacementPattern.SHARD, dims=tuple(dims), num_partitions=tuple(num_partitions))
