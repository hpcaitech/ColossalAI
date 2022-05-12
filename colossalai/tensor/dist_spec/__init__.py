from .dist_spec import _DistSpec, DistPlacementPattern
from .dist_spec_mgr import DistSpecManager
from torch.distributed import ProcessGroup
from typing import Optional, List


def replicate(process_group: Optional[ProcessGroup] = None) -> _DistSpec:
    return _DistSpec(DistPlacementPattern.REPLICATE, process_group)


def shard(process_group: ProcessGroup, dims: List[int], num_partitions: List[int]) -> _DistSpec:
    assert process_group is not None
    assert isinstance(dims, list) and isinstance(num_partitions, list)
    assert len(dims) == len(num_partitions)
    return _DistSpec(DistPlacementPattern.SHARD, process_group, dims=tuple(dims), num_partitions=tuple(num_partitions))


__all__ = ['DistSpecManager', 'replicate', 'shard']
