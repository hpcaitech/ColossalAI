import torch.distributed as dist
from enum import Enum
from typing import List, Optional
from colossalai.tensor.distspec import _DistSpec, DistPlacementPattern


class ComputePattern(Enum):
    TP1D = 0
    TP2D = 1
    TP2P5D = 2
    TP3D = 3


class ParallelAction(object):

    def __init__(self, compute_pattern: ComputePattern) -> None:
        assert isinstance(compute_pattern, ComputePattern)
        self.compute_pattern = compute_pattern

    def __repr__(self):
        return f'compute pattern: {self.compute_pattern}'


class TensorSpec(object):
    """
    The specification of the ColoTensor.
    Args:
        dist_spec (_DistSpec): descriping the layout among processes.
        parallel_action (Optional[ParallelAction], optional): actions conducted on the tensor after initialization if it's a model data tensor. 
        Defaults to None.
    """

    def __init__(self, dist_spec: _DistSpec, parallel_action: Optional[ParallelAction] = None):
        self.parallel_action = parallel_action
        self.dist_spec = dist_spec

    def get_process_group(self):
        return self.dist_spec.process_group

    def get_process_group_size(self):
        return dist.get_world_size(self.dist_spec.process_group)

    def get_placement(self):
        return self.dist_spec.placement

    def is_gathered(self):
        return self.dist_spec.placement == DistPlacementPattern.REPLICATE \
            or (len(self.dist_spec.num_partitions) == 1
                and self.dist_spec.num_partitions[0] == 1) \
            or (self.dist_spec.process_group.size() == 1)

    def is_1D_col(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == -1

    def is_1D_row(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == 0

    def has_compute_pattern(self, compute_pattern: ComputePattern):
        return self.parallel_action.compute_pattern == compute_pattern

    def __repr__(self):
        return f'parallel action: {self.parallel_action}, dist_spec: {self.dist_spec}'
