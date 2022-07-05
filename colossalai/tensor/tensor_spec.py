import torch.distributed as dist
from typing import Optional
from colossalai.tensor.distspec import _DistSpec, DistPlacementPattern
from .compute_spec import ComputeSpec, ComputePattern


class TensorSpec(object):
    """
    The specification of the ColoTensor.
    Args:
        dist_spec (_DistSpec): descriping the layout among processes.
        compute_spec (Optional[ComputeSpec], optional): actions conducted on the tensor after initialization if it's a model data tensor. 
        Defaults to None.
    """

    def __init__(self, dist_spec: _DistSpec, compute_spec: Optional[ComputeSpec] = None):
        self.compute_spec = compute_spec
        self.dist_spec = dist_spec

    def get_process_group(self):
        return self.dist_spec.process_group

    def get_placement(self):
        return self.dist_spec.placement

    def is_replicate(self):
        return self.dist_spec.placement == DistPlacementPattern.REPLICATE \
            or (len(self.dist_spec.num_partitions) == 1
                and self.dist_spec.num_partitions[0] == 1) \
            or (self.dist_spec.process_group.tp_world_size() == 1)

    def is_shard_1dcol(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == -1

    def is_shard_1drow(self):
        return self.dist_spec.placement == DistPlacementPattern.SHARD \
            and len(self.dist_spec.dims) == 1 and self.dist_spec.dims[0] == 0

    def has_compute_pattern(self, compute_pattern: ComputePattern):
        return self.compute_spec.compute_pattern == compute_pattern

    def __repr__(self):
        return f'parallel action: {self.compute_spec}, dist_spec: {self.dist_spec}'
