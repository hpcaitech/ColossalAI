import torch.distributed as dist
from enum import Enum
from typing import List
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor.distspec import _DistSpec, DistPlacementPattern


class ComputePattern(Enum):
    TP1D = 0
    ZeRO = 1
    DP = 2


class ParallelAction(object):

    def __init__(self,
                 priority=0,
                 compute_pattern=ComputePattern.DP,
                 parallel_mode=ParallelMode.DATA,
                 gather_out=True) -> None:
        self.priority = priority
        self.compute_pattern = compute_pattern
        self.parallel_mode = parallel_mode
        self.gather_out = gather_out


class TensorSpec(object):
    """
    It contains two aspects of information: 
    First, How are tensors distributed in Heterougenous memory space.
    Second, if the tensor is a model parameter, the Spec contains the 
    parallel computation pattern of the Operator (Layer).
    We have to consider the hybrid parallel mode.
    """

    # a list of parallel actions.
    # For example: On 8 GPUs, a hybrid parallel strategy is applied using
    # using ZeRO with DP-degree = 4 and 1DRowTP with TP-degree = 2.
    # parallel_action_list = [
    # ParallelAction(10, ComputePattern.ZeRO, gpc.get_group(ParallelMode.DATA)),
    # ParallelAction(1, ComputePattern.TP1D_Linear, gpc.get_group(ParallelMode.PARALLEL_1D))
    # ]
    # When the ColoTensor is initialized,
    # we first splitting tensor according to ParallelAction of ZeRO,
    # then splitting tensor according to ParallelAction of TP1D_Linear.
    # During Linear computation
    # Before Linear Op, we gather the tensors according to ZeRO.
    # We perform Linear Op according to compute pattern of TP1D_Linear.
    # After Linear Op, we split the tensors according to ZeRO.

    def __init__(self, dist_spec: _DistSpec, parallel_action_list: List[ParallelAction] = []):
        self._parallel_action_list = parallel_action_list
        self.dist_spec = dist_spec
        self.sort()

    @property
    def parallel_action_list(self):
        return self._parallel_action_list

    @property
    def num_action(self):
        return len(self._parallel_action_list)

    @property
    def compute_patterns(self):
        return [parallel_action.compute_pattern for parallel_action in self._parallel_action_list]

    def sort(self):
        if len(self._parallel_action_list) > 0:
            self._parallel_action_list.sort(key=lambda parallel_action: parallel_action.priority)

    def get_action_by_compute_pattern(self, compute_pattern: ComputePattern):
        for parallel_action in self._parallel_action_list:
            if parallel_action.compute_pattern == compute_pattern:
                return parallel_action
        return None

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
        return self.get_action_by_compute_pattern(compute_pattern) is not None
