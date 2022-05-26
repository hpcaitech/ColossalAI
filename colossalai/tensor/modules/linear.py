from .colo_module import ColoModule
from colossalai.tensor import ComputePattern, distspec
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode

class ColoLinear(ColoModule):
    def __init__(self):
        super(ColoLinear, self).__init__()
        self._register_shard_params(['weight', 'bias'])
        self._register = False
        
    def register(self):
        if self._register == False:
            self._set_TP1D()
            self._register = True
        
    def _set_TP1D(self):
        # TP1D Row Linear
        _compute_pattern = ComputePattern.TP1D
        self._register_allowed_patterns(
            compute_pattern=_compute_pattern,
            dist_specs={
                'weight': distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
                'bias': None
            },
            label='row',
        )

        # TP1D Col Linear
        self._register_allowed_patterns(
            compute_pattern=_compute_pattern,
            dist_specs={
                'weight': distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
                'bias': distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)])
            },
            label='col',
        )

        self._set_default(compute_pattern=_compute_pattern, target_label='row')
