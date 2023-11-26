from colossalai.legacy.tensor import ComputePattern, ProcessGroup, ShardSpec

from .colo_module import ColoModule


class ColoEmbedding(ColoModule):
    def __init__(self):
        super(ColoEmbedding, self).__init__()
        self._register_shard_params(["weight"])

    def register(self, compute_pattern, pg: ProcessGroup):
        if not compute_pattern in self._allowed_patterns:
            if ComputePattern.TP1D == compute_pattern:
                self._set_TP1D(pg)

    def _set_TP1D(self, pg: ProcessGroup):
        # TP1D Row Linear
        _compute_pattern = ComputePattern.TP1D
        self._register_allowed_patterns(
            compute_pattern=_compute_pattern,
            dist_specs={
                "weight": ShardSpec([0], [pg.tp_world_size()]),
            },
            mode="row",
        )

        # TP1D Col Linear
        self._register_allowed_patterns(
            compute_pattern=_compute_pattern,
            dist_specs={
                "weight": ShardSpec([-1], [pg.tp_world_size()]),
            },
            mode="col",
        )

        self._set_default(compute_pattern=_compute_pattern, target_mode="row")
