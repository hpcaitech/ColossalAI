from colossalai.tensor.distspec import _DistSpec
from colossalai.tensor import ComputePattern
from typing import List, Dict


class ColoModule(object):

    def __init__(self):
        self._shard_params: List[str] = []
        self._allowed_patterns: Dict[ComputePattern, Dict[str, Dict[str, _DistSpec]]] = {}

    def _register_shard_params(self, params: List[str]):
        self._shard_params = params

    def _register_allowed_patterns(self,
                                   compute_pattern: ComputePattern,
                                   dist_specs: Dict[str, _DistSpec],
                                   mode='default'):
        assert list(
            dist_specs.keys()).sort() == self._shard_params.sort(), 'Every registered param should have dist_spec.'
        if not compute_pattern in self._allowed_patterns:
            self._allowed_patterns[compute_pattern] = {}
        self._allowed_patterns[compute_pattern][mode] = dist_specs

    def _set_default(self, compute_pattern: ComputePattern, target_mode):
        self._allowed_patterns[compute_pattern]['default'] = self._allowed_patterns[compute_pattern][target_mode]

    def has_compute_pattern(self, compute_pattern: ComputePattern):
        return compute_pattern in self._allowed_patterns

    def get_dist_specs(self, compute_pattern: ComputePattern):
        assert self.has_compute_pattern(compute_pattern)
        return self._allowed_patterns[compute_pattern]

    def has_compute_pattern_with_mode(self, compute_pattern: ComputePattern, mode='default'):
        return compute_pattern in self._allowed_patterns and mode in self._allowed_patterns[compute_pattern]

    def get_dist_specs_with_mode(self, compute_pattern: ComputePattern, mode='default'):
        assert self.has_compute_pattern_with_mode(compute_pattern, mode)
        return self._allowed_patterns[compute_pattern][mode]

    def get_param_names(self):
        return self._shard_params

    def register(self, compute_pattern, pg):
        raise NotImplementedError
