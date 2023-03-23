from enum import Enum


class ComputePattern(Enum):
    TP1D = 0
    TP2D = 1
    TP2P5D = 2
    TP3D = 3


class ComputeSpec(object):
    """ComputeSpec
    The Specification for compuattion pattern

    Args:
        compute_pattern (ComputePattern): an Enum instance for compute pattern.
    """

    def __init__(self, compute_pattern: ComputePattern) -> None:
        assert isinstance(compute_pattern, ComputePattern)
        self.compute_pattern = compute_pattern
        # Make sure output tensors are replicate
        self.output_replicate = True

    def __repr__(self):
        return f'ComputeSpec(pattern={self.compute_pattern}, replicate_output={self.output_replicate})'

    def set_output_replicate(self, flag: bool = True):
        self.output_replicate = flag
