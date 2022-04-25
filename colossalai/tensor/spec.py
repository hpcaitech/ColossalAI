from enum import Enum
from typing import Tuple, List


class ComputePattern(Enum):
    TP1DRow = 1,
    TP1DCol = 2,
    ZeRO = 3,


class TensorSpec(Enum):
    """
    It contains two aspects of information: 
    First, How are tensors distributed in Heterougenous memory space.
    Second, if the tensor is a model parameter, the Spec contains the 
    parallel computation pattern of the Operator (Layer).
    We have to consider the hybrid parallel mode.
    """
    # a list of tuple [(Priority, ComputePattern, ParallelGroup)]
    # For example: On 8 GPUs, a hybrid parallel strategy is applied.
    # We are using ZeRO with DP-degree = 4 and 1DRowTP with TP-degree = 2.
    # _dist_pattern = [(0, ComputePattern.ZeRO, gpc.get_group(ParallelMode.DATA)), \
    # (1, ComputePattern.TP1DRow, gpc.get_group(ParallelMode.PARALLEL_1D))]
    dist_pattern: List[Tuple] = []
