from enum import Enum
from typing import Tuple, List
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


class ComputePattern(Enum):
    TP1DRow = 1
    TP1DCol = 2
    ZeRO = 3
    DP = 4


class ParallelAction(object):
    priority = 0
    compute_pattern = ComputePattern.DP
    process_group = gpc.get_group(ParallelMode.DATA)

    def __init__(self, priority, compute_pattern, process_group) -> None:
        self.priority = priority
        self.compute_pattern = compute_pattern
        self.process_group = process_group


class TensorSpec(Enum):
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
    # ParallelAction(1, ComputePattern.TP1DRow, gpc.get_group(ParallelMode.PARALLEL_1D))
    # ]
    # When the ColoTensor is initialized,
    # we first splitting tensor according to ParallelAction of ZeRO,
    # then splitting tensor according to ParallelAction of TP1DRow.
    # During Linear computation
    # Before Linear Op, we gather the tensors according to ZeRO.
    # We perform Linear Op according to compute pattern of TP1DRow.
    # After Linear Op, we split the tensors according to ZeRO.
    parallel_action_list: List[ParallelAction] = []
