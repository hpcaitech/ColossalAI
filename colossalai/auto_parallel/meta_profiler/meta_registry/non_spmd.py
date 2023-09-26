import operator
from typing import List, Tuple

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, TrainCycleItem

from ..registry import meta_register

__all__ = ["non_spmd_meta_info"]


@meta_register.register(torch.Size)
@meta_register.register(torch.Tensor.size)
@meta_register.register(torch.finfo)
@meta_register.register(operator.le)
def non_spmd_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """Non-SPMD node meta information generator
    Those nodes will not be handled by SPMD solver, so we just return all zero meta information for it

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """
    compute_cost = TrainCycleItem(fwd=0, bwd=0, total=0)
    memory_cost = TrainCycleItem(fwd=MemoryCost(), bwd=MemoryCost(), total=MemoryCost())
    fwd_in, fwd_buffer, fwd_out = [], [], []
    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
