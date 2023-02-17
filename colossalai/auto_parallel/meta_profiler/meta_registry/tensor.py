from typing import List, Tuple

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem
from colossalai.fx.profiler.memory_utils import activation_size
from colossalai.fx.profiler.opcount import flop_mapping

from ..registry import meta_register

__all__ = []


def tensor_related_metainfo(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.Tensor related operations metainfo generator
    torch.tensor: all zero, fwd_out
    torch.Tensor.size: all zero, fwd_out
    torch.Tensor.to: all zero, fwd_out
    torch.Tensor.type: bwd_mem_out, bwd_mem_tmp, fwd_out
    torch.Tensor.contiguous: bwd_mem_out, bwd_mem_tmp, fwd_out
    torch.Tensor.transpose: bwd_mem_out, fwd_out
    torch.Tensor.permute: bwd_mem_out, fwd_out
    torch.Tensor.split: bwd_mem_out, fwd_out
    torch.Tensor.view: bwd_mem_out, fwd_out


    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """
    pass
