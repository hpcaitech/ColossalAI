from typing import Callable, List, Tuple

import torch

from colossalai._analyzer.fx.node_util import compute_size_in_bytes
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem

from ..registry import meta_register

__all__ = ["tensor_related_metainfo"]


def tensor_related_metainfo(bwd_mem_out_factor: float = 1, bwd_mem_tmp_factor: float = 0) -> Callable:
    """torch.Tensor related metainfo generator template

    Args:
        bwd_mem_out_factor (float, optional): backward activation memory cost factor. Defaults to 1.
        bwd_mem_tmp_factor (float, optional): backward temp memory cost factor. Defaults to 0.

    Returns:
        Callable: torch.Tensor related metainfo generator
    """

    def meta_func(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
        """torch.Tensor related metainfo generator

        Returns:
            Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
        """
        outputs = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data

        # compute costs are all zero
        compute_cost = TrainCycleItem(fwd=0, bwd=0, total=0)

        # memory costs
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(outputs) * 2, parameter=0, temp=0, buffer=0)

        bwd_mem_cost = MemoryCost(
            activation=compute_size_in_bytes(outputs) * bwd_mem_out_factor,
            parameter=0,
            temp=compute_size_in_bytes(outputs) * bwd_mem_tmp_factor,
            buffer=0,
        )

        total_mem_cost = MemoryCost(
            activation=fwd_mem_cost.activation + bwd_mem_cost.activation,
            parameter=fwd_mem_cost.parameter + bwd_mem_cost.parameter,
            temp=fwd_mem_cost.temp + bwd_mem_cost.temp,
            buffer=fwd_mem_cost.buffer + bwd_mem_cost.buffer,
        )

        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)

        # store fwd_in, fwd_buffer, fwd_out
        fwd_in = []
        fwd_buffer = []
        if isinstance(outputs, tuple) or isinstance(outputs, list) or isinstance(outputs, dict):
            # tuple of tensors
            fwd_out = [torch.zeros_like(tensor) for tensor in outputs]
        else:
            # enaged_tensors is a single tensor
            fwd_out = [torch.zeros_like(outputs)]

        return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out

    return meta_func


# register torch.Tensor related metainfo
# (0, 0)
meta_register.register([torch.tensor, torch.Tensor.to, torch.Tensor.unsqueeze, torch.unsqueeze, torch.arange])(
    tensor_related_metainfo(0, 0)
)

# (1, 0)
meta_register.register(
    [
        torch.Tensor.flatten,
        torch.flatten,
        torch.Tensor.transpose,
        torch.transpose,
        torch.Tensor.permute,
        torch.permute,
        torch.Tensor.split,
        torch.split,
        torch.Tensor.view,
    ]
)(tensor_related_metainfo(1, 0))

# (1, 1)
meta_register.register([torch.Tensor.type, torch.Tensor.contiguous])(tensor_related_metainfo(1, 1))
