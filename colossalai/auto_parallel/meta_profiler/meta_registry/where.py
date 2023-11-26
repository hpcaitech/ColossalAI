from typing import List, Tuple

import torch

from colossalai._analyzer._subclasses.flop_tensor import flop_mapping
from colossalai._analyzer.fx.node_util import compute_size_in_bytes as activation_size
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, TrainCycleItem

from ..registry import meta_register

__all__ = ["where_meta_info"]


@meta_register.register(torch.where)
def where_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.where meta information generator

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    condition_tensor, x_tensor, y_tensor, output_tensor = [arg.data for arg in args]

    # compute cost
    fwd_compute_cost = 0

    # if we need to broadcast the condition tensor, during backward we need to do a reduce_sum
    bwd_compute_cost = 0
    if x_tensor.shape != output_tensor.shape:
        bwd_compute_cost += flop_mapping[torch.ops.aten.sum.dim_IntList]([output_tensor], [x_tensor])
    if y_tensor.shape != output_tensor.shape:
        bwd_compute_cost += flop_mapping[torch.ops.aten.sum.dim_IntList]([output_tensor], [y_tensor])

    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # memory cost
    # during the forward phase, torch.where will allocate memory for output tensor and condition tensor
    # during the backward phase, torch.where will allocate temp memory which is 3 times as output tensor, then generate
    # gradient matrix for input x and input y, remove the temp memory and condition tensor generated in forward phase
    # NOTE: currently in SPMD solver we always believe that there will be a new input tensor created in forward
    fwd_mem_cost = MemoryCost(activation=activation_size([condition_tensor, x_tensor, y_tensor, output_tensor]))
    bwd_mem_cost = MemoryCost(
        activation=activation_size([x_tensor, y_tensor]) - activation_size([condition_tensor]),
        parameter=0,
        temp=activation_size([output_tensor]) * 3
        + activation_size([condition_tensor])
        - activation_size([x_tensor, y_tensor]),
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
    fwd_in = [condition_tensor]
    fwd_buffer = []
    fwd_out = [output_tensor]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
