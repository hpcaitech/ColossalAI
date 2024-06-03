from typing import List, Tuple

import torch

from colossalai._analyzer._subclasses.flop_tensor import flop_mapping
from colossalai._analyzer.fx.node_util import compute_size_in_bytes
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem

from ..registry import meta_register

__all__ = ["avgpool_meta_info", "maxpool_meta_info"]


@meta_register.register(torch.nn.AdaptiveAvgPool1d)
@meta_register.register(torch.nn.AdaptiveAvgPool2d)
@meta_register.register(torch.nn.AdaptiveAvgPool3d)
def avgpool_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """Meta info for AdaptiveAvgPool
    The aten graph of AdaptiveAvgPool is
    graph():
    %input_2 : [#users=2] = placeholder[target=placeholder](default=)
    %_adaptive_avg_pool2d_default : [#users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%input_2, [None, None]), kwargs = {})
    %zeros_like_default : [#users=1] = call_function[target=torch.ops.aten.zeros_like.default](args = (%_adaptive_avg_pool2d_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%input_2,), kwargs = {})
    %_adaptive_avg_pool2d_backward_default : [#users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d_backward.default](args = (%zeros_like_default, %detach_default), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%_adaptive_avg_pool2d_backward_default,), kwargs = {})
    %detach_default_2 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_1,), kwargs = {})

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    input_tensor = args[0].data
    output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
    is_inplace = kwargs.get("inplace", False)

    # construct forward args for flop mapping
    fwd_in_args = [input_tensor]
    fwd_out_args = [output_tensor]

    # construct backward args for flop mapping
    bwd_in_args = [output_tensor]
    bwd_out_args = [input_tensor]

    # calculate cost
    # the fwd op with compute cost is _adaptive_avg_pool2d.default
    # the bwd op with compute cost is _adaptive_avg_pool2d_backward.default

    # calculate compute cost
    fwd_compute_cost = flop_mapping[torch.ops.aten._adaptive_avg_pool2d.default](fwd_in_args, fwd_out_args)
    bwd_compute_cost = flop_mapping[torch.ops.aten._adaptive_avg_pool2d_backward.default](bwd_in_args, bwd_out_args)
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # calculate memory cost
    fwd_mem_cost = MemoryCost() if is_inplace else MemoryCost(activation=compute_size_in_bytes(output_tensor))
    bwd_mem_cost = MemoryCost() if is_inplace else MemoryCost(activation=compute_size_in_bytes(input_tensor))

    # total cost
    total_mem_cost = MemoryCost(activation=fwd_mem_cost.activation + bwd_mem_cost.activation)

    mem_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = []
    fwd_buffer = []
    fwd_out = [torch.zeros_like(output_tensor, device="meta")]

    return compute_cost, mem_cost, fwd_in, fwd_buffer, fwd_out


@meta_register.register(torch.nn.MaxPool1d)
@meta_register.register(torch.nn.MaxPool2d)
@meta_register.register(torch.nn.MaxPool3d)
def maxpool_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """Meta info for MaxPool
    The aten graph of MaxPool is
    graph():
    %input_2 : [#users=2] = placeholder[target=placeholder](default=)
    %max_pool2d_with_indices_default : [#users=2] = call_function[target=torch.ops.aten.max_pool2d_with_indices.default](args = (%input_2, [None, None], [None, None]), kwargs = {})
    %zeros_like_default : [#users=1] = call_function[target=torch.ops.aten.zeros_like.default](args = (%max_pool2d_with_indices_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%input_2,), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%max_pool2d_with_indices_default,), kwargs = {})
    %max_pool2d_with_indices_backward_default : [#users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%zeros_like_default, %detach_default, [None, None], [None, None], [None, None], [None, None], None, %detach_default_1), kwargs = {})
    %detach_default_2 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%max_pool2d_with_indices_backward_default,), kwargs = {})
    %detach_default_3 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_2,), kwargs = {})

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    input_tensor = next(filter(lambda x: x.type == OperationDataType.ARG, args)).data
    output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data

    # construct forward args for flop mapping
    fwd_in_args = [input_tensor]
    fwd_out_args = [output_tensor]

    # construct backward args for flop mapping
    bwd_in_args = [output_tensor]
    bwd_out_args = [input_tensor]

    # construct index matrix
    index_matrix = torch.zeros_like(output_tensor, device="meta", dtype=torch.int64)

    # calculate cost
    # the fwd op with compute cost is max_pool2d_with_indices.default
    # the bwd op with compute cost is max_pool2d_with_indices_backward.default

    # calculate compute cost
    fwd_compute_cost = flop_mapping[torch.ops.aten.max_pool2d_with_indices.default](fwd_in_args, fwd_out_args)
    bwd_compute_cost = flop_mapping[torch.ops.aten.max_pool2d_with_indices_backward.default](bwd_in_args, bwd_out_args)
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # calculate memory cost
    # NOTE: the index matrix will be discarded in backward phase
    # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
    fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes([input_tensor, output_tensor, index_matrix]))

    # temp memory for backward is the index matrix to be discarded
    bwd_mem_cost = MemoryCost(
        activation=compute_size_in_bytes(input_tensor) - compute_size_in_bytes(index_matrix),
        temp=compute_size_in_bytes(index_matrix),
    )

    # total cost
    total_mem_cost = MemoryCost(activation=fwd_mem_cost.activation + bwd_mem_cost.activation, temp=bwd_mem_cost.temp)

    mem_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = [torch.zeros_like(input_tensor, device="meta")]
    fwd_buffer = [torch.zeros_like(index_matrix, device="meta")]
    fwd_out = [torch.zeros_like(output_tensor, device="meta")]

    return compute_cost, mem_cost, fwd_in, fwd_buffer, fwd_out
