from typing import Callable, Dict, List, Tuple, Union

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    MemoryCost,
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
    TrainCycleItem,
)
from colossalai.fx.profiler.memory_utils import activation_size
from colossalai.fx.profiler.opcount import flop_mapping
from colossalai.tensor.sharding_spec import ShardingSpec

from ..registry import meta_register

__all__ = ['linear_meta_info']


@meta_register.register(torch.nn.functional.linear)
@meta_register.register(torch.nn.Linear)
def linear_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.nn.Linear & torch.nn.functional.linear meta info generator
    NOTE: currently we separate the bias part from the biased linear ops, we will consider the memory consumption in add metainfo generator,
    but we will hold the bias mechanism in the linear metainfo generator for future use.

    graph():
    %input_2 : [#users=2] = placeholder[target=placeholder](default=)
    %addmm_default : [#users=1] = call_function[target=torch.ops.aten.addmm.default](args = (None, %input_2, None), kwargs = {})
    %zeros_like_default : [#users=3] = call_function[target=torch.ops.aten.zeros_like.default](args = (%addmm_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%input_2,), kwargs = {})
    %mm_default : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%zeros_like_default, None), kwargs = {})
    %t_default : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%zeros_like_default,), kwargs = {})
    %mm_default_1 : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t_default, %detach_default), kwargs = {})
    %t_default_1 : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%mm_default_1,), kwargs = {})
    %sum_dim_int_list : [#users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%zeros_like_default, [None], None), kwargs = {})
    %view_default : [#users=1] = call_function[target=torch.ops.aten.view.default](args = (%sum_dim_int_list, [None]), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%view_default,), kwargs = {})
    %detach_default_2 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_1,), kwargs = {})
    %detach_default_3 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%mm_default,), kwargs = {})
    %detach_default_4 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_3,), kwargs = {})
    %t_default_2 : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%t_default_1,), kwargs = {})
    %detach_default_5 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%t_default_2,), kwargs = {})
    %detach_default_6 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_5,), kwargs = {})

    The one without bias is
    graph():
    %input_2 : [#users=2] = placeholder[target=placeholder](default=)
    %mm_default : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%input_2, None), kwargs = {})
    %zeros_like_default : [#users=2] = call_function[target=torch.ops.aten.zeros_like.default](args = (%mm_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%input_2,), kwargs = {})
    %t_default : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%zeros_like_default,), kwargs = {})
    %mm_default_1 : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%t_default, %detach_default), kwargs = {})
    %t_default_1 : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%mm_default_1,), kwargs = {})
    %mm_default_2 : [#users=1] = call_function[target=torch.ops.aten.mm.default](args = (%zeros_like_default, None), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%mm_default_2,), kwargs = {})
    %detach_default_2 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_1,), kwargs = {})
    %t_default_2 : [#users=1] = call_function[target=torch.ops.aten.t.default](args = (%t_default_1,), kwargs = {})
    %detach_default_3 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%t_default_2,), kwargs = {})
    %detach_default_4 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_3,), kwargs = {})

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, bool]: compute cost, memory cost and forward inputs
    """

    has_bias: bool = False

    input_tensor = args[0].data
    output_tensor = args[2].data
    if len(args) == 4:
        weight_tensors = [args[1].data, args[3].data]
    else:
        weight_tensors = [args[1].data]

    # process the dimension of input and output
    if len(input_tensor.shape) > 2:
        input_tensor: torch.Tensor
        input_tensor = input_tensor.view(-1, input_tensor.shape[-1])

    if len(output_tensor.shape) > 2:
        output_tensor: torch.Tensor
        output_tensor = output_tensor.view(-1, output_tensor.shape[-1])

    if len(weight_tensors) > 1:
        has_bias = True
        if len(weight_tensors[0].shape) == 2:
            weight_tensor, bias_tensor = weight_tensors
        else:
            bias_tensor, weight_tensor = weight_tensors
    else:
        weight_tensor = weight_tensors[0]

    if has_bias:
        # calculate cost with bias
        # the fwd op with compute cost is addmm
        # the bwd op with compute cost is mm * 2 and sum.dim_IntList

        # calculate compute cost
        fwd_compute_cost = flop_mapping[torch.ops.aten.addmm.default](
            [bias_tensor, input_tensor, torch.transpose(weight_tensor, 0, 1)], (output_tensor,))
        bwd_compute_cost = flop_mapping[torch.ops.aten.mm.default]([output_tensor, weight_tensor], (input_tensor,)) + \
                           flop_mapping[torch.ops.aten.mm.default]([torch.transpose(output_tensor, 0, 1), input_tensor], (weight_tensor,)) + \
                           flop_mapping[torch.ops.aten.sum.dim_IntList]([output_tensor], (bias_tensor,))
        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)

        # calculate memory cost
        # NOTE: Linear don't have buffer and temp in forward and backward phase
        # the forward activation cost is the size of output_tensor, parameter cost is the size of weight_tensor and bias_tensor
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        fwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, output_tensor]),
                                     parameter=activation_size([weight_tensor, bias_tensor]),
                                     temp=0,
                                     buffer=0)

        # the backward activation cost is the size of input_tensor, weight_tensor and bias_tensor, parameter cost is 0
        bwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, weight_tensor, bias_tensor]),
                                     parameter=activation_size([weight_tensor, bias_tensor]),
                                     temp=0,
                                     buffer=0)

        # total cost is to sum the forward and backward cost
        total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
                                parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter)

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    else:
        # calculate cost without bias
        # the fwd op with compute cost is mm
        # the bwd op with compute cost is mm * 2

        # calculate compute cost
        fwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [input_tensor, torch.transpose(weight_tensor, 0, 1)], (output_tensor,))
        bwd_compute_cost = flop_mapping[torch.ops.aten.mm.default]([output_tensor, weight_tensor], (input_tensor,)) + \
                           flop_mapping[torch.ops.aten.mm.default]([torch.transpose(output_tensor, 0, 1), input_tensor], (weight_tensor,))

        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)

        # calculate memory cost
        # NOTE: Linear don't have buffer and temp in forward and backward phase
        # the forward activation cost is the size of output_tensor, parameter cost is the size of weight_tensor
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        fwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, output_tensor]),
                                     parameter=activation_size(weight_tensor),
                                     temp=0,
                                     buffer=0)

        # the backward activation cost is the size of input_tensor and weight_tensor, parameter cost is 0
        bwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, weight_tensor]),
                                     parameter=activation_size(weight_tensor),
                                     temp=0,
                                     buffer=0)

        # total cost is to sum the forward and backward cost
        total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
                                parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter)

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = [torch.zeros_like(input_tensor, device='meta')]
    fwd_buffer = []
    fwd_out = [torch.zeros_like(output_tensor, device='meta')]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
