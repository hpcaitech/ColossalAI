from typing import List, Tuple

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem
from colossalai.fx.profiler.memory_utils import activation_size
from colossalai.fx.profiler.opcount import flop_mapping

from ..registry import meta_register

__all__ = ["relu_meta_info"]


@meta_register.register(torch.nn.ReLU)
def relu_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.nn.ReLU metainfo generator
    The aten graph of torch.nn.ReLU is
    graph():
    %input_2 : [#users=1] = placeholder[target=placeholder](default=)
    %relu_default : [#users=2] = call_function[target=torch.ops.aten.relu.default](args = (%input_2,), kwargs = {})
    %zeros_like_default : [#users=1] = call_function[target=torch.ops.aten.zeros_like.default](args = (%relu_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%relu_default,), kwargs = {})
    %threshold_backward_default : [#users=1] = call_function[target=torch.ops.aten.threshold_backward.default](args = (%zeros_like_default, %detach_default, None), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%threshold_backward_default,), kwargs = {})
    %detach_default_2 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_1,), kwargs = {})

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    input_tensor = next(filter(lambda x: x.type == OperationDataType.ARG, args)).data
    output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
    inplace = kwargs.get("inplace", False)

    # construct input args for forward
    fwd_in_args = [input_tensor]

    # construct input args for backward
    bwd_in_args = [output_tensor]

    # calculate cost
    # the fwd op with compute cost is relu.default
    # the bwd op with compute cost is threshold_backward

    # calculate compute cost
    fwd_compute_cost = flop_mapping[torch.ops.aten.relu.default](fwd_in_args, (output_tensor,))
    bwd_compute_cost = flop_mapping[torch.ops.aten.threshold_backward.default](bwd_in_args, (input_tensor,))
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # calculate memory cost
    # NOTE: the inplace ReLU don't have forward memory cost
    fwd_memory_cost = MemoryCost(activation=0 if inplace else activation_size(output_tensor),
                                 parameter=0,
                                 temp=0,
                                 buffer=0)

    bwd_memory_cost = MemoryCost(activation=activation_size(input_tensor), parameter=0, temp=0, buffer=0)

    # total cost is the sum of forward and backward cost
    total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
                            parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter)

    memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    # store fwd_in
    fwd_in = [input_tensor]

    return compute_cost, memory_cost, fwd_in
