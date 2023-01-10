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

__all__ = ['batchnormnd_meta_info']


@meta_register.register(torch.nn.BatchNorm1d)
@meta_register.register(torch.nn.BatchNorm2d)
@meta_register.register(torch.nn.BatchNorm3d)
def batchnormnd_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """BatchNorm1d, BatchNorm2d, BatchNorm3d, meta info generator
    The aten graph of BatchNorm2d is like

    graph():
    %input_2 : [#users=2] = placeholder[target=placeholder](default=)
    %cudnn_batch_norm_default : [#users=4] = call_function[target=torch.ops.aten.cudnn_batch_norm.default](args = (%input_2, None, None, None, None, None, None, None), kwargs = {})
    %zeros_like_default : [#users=1] = call_function[target=torch.ops.aten.zeros_like.default](args = (%cudnn_batch_norm_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
    %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%input_2,), kwargs = {})
    %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_default,), kwargs = {})
    %detach_default_2 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_default,), kwargs = {})
    %detach_default_3 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_default,), kwargs = {})
    %cudnn_batch_norm_backward_default : [#users=3] = call_function[target=torch.ops.aten.cudnn_batch_norm_backward.default](args = (%detach_default, %zeros_like_default, None, None, None, %detach_default_1, %detach_default_2, None, %detach_default_3), kwargs = {})
    %detach_default_4 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_backward_default,), kwargs = {})
    %detach_default_5 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_4,), kwargs = {})
    %detach_default_6 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_backward_default,), kwargs = {})
    %detach_default_7 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_6,), kwargs = {})
    %detach_default_8 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%cudnn_batch_norm_backward_default,), kwargs = {})
    %detach_default_9 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_8,), kwargs = {})
    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    input_tensor = args[0].data
    output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
    weight_tensor = next(filter(lambda x: x.name == "weight", args)).data
    bias_tensor = next(filter(lambda x: x.name == "bias", args)).data
    mean_tensor = next(filter(lambda x: x.name == "running_mean", args)).data
    var_tensor = next(filter(lambda x: x.name == "running_var", args)).data
    num_batch = next(filter(lambda x: x.name == "num_batches_tracked", args)).data

    # construct fwd args
    # the fwd inputs are input, weight, bias, running_mean, running_var and some other args
    # indicating the status of the module
    # the fwd outputs are output, saved mean, saved inv std and num batches tracked
    fwd_in_args = [input_tensor, weight_tensor, bias_tensor, mean_tensor, var_tensor, True, 0.1, 1e-5]
    fwd_out_args = [output_tensor, mean_tensor, var_tensor, num_batch]

    # construct bwd args
    # the bwd inputs are upstream grad, input, weight, running_mean, running_var, saved mean,
    # saved inv std and some other args indicating the status of the module
    # the bwd outputs are input grad, weight grad and bias grad
    bwd_in_args = [
        output_tensor, output_tensor, weight_tensor, mean_tensor, var_tensor, mean_tensor, var_tensor, 1e-5, num_batch
    ]
    bwd_out_args = [input_tensor, weight_tensor, bias_tensor]

    # calculate cost
    fwd_compute_cost = flop_mapping[torch.ops.aten.cudnn_batch_norm.default](fwd_in_args, fwd_out_args)
    bwd_compute_cost = flop_mapping[torch.ops.aten.cudnn_batch_norm_backward.default](bwd_in_args, bwd_out_args)
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # calculate memory cost
    # the fwd activation cost is output plus saved mean and saved inv std
    # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
    fwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, output_tensor, mean_tensor, var_tensor]),
                                 parameter=activation_size([weight_tensor, bias_tensor]),
                                 temp=0,
                                 buffer=activation_size([mean_tensor, var_tensor]))

    # the bwd memory cost is quite tricky here, BatchNorm will remove saved mean
    # and saved inv std during backward phase
    bwd_memory_cost = MemoryCost(activation=activation_size([input_tensor]),
                                 parameter=activation_size([weight_tensor, bias_tensor]),
                                 temp=activation_size([mean_tensor, var_tensor]),
                                 buffer=activation_size([mean_tensor, var_tensor]))

    # total cost is the sum of forward and backward cost
    total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
                            parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter)

    memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = [torch.zeros_like(input_tensor, device='meta')]
    fwd_buffer = [torch.zeros_like(mean_tensor, device='meta'), torch.zeros_like(var_tensor, device='meta')]
    fwd_out = [torch.zeros_like(output_tensor, device='meta')]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
