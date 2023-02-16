from typing import Callable, List, Tuple

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem
from colossalai.fx.profiler.memory_utils import activation_size
from colossalai.fx.profiler.opcount import elementwise_flop_counter

from ..registry import meta_register

# __all__ = ["relu_meta_info"]
__all__ = ["elementwise_meta_info"]

# @meta_register.register(torch.nn.ReLU)
# def relu_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
#     """torch.nn.ReLU metainfo generator
#     The aten graph of torch.nn.ReLU is
#     graph():
#     %input_2 : [#users=1] = placeholder[target=placeholder](default=)
#     %relu_default : [#users=2] = call_function[target=torch.ops.aten.relu.default](args = (%input_2,), kwargs = {})
#     %zeros_like_default : [#users=1] = call_function[target=torch.ops.aten.zeros_like.default](args = (%relu_default,), kwargs = {dtype: None, layout: None, device: None, pin_memory: None})
#     %detach_default : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%relu_default,), kwargs = {})
#     %threshold_backward_default : [#users=1] = call_function[target=torch.ops.aten.threshold_backward.default](args = (%zeros_like_default, %detach_default, None), kwargs = {})
#     %detach_default_1 : [#users=1] = call_function[target=torch.ops.aten.detach.default](args = (%threshold_backward_default,), kwargs = {})
#     %detach_default_2 : [#users=0] = call_function[target=torch.ops.aten.detach.default](args = (%detach_default_1,), kwargs = {})

#     Returns:
#         Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
#     """

#     input_tensor = args[0].data
#     output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
#     is_inplace = kwargs.get("inplace", False)

#     # construct input args for forward
#     fwd_in_args = [input_tensor]

#     # construct input args for backward
#     bwd_in_args = [output_tensor]

#     # calculate cost
#     # the fwd op with compute cost is relu.default
#     # the bwd op with compute cost is threshold_backward

#     # calculate compute cost
#     fwd_compute_cost = flop_mapping[torch.ops.aten.relu.default](fwd_in_args, (output_tensor,))
#     bwd_compute_cost = flop_mapping[torch.ops.aten.threshold_backward.default](bwd_in_args, (input_tensor,))
#     compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

#     # calculate memory cost
#     # NOTE: the inplace ReLU don't have forward memory cost
#     # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
#     fwd_memory_cost = MemoryCost(
#         activation=activation_size(input_tensor) if is_inplace else activation_size([output_tensor, input_tensor]),
#         parameter=0,
#         temp=0,
#         buffer=0)

#     bwd_memory_cost = MemoryCost(activation=activation_size(input_tensor), parameter=0, temp=0, buffer=0)

#     # total cost is the sum of forward and backward cost
#     total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
#                             parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter)

#     memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

#     # store fwd_in, fwd_buffer, fwd_out
#     # NOTE: It might seems a little bit weird here, we just want to align it with the older version
#     # of MetaInfoProp. In the future we might modify this part to make it clearer.
#     fwd_in = []
#     fwd_buffer = [torch.zeros_like(output_tensor, device='meta')]
#     fwd_out = [torch.zeros_like(output_tensor, device='meta')]

#     return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out

# @meta_register.register(torch.nn.Softmax)
# @meta_register.register(torch.nn.functional.softmax)
# def softmax_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
#     """torch.nn.Softmax metainfo generator
#     Returns:
#         Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
#     """
#     input_tensor = next(
#         filter(
#             lambda x:
#             (x.type == OperationDataType.ARG or x.type == OperationDataType.PARAM) and x.name != 'softmax_dim',
#             args)).data
#     output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
#     softmax_dim = next(filter(lambda x: x.name == 'softmax_dim', args)).data

#     # calculate cost

#     # calculate compute cost
#     fwd_compute_cost = flop_mapping[torch.ops.aten._softmax.default]([input_tensor], [output_tensor])
#     bwd_compute_cost = flop_mapping[torch.ops.aten._softmax_backward_data.default]([output_tensor], [input_tensor])

#     compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

#     # calculate memory cost
#     # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
#     fwd_memory_cost = MemoryCost(activation=activation_size([input_tensor, output_tensor]),
#                                  parameter=0,
#                                  temp=0,
#                                  buffer=0)
#     bwd_memory_cost = MemoryCost(activation=activation_size(input_tensor),
#                                  parameter=0,
#                                  temp=activation_size(input_tensor),
#                                  buffer=0)

#     # total cost is the sum of forward and backward cost
#     total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
#                             parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter,
#                             temp=fwd_memory_cost.temp + bwd_memory_cost.temp,
#                             buffer=fwd_memory_cost.buffer + bwd_memory_cost.buffer)

#     memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

#     # store fwd_in, fwd_buffer, fwd_out
#     fwd_in = []
#     fwd_buffer = [torch.zeros_like(output_tensor, device='meta')]
#     fwd_out = [torch.zeros_like(output_tensor, device='meta')]

#     return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out


def elementwise_meta_info(temp_mem_scale: float = 0) -> Callable:
    """This is a function to create the meta information generator for elementwise operations

    Args:
        temp_mem_scale (float, optional): temp memory scaling factor. Defaults to 0.

    Returns:
        Callable: meta information generator
    """

    def meta_func(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
        input_tensor = next(
            filter(
                lambda x:
                (x.type == OperationDataType.ARG or x.type == OperationDataType.PARAM) and x.name != 'softmax_dim',
                args)).data
        output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
        is_inplace = 1 if kwargs.get('inplace', False) else 0

        flop_counter = elementwise_flop_counter(1, 0)
        # calculate compute cost
        fwd_compute_cost = flop_counter([input_tensor], [output_tensor])
        bwd_compute_cost = flop_counter([output_tensor], [input_tensor])

        compute_cost = TrainCycleItem(fwd=fwd_compute_cost,
                                      bwd=bwd_compute_cost,
                                      total=fwd_compute_cost + bwd_compute_cost)

        # calculate memory cost
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        # NOTE: if in_place is True, we will not create a new tensor in forward
        fwd_memory_cost = MemoryCost(activation=activation_size(input_tensor) * (2 - is_inplace),
                                     parameter=0,
                                     temp=0,
                                     buffer=0)

        # temp_mem_scale is for situation like softmax backward
        bwd_memory_cost = MemoryCost(activation=activation_size(input_tensor),
                                     parameter=0,
                                     temp=activation_size(input_tensor) * temp_mem_scale,
                                     buffer=0)

        # total cost is the sum of forward and backward cost
        total_cost = MemoryCost(activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
                                parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter,
                                temp=fwd_memory_cost.temp + bwd_memory_cost.temp,
                                buffer=fwd_memory_cost.buffer + bwd_memory_cost.buffer)

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

        # store fwd_in, fwd_buffer, fwd_out
        fwd_in = []
        fwd_buffer = [torch.zeros_like(output_tensor, device='meta')]
        fwd_out = [torch.zeros_like(output_tensor, device='meta')]

        return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out

    return meta_func


# the following elementwise ops doesn't have temp memory during backward
zero_temp_mem_ops = [torch.nn.ReLU, torch.nn.functional.relu, torch.tanh]

# the following elementwise ops have temp memory the same size as input during backward
one_temp_mem_ops = [torch.nn.Softmax, torch.nn.functional.softmax]

# register meta information
meta_register.register(zero_temp_mem_ops)(elementwise_meta_info())
meta_register.register(one_temp_mem_ops)(elementwise_meta_info(1))
