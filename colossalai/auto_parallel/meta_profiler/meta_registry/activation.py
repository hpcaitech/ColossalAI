from typing import Callable, List, Tuple

import torch

from colossalai._analyzer._subclasses.flop_tensor import ewise_flop_counter as elementwise_flop_counter
from colossalai._analyzer.fx.node_util import compute_size_in_bytes as activation_size
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem

from ..registry import meta_register

__all__ = ["elementwise_meta_info"]


def elementwise_meta_info(temp_mem_scale: float = 0, buffer_mem_scale: float = 0) -> Callable:
    """This is a function to create the meta information generator for elementwise operations

    Args:
        temp_mem_scale (float, optional): temp memory scaling factor for backward. Defaults to 0.
        buffer_mem_scale (float, optional): buffer memory scaling factor for forward. Defaults to 0.

    Returns:
        Callable: meta information generator
    """

    def meta_func(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
        input_tensor = next(
            filter(
                lambda x: (x.type == OperationDataType.ARG or x.type == OperationDataType.PARAM)
                and x.name != "softmax_dim",
                args,
            )
        ).data
        output_tensor = next(filter(lambda x: x.type == OperationDataType.OUTPUT, args)).data
        is_inplace = 1 if kwargs.get("inplace", False) else 0

        flop_counter = elementwise_flop_counter(1, 0)
        # calculate compute cost
        fwd_compute_cost = flop_counter([input_tensor], [output_tensor])
        bwd_compute_cost = flop_counter([output_tensor], [input_tensor])

        compute_cost = TrainCycleItem(
            fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost
        )

        # calculate memory cost
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        # NOTE: if in_place is True, we will not create a new tensor in forward
        fwd_memory_cost = MemoryCost(
            activation=activation_size(input_tensor) * (2 - is_inplace),
            parameter=0,
            temp=0,
            buffer=activation_size(input_tensor) * buffer_mem_scale,
        )

        # temp_mem_scale is for situation like softmax backward
        # the buffer will be removed during backward phase
        bwd_memory_cost = MemoryCost(
            activation=activation_size(input_tensor) - activation_size(input_tensor) * buffer_mem_scale,
            parameter=0,
            temp=activation_size(input_tensor) * temp_mem_scale + activation_size(input_tensor) * buffer_mem_scale,
            buffer=0,
        )

        # total cost is the sum of forward and backward cost
        total_cost = MemoryCost(
            activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
            parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter,
            temp=fwd_memory_cost.temp + bwd_memory_cost.temp,
            buffer=fwd_memory_cost.buffer + bwd_memory_cost.buffer,
        )

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

        # store fwd_in, fwd_buffer, fwd_out
        fwd_in = []
        fwd_buffer = [torch.zeros_like(output_tensor, device="meta")]
        fwd_out = [torch.zeros_like(output_tensor, device="meta")]

        return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out

    return meta_func


# register meta information
# (0, 0)
meta_register.register([torch.nn.ReLU, torch.nn.functional.relu, torch.tanh])(elementwise_meta_info(0, 0))

# (1, 0)
meta_register.register([torch.nn.Softmax, torch.nn.functional.softmax])(elementwise_meta_info(1, 0))

# (0, 0.25) for dropout, the buffer is in bool type so that the buffer memory cost is 0.25 times of input tensor
meta_register.register([torch.nn.Dropout, torch.nn.functional.dropout])(elementwise_meta_info(0, 0.25))
