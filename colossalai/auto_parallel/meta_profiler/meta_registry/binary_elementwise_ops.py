from typing import List, Tuple

import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, OperationDataType, TrainCycleItem
from colossalai.fx.profiler.memory_utils import activation_size
from colossalai.fx.profiler.opcount import flop_mapping

from ..constants import BCAST_FUNC_OP, NO_SAVE_ACTIVATION
from ..registry import meta_register

__all__ = ['binary_elementwise_meta_info']


@meta_register.register(BCAST_FUNC_OP)
def binary_elementwise_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """Meta information generator for binary elementwise operations
    NOTE: Some of the binary elementwise operations will discard the input activation after computation, as they
    don't need those tensors for back propagation, for example, if there are two tensors being sent for `torch.add`,
    they will be discarded right after add operation is done. We create a simple API in `MetaInfo` class to identify
    this behavior, it is critical for better memory estimation.

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]: compute cost, memory cost and forward inputs
    """

    input_op_data = [arg for arg in args if arg.type != OperationDataType.OUTPUT]
    output_op_data = next(filter(lambda arg: arg.type == OperationDataType.OUTPUT, args))

    # construct forward args for flop mapping
    fwd_in_args = [opdata.data for opdata in input_op_data]
    fwd_out_args = [output_op_data.data]

    # calculate cost

    # calculate compute cost
    # NOTE: we set bwd_compute_cost two times of fwd_compute_cost in this case
    fwd_compute_cost = flop_mapping[torch.ops.aten.add.Tensor](fwd_in_args, fwd_out_args)
    bwd_compute_cost = fwd_compute_cost * 2
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # calculate memory cost
    param_mem_cost = activation_size([arg.data for arg in input_op_data if arg.type == OperationDataType.PARAM])
    fwd_mem_cost = MemoryCost(
        activation=activation_size(output_op_data.data),
        parameter=param_mem_cost,
    )
    bwd_mem_cost = MemoryCost(
        activation=activation_size(fwd_in_args),
        parameter=param_mem_cost,
    )

    # total cost
    total_mem_cost = MemoryCost(
        activation=fwd_mem_cost.activation + bwd_mem_cost.activation,
        parameter=fwd_mem_cost.parameter + bwd_mem_cost.parameter,
    )

    memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = []
    fwd_buffer = []
    fwd_out = [torch.zeros_like(output_op_data.data, device='meta')]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
