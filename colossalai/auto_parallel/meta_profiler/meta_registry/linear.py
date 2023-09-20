from functools import reduce
from typing import List, Tuple

import torch

from colossalai._analyzer._subclasses.flop_tensor import flop_mapping
from colossalai._analyzer.fx.node_util import compute_size_in_bytes
from colossalai.auto_parallel.tensor_shard.sharding_strategy import MemoryCost, TrainCycleItem

from ..registry import meta_register

__all__ = ["linear_meta_info", "matmul_meta_info"]


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
            [bias_tensor, input_tensor, torch.transpose(weight_tensor, 0, 1)], (output_tensor,)
        )
        bwd_compute_cost = (
            flop_mapping[torch.ops.aten.mm.default]([output_tensor, weight_tensor], (input_tensor,))
            + flop_mapping[torch.ops.aten.mm.default](
                [torch.transpose(output_tensor, 0, 1), input_tensor], (weight_tensor,)
            )
            + flop_mapping[torch.ops.aten.sum.dim_IntList]([output_tensor], (bias_tensor,))
        )
        compute_cost = TrainCycleItem(
            fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost
        )

        # calculate memory cost
        # NOTE: Linear don't have buffer and temp in forward and backward phase
        # the forward activation cost is the size of output_tensor, parameter cost is the size of weight_tensor and bias_tensor
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        fwd_memory_cost = MemoryCost(
            activation=compute_size_in_bytes([input_tensor, output_tensor]),
            parameter=compute_size_in_bytes([weight_tensor, bias_tensor]),
            temp=0,
            buffer=0,
        )

        # the backward activation cost is the size of input_tensor, weight_tensor and bias_tensor, parameter cost is 0
        bwd_memory_cost = MemoryCost(
            activation=compute_size_in_bytes([input_tensor, weight_tensor, bias_tensor]),
            parameter=compute_size_in_bytes([weight_tensor, bias_tensor]),
            temp=0,
            buffer=0,
        )

        # total cost is to sum the forward and backward cost
        total_cost = MemoryCost(
            activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
            parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter,
        )

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    else:
        # calculate cost without bias
        # the fwd op with compute cost is mm
        # the bwd op with compute cost is mm * 2

        # calculate compute cost
        fwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [input_tensor, torch.transpose(weight_tensor, 0, 1)], (output_tensor,)
        )
        bwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [output_tensor, weight_tensor], (input_tensor,)
        ) + flop_mapping[torch.ops.aten.mm.default](
            [torch.transpose(output_tensor, 0, 1), input_tensor], (weight_tensor,)
        )

        compute_cost = TrainCycleItem(
            fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost
        )

        # calculate memory cost
        # NOTE: Linear don't have buffer and temp in forward and backward phase
        # the forward activation cost is the size of output_tensor, parameter cost is the size of weight_tensor
        # NOTE: currently in SPMD solver we always believe that there will be a new tensor created in forward
        fwd_memory_cost = MemoryCost(
            activation=compute_size_in_bytes([input_tensor, output_tensor]),
            parameter=compute_size_in_bytes(weight_tensor),
            temp=0,
            buffer=0,
        )

        # the backward activation cost is the size of input_tensor and weight_tensor, parameter cost is 0
        bwd_memory_cost = MemoryCost(
            activation=compute_size_in_bytes([input_tensor, weight_tensor]),
            parameter=compute_size_in_bytes(weight_tensor),
            temp=0,
            buffer=0,
        )

        # total cost is to sum the forward and backward cost
        total_cost = MemoryCost(
            activation=fwd_memory_cost.activation + bwd_memory_cost.activation,
            parameter=fwd_memory_cost.parameter + bwd_memory_cost.parameter,
        )

        memory_cost = TrainCycleItem(fwd=fwd_memory_cost, bwd=bwd_memory_cost, total=total_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = [torch.zeros_like(input_tensor, device="meta")]
    fwd_buffer = []
    fwd_out = [torch.zeros_like(output_tensor, device="meta")]

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out


@meta_register.register(torch.matmul)
def matmul_meta_info(*args, **kwargs) -> Tuple[TrainCycleItem, TrainCycleItem, List[torch.Tensor]]:
    """torch.matmul meta info generator
    There are several cases for torch.matmul:
    1. Vector-vector multiplication => no temp memory, forward memory cost is 1 element (could be neglected), backward memory cost is the same
    as two input vectors.
    2. Matrix-vector multiplication => if the first input is matrix, no temp memory is needed, otherwise, there is a temp memory in the backward
    phase for the transpose of the matrix. The forward memory cost is the size of output tensor, backward memory cost is the size of the two inputs; if
    the first input is vector, the forward memory cost is the size of the output tensor, and during the backward phase, it will allocate a temp memory
    the same size as the input matrix, and allocate memory for the gradient of two inputs.
    3. Batched Matrix-vector multiplication => if the first input is the batched matrix, no temp memory, the forward memory cost is the size of
    output tensor, backward memory cost is the size of the two inputs; if the second input is the batched matrix, the matmul will allocate memory for
    the gradient of the batched matrix in the forward phase (as they create a new tensor without the former batches), so the forward memory cost is
    the output tensor and the newly created matrix (take the same amount of memory of the input batched matrix). During the backward phase, it will
    allocate a temp memory the same size as input batched matrix, and allocate a tensor for the gradient of the input vector. The gradient of the batched
    matrix will be stored in the memory allocated during the forward phase.
    3. Matrix-matrix multiplication => no temp memory, forward memory is the size of output tensor, backward memory is the size of the two inputs
    4. Batched matrix-matrix multiplication => if the first input is the batched matrix, no temp memory, the forward memory cost is the size of two
    inputs and backward memory cost is the size of the output tensor; if the second input is the batched matrix, during the forward phase it will allocate
    memory for the output and gradient of the second input, and has a temp memory the same size as the output, during the backward phase, it
    will allocate memory for the gradient of the first input and has a temp memory which is as big as output and the second input.
    5. Batched matrix-batched matrix multiplication => if the two inputs have the same batch dimensions, no temp memory, the forward memory cost is the size
    of output, backward memory cost is the size of the two inputs; it the two inputs have different batch dimensions, during the forward phase it will allocate
    memory of the expanded inputs (so that the batch dimensions could match) and the output, and during the backward phase, it has a temp memory of the size of
    two expanded inputs, and it will allocate memory for the gradient of the two inputs and discard the expanded inputs allocated during the forward phase.

    Returns:
        Tuple[TrainCycleItem, TrainCycleItem, bool]: compute cost, memory cost and forward inputs

    """
    # Get input and output tensors
    input_tensors = [args[0].data, args[1].data]
    output_tensors = [args[-1].data]

    # Check dimension
    if all(len(tensor.shape) == 1 for tensor in input_tensors):
        # Dot
        fwd_compute_cost = flop_mapping[torch.ops.aten.matmul.default](input_tensors, output_tensors)
        bwd_compute_cost = flop_mapping[torch.ops.aten.mul.Tensor](input_tensors[0], output_tensors) * 2

        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors), parameter=0, temp=0, buffer=0)
        bwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(input_tensors), parameter=0, temp=0, buffer=0)

    elif len(input_tensors[0].shape) >= 2 and len(input_tensors[1].shape) == 1:
        # gemv case 1: matrix-vector multiplication
        # &
        # batched gemv case 1: batched matrix-vector multiplication

        fwd_compute_cost = flop_mapping[torch.ops.aten.matmul.default](
            [input_tensors[0].reshape(-1, input_tensors[0].shape[-1]), input_tensors[1]], output_tensors
        )

        # combine the dimensions of output
        bwd_compute_cost = flop_mapping[torch.ops.aten.mul.Tensor](
            [output_tensors[0].reshape(-1), input_tensors[1]], output_tensors
        ) + flop_mapping[torch.ops.aten.matmul.default](
            [input_tensors[0].reshape(-1, input_tensors[0].shape[-1]).transpose(0, 1), output_tensors[0].reshape(-1)],
            output_tensors,
        )

        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors), parameter=0, temp=0, buffer=0)
        bwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(input_tensors), parameter=0, temp=0, buffer=0)

    elif len(input_tensors[0].shape) == 1 and len(input_tensors[1].shape) == 2:
        # gemv case 2: vector-matrix multiplication
        fwd_compute_cost = flop_mapping[torch.ops.aten.matmul.default](input_tensors, output_tensors)

        bwd_compute_cost = flop_mapping[torch.ops.aten.mul.Tensor](
            [output_tensors[0], input_tensors[0]], output_tensors
        ) + flop_mapping[torch.ops.aten.matmul.default]([input_tensors[1], output_tensors[0]], output_tensors)

        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors), parameter=0, temp=0, buffer=0)
        bwd_mem_cost = MemoryCost(
            activation=compute_size_in_bytes(input_tensors),
            parameter=0,
            temp=compute_size_in_bytes(input_tensors[1]),
            buffer=0,
        )

    elif len(input_tensors[0].shape) == 1 and len(input_tensors[1].shape) >= 3:
        # batched gemv case 2: vector-batched matrix multiplication

        fwd_compute_cost = flop_mapping[torch.ops.aten.matmul.default](
            [input_tensors[1].transpose(-2, -1).reshape(-1, input_tensors[1].shape[-2]), input_tensors[0]],
            [output_tensors[0].reshape(-1)],
        )

        # combine the dimensions of output
        bwd_compute_cost = flop_mapping[torch.ops.aten.mul.Tensor](
            [output_tensors[0].reshape(-1), input_tensors[0]], output_tensors
        ) + flop_mapping[torch.ops.aten.matmul.default](
            [
                input_tensors[1].transpose(-2, -1).reshape(-1, input_tensors[1].shape[-2]).transpose(0, 1),
                output_tensors[0].reshape(-1),
            ],
            output_tensors,
        )

        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors + [input_tensors[1]]))
        bwd_mem_cost = MemoryCost(
            activation=compute_size_in_bytes(input_tensors[0]),
            parameter=0,
            temp=compute_size_in_bytes(input_tensors[1]),
            buffer=0,
        )

    elif len(input_tensors[0].shape) >= 2 and len(input_tensors[1].shape) == 2:
        # gemm & batched gemm case 1: batched matrix-matrix multiplication

        fwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [input_tensors[0].reshape(-1, input_tensors[0].shape[-1]), input_tensors[1]],
            [output_tensors[0].reshape(-1, output_tensors[0].shape[-1])],
        )

        bwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [
                input_tensors[0].reshape(-1, input_tensors[0].shape[-1]).transpose(0, 1),
                output_tensors[0].reshape(-1, output_tensors[0].shape[-1]),
            ],
            [input_tensors[1]],
        ) + flop_mapping[torch.ops.aten.mm.default](
            [output_tensors[0].reshape(-1, output_tensors[0].shape[-1]), input_tensors[1].transpose(0, 1)],
            [input_tensors[0].reshape(-1, input_tensors[0].shape[-1])],
        )

        fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors), parameter=0, temp=0, buffer=0)
        bwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(input_tensors), parameter=0, temp=0, buffer=0)

    elif len(input_tensors[0].shape) == 2 and len(input_tensors[1].shape) >= 3:
        # batched gemm case 2: matrix-batched matrix multiplication
        fwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [
                input_tensors[1].transpose(-2, -1).reshape(-1, input_tensors[1].shape[-2]),
                input_tensors[0].transpose(0, 1),
            ],
            [output_tensors[0].transpose(-2, -1)],
        )

        bwd_compute_cost = flop_mapping[torch.ops.aten.mm.default](
            [
                output_tensors[0].transpose(-2, -1).reshape(-1, output_tensors[0].shape[-2]).transpose(0, 1),
                input_tensors[1].transpose(-2, -1).reshape(-1, input_tensors[1].shape[-2]),
            ],
            [input_tensors[0]],
        ) + flop_mapping[torch.ops.aten.mm.default](
            [output_tensors[0].transpose(-2, -1).reshape(-1, output_tensors[0].shape[-2]), input_tensors[0]],
            [input_tensors[1].transpose(-2, -1).reshape(-1, input_tensors[1].shape[-2])],
        )

        fwd_mem_cost = MemoryCost(
            activation=compute_size_in_bytes(output_tensors) + compute_size_in_bytes(input_tensors[1]),
            temp=compute_size_in_bytes(output_tensors),
        )
        bwd_mem_cost = MemoryCost(
            activation=compute_size_in_bytes(input_tensors[0]),
            parameter=0,
            temp=compute_size_in_bytes(input_tensors[1]) + compute_size_in_bytes(output_tensors),
        )

    elif all(len(tensor.shape) >= 3 for tensor in input_tensors):
        # Batched matrix-batched matrix multiplication
        # Fetch shape of the two inputs and see if the batch dimensions are the same
        _is_batch_dims_same = True
        if len(input_tensors[0].shape) == len(input_tensors[1].shape):
            for shape_0, shape_1 in zip(input_tensors[0].shape[:-2], input_tensors[1].shape[:-2]):
                if shape_0 != shape_1:
                    _is_batch_dims_same = False
                    break
        else:
            _is_batch_dims_same = False

        # retrieve dimensions
        input_dim_00 = input_tensors[0].shape[-2]
        input_dim_01 = input_tensors[0].shape[-1]
        input_dim_10 = input_tensors[1].shape[-2]
        input_dim_11 = input_tensors[1].shape[-1]
        output_dim_0 = output_tensors[0].shape[-2]
        output_dim_1 = output_tensors[0].shape[-1]

        if _is_batch_dims_same:
            # Case 1: batch dimensions are the same

            # Forward compute cost: C = A * B
            fwd_compute_cost = flop_mapping[torch.ops.aten.bmm.default](
                [
                    input_tensors[0].reshape(-1, input_dim_00, input_dim_01),
                    input_tensors[1].reshape(-1, input_dim_10, input_dim_11),
                ],
                [output_tensors[0].reshape(-1, output_dim_0, output_dim_1)],
            )

            # Backward compute cost: dB = A^T * dC, dA = dC * B^T
            bwd_compute_cost = flop_mapping[torch.ops.aten.bmm.default](
                [
                    input_tensors[0].transpose(-2, -1).reshape(-1, input_dim_01, input_dim_00),
                    output_tensors[0].reshape(-1, output_dim_0, output_dim_1),
                ],
                [input_tensors[1].reshape(-1, input_dim_11, input_dim_10)],
            ) + flop_mapping[torch.ops.aten.bmm.default](
                [
                    output_tensors[0].reshape(-1, output_dim_0, output_dim_1),
                    input_tensors[1].transpose(-2, -1).reshape(-1, input_dim_11, input_dim_10),
                ],
                [input_tensors[0].reshape(-1, input_dim_00, input_dim_01)],
            )

            fwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(output_tensors))
            bwd_mem_cost = MemoryCost(activation=compute_size_in_bytes(input_tensors))

        else:
            # Case 2: batch dimensions are different
            batch_dims = output_tensors[0].shape[:-2]
            extended_input_0 = torch.rand(
                reduce(lambda x, y: x * y, batch_dims), input_dim_00, input_dim_01, device="meta"
            )
            extended_input_1 = torch.rand(
                reduce(lambda x, y: x * y, batch_dims), input_dim_10, input_dim_11, device="meta"
            )

            # Forward compute cost: C = A * B
            fwd_compute_cost = flop_mapping[torch.ops.aten.bmm.default](
                [extended_input_0, extended_input_1], [output_tensors[0].reshape(-1, output_dim_0, output_dim_1)]
            )

            # Backward compute cost: dB = A^T * dC, dA = dC * B^T
            bwd_compute_cost = flop_mapping[torch.ops.aten.bmm.default](
                [extended_input_0.transpose(-2, -1), output_tensors[0].reshape(-1, output_dim_0, output_dim_1)],
                [extended_input_1],
            ) + flop_mapping[torch.ops.aten.bmm.default](
                [output_tensors[0].reshape(-1, output_dim_0, output_dim_1), extended_input_1.transpose(-2, -1)],
                [extended_input_0],
            )

            fwd_mem_cost = MemoryCost(
                activation=compute_size_in_bytes([output_tensors[0], extended_input_0, extended_input_1])
            )
            bwd_mem_cost = MemoryCost(
                activation=compute_size_in_bytes(input_tensors)
                - compute_size_in_bytes([extended_input_0, extended_input_1]),
                temp=compute_size_in_bytes([extended_input_0, extended_input_1]),
            )

    # compute cost
    compute_cost = TrainCycleItem(fwd=fwd_compute_cost, bwd=bwd_compute_cost, total=fwd_compute_cost + bwd_compute_cost)

    # memory cost
    total_cost = MemoryCost(
        activation=fwd_mem_cost.activation + bwd_mem_cost.activation,
        parameter=fwd_mem_cost.parameter + bwd_mem_cost.parameter,
        temp=fwd_mem_cost.temp + bwd_mem_cost.temp,
        buffer=fwd_mem_cost.buffer + bwd_mem_cost.buffer,
    )

    memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_cost)

    # store fwd_in, fwd_buffer, fwd_out
    fwd_in = input_tensors
    fwd_buffer = []
    fwd_out = output_tensors

    return compute_cost, memory_cost, fwd_in, fwd_buffer, fwd_out
