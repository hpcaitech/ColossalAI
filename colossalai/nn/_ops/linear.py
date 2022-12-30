from copy import deepcopy
from typing import Optional

import torch.nn.functional as F

from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ReplicaSpec, ShardSpec
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor.sharding_spec import ShardingSpec

from ._utils import GeneralTensor, convert_to_colo_tensor, reduce_grad, reduce_input


def colo_linear_1drow(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    pg = weight.get_process_group()
    input_tensor = input_tensor.redistribute(ShardSpec([-1], [weight.get_tp_world_size()]), pg)

    # Output:P
    partial_output = F.linear(input_tensor, weight)
    # Reduce(Output)

    output = reduce_input(partial_output, pg)
    # Bias
    if bias is not None:
        assert not bias.has_compute_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias

    output = ColoTensor.from_torch_tensor(output, spec=ColoTensorSpec(pg, ReplicaSpec()))
    return output


def colo_linear_1dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    compute_spec = weight.compute_spec
    input_tensor = input_tensor.redistribute(ReplicaSpec())
    input_parallel = reduce_grad(input_tensor, weight.get_process_group())

    output_parallel = F.linear(input_parallel, weight, bias)
    output = ColoTensor.from_torch_tensor(output_parallel,
                                          spec=ColoTensorSpec(weight.get_process_group(),
                                                              ShardSpec([-1], [weight.get_tp_world_size()]),
                                                              ComputeSpec(ComputePattern.TP1D)))
    if compute_spec.output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_linear_1d(mode: str, input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    assert mode in ('row', 'col')
    funcs = {'row': colo_linear_1drow, 'col': colo_linear_1dcol}
    return funcs[mode](input_tensor, weight, bias)


# @register_colo_graph(input_pos=[1], param_pos=[2, 3])
def colo_linear_imp(input_tensor: GeneralTensor,
                    weight: GeneralTensor,
                    bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    assert isinstance(weight, ColoTensor)
    pg = weight.get_process_group()
    assert pg
    input_tensor = convert_to_colo_tensor(input_tensor, pg)
    bias = convert_to_colo_tensor(bias, pg)
    # input_tensor, weight, bias = tuple(map(convert_to_colo_tensor, (input_tensor, weight, bias)))

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.is_replicate(), 'Invalid weight spec for native Linear op'
        assert bias is None or bias.is_replicate(), 'Invalid bias spec for native Linear op'
        ret_tensor = ColoTensor.from_torch_tensor(F.linear(input_tensor, weight, bias), spec=ColoTensorSpec(pg))
    elif weight.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.is_shard_1dcol() and (bias is None or bias.is_replicate()):
            mode = 'row'
        elif weight.is_shard_1drow() and (bias is None or bias.is_shard_1drow() or bias.is_shard_1dcol()):
            mode = 'col'
        else:
            raise RuntimeError(f"the weight or bias tensor spec is not valid, weight {weight}, bias {bias}")
        ret_tensor = colo_linear_1d(mode, input_tensor, weight, bias)
    else:
        raise NotImplementedError

    return ret_tensor


def _new_colo_linear_imp(input_tensor: GeneralTensor,
                         weight: GeneralTensor,
                         bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    """
    A tentative function to compute the distributed linear layer with the latest sharding spec.
    This function is subject to future change as the current sharding API is not stable.
    """
    # get mesh info
    input_sharding_seq = input_tensor.sharding_spec.sharding_sequence
    weight_sharding_seq = weight.sharding_spec.sharding_sequence
    if bias is not None:
        bias_sharding_seq = bias.sharding_spec.sharding_sequence
    device_mesh = weight.sharding_spec.device_mesh
    pg_axis0 = weight.pg_axis0
    pg_axis1 = weight.pg_axis1

    # the last dim of input should have the same spec as the first dim of weight
    # the weight is transposed, so we look at the second dimension
    assert input_sharding_seq[-1] == weight_sharding_seq[1]

    if bias is not None:
        assert bias_sharding_seq[0] == weight_sharding_seq[0]

    # compute the output sharding sequence
    # as weight is transposed, so we look at the first dimension
    output_shard_seq = input_sharding_seq[:-1] + weight_sharding_seq[:1]
    output_shard_seq = deepcopy(output_shard_seq)

    # TODO: add reduce grad logic

    # handle column and row parallel linear
    # by reusing the implementation above
    out = F.linear(input_tensor, weight)

    # run all reduce if necessary
    last_dim_spec = input_sharding_seq[-1]
    if last_dim_spec.is_replica:
        pass
    elif last_dim_spec.shard_list is not None:
        for dim in last_dim_spec.shard_list:
            if dim == 0:
                reduce_input(out, pg_axis0)
            elif dim == 1:
                reduce_input(out, pg_axis1)
            else:
                raise RuntimeError("Found invalid sharding axis {dim}, only 0 or 1 is expected")
    # add bias
    if bias is not None:
        out += bias

    # convert shard seq to partition dict
    output_partition_dict = {}
    for index, dim_spec in enumerate(output_shard_seq):
        if not dim_spec.is_replica:
            if index not in output_partition_dict:
                output_partition_dict[index] = []
            output_partition_dict[index].extend(dim_spec.shard_list)

    entire_shape = out.shape
    output_sharding_spec = ShardingSpec(device_mesh, entire_shape, output_partition_dict)
    ret_tensor = ColoTensor.from_torch_tensor(out)
    setattr(ret_tensor, 'sharding_spec', output_sharding_spec)
    return ret_tensor


def _has_sharding_spec(tensor):
    """
    A tentative function to check whether the tensor is using the new sharding spec API. We assume that the sharding spec object is
    set as the attribute `sharding_spec` on a tensor.
    """
    return hasattr(tensor, 'sharding_spec')


@colo_op_impl(F.linear)
def colo_linear(input: GeneralTensor, weight: GeneralTensor, bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    if _has_sharding_spec(weight):
        return _new_colo_linear_imp(input, weight, bias)
    else:
        return colo_linear_imp(input, weight, bias)
