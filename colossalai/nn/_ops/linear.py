import torch.nn.functional as F
from typing import Optional
from ._utils import GeneralTensor, convert_to_colo_tensor
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import reduce_input, reduce_grad
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ComputeSpec, ColoTensor, distspec
from colossalai.context import ParallelMode
from colossalai.nn.graph import register_colo_graph, GraphOpNode, GraphGlobalEnv


def colo_linear_1Drow(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    input_tensor = input_tensor.convert_to_dist_spec(
        distspec.shard(weight.tensor_spec.get_process_group(), [-1], [weight.tensor_spec.get_process_group_size()]))

    # Output:P
    partial_output = F.linear(input_tensor, weight)
    # Reduce(Output)
    output = reduce_input(partial_output, ParallelMode.PARALLEL_1D)
    # Bias
    if bias is not None:
        assert not bias.has_compute_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias

    output = ColoTensor.from_torch_tensor(output,
                                          spec=TensorSpec(distspec.replicate(weight.tensor_spec.get_process_group())))
    return output


def colo_linear_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    compute_spec = weight.tensor_spec.compute_spec
    input_tensor = input_tensor.convert_to_dist_spec(distspec.replicate(weight.tensor_spec.get_process_group()))
    input_parallel = reduce_grad(input_tensor, ParallelMode.PARALLEL_1D)

    output_parallel = F.linear(input_parallel, weight, bias)
    output = ColoTensor.from_torch_tensor(output_parallel,
                                          spec=TensorSpec(
                                              distspec.shard(weight.tensor_spec.get_process_group(), [-1],
                                                             [weight.tensor_spec.get_process_group_size()]),
                                              ComputeSpec(ComputePattern.TP1D)))
    if compute_spec.output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_linear_1d(mode: str, input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> 'ColoTensor':
    assert mode in ('row', 'col')
    funcs = {'row': colo_linear_1Drow, 'col': colo_linear_1Dcol}
    return funcs[mode](input_tensor, weight, bias)


@register_colo_graph(input_pos=[1], param_pos=[2, 3])
def colo_linear_imp(input_tensor: GeneralTensor,
                    weight: GeneralTensor,
                    bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor, weight, bias = tuple(map(convert_to_colo_tensor, (input_tensor, weight, bias)))

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.tensor_spec.is_replicate(), 'Invalid weight spec for native Linear op'
        assert bias is None or bias.tensor_spec.is_replicate(), 'Invalid bias spec for native Linear op'
        ret_tensor = ColoTensor.from_torch_tensor(F.linear(input_tensor, weight, bias))
    elif weight.tensor_spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.tensor_spec.is_shard_1dcol() and (bias is None or bias.tensor_spec.is_replicate()):
            mode = 'row'
        elif weight.tensor_spec.is_shard_1drow() and (bias is None or bias.tensor_spec.is_shard_1drow()
                                                      or bias.tensor_spec.is_shard_1dcol()):
            mode = 'col'
        else:
            raise RuntimeError(f"the weight or bias tensor spec is not valid, weight {weight.tensor_spec}, bias {bias}")
        ret_tensor = colo_linear_1d(mode, input_tensor, weight, bias)
    else:
        raise NotImplementedError

    return ret_tensor


@colo_op_impl(F.linear)
def colo_linear(input_tensor: GeneralTensor,
                weight: GeneralTensor,
                bias: Optional[GeneralTensor] = None) -> 'ColoTensor':
    return colo_linear_imp(input_tensor, weight, bias)
