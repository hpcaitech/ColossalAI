import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import reduce_input, reduce_grad
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, distspec
from colossalai.tensor.graph import GraphOpNode, GraphGlobalEnv
from ._utils import GeneralTensor, convert_to_colo_tensor


def colo_linear_1Drow(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> ColoTensor:
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    input_tensor = input_tensor.convert_to_dist_spec(
        distspec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group_size()]))

    # Output:P
    partial_output = F.linear(input_tensor, weight)
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # Bias
    if bias is not None:
        assert not bias.has_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias

    output = ColoTensor.from_torch_tensor(output, spec=TensorSpec(distspec.replicate(weight.spec.get_process_group())))
    return output


def colo_linear_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> ColoTensor:
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    input_tensor = input_tensor.convert_to_dist_spec(distspec.replicate(weight.spec.get_process_group()))
    input_parallel = reduce_grad(input_tensor, parallel_action.parallel_mode)

    output_parallel = F.linear(input_parallel, weight, bias)
    output = ColoTensor.from_torch_tensor(
        output_parallel,
        spec=TensorSpec(distspec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group_size()]),
                        [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)]))
    if parallel_action.gather_out:
        # All-Gather(Output)
        output = output.convert_to_dist_spec(distspec.replicate(weight.spec.get_process_group()))
    return output


def colo_linear_1d(mode: str, input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> ColoTensor:
    assert mode in ('row', 'col')
    funcs = {'row': colo_linear_1Drow, 'col': colo_linear_1Dcol}
    return funcs[mode](input_tensor, weight, bias)


@colo_op_impl(F.linear)
def colo_linear(input_tensor: GeneralTensor, weight: GeneralTensor, bias: Optional[GeneralTensor] = None):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor, weight, bias = tuple(map(convert_to_colo_tensor, (input_tensor, weight, bias)))

    # building the computing graph, inputs -> op
    if GraphGlobalEnv().graph_building:
        cur_op_node = GraphOpNode('linear', [weight, bias])
        cur_op_node.add_prev_tensor(input_tensor)
    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_spec():    # No Model Parallel Applied
        assert weight.spec.is_gathered(), 'Invalid weight spec for native Linear op'
        assert bias is None or bias.spec.is_gathered(), 'Invalid bias spec for native Linear op'
        ret_tensor = ColoTensor.from_torch_tensor(F.linear(input_tensor, weight, bias))
    elif weight.spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.spec.is_1D_col() and (bias is None or bias.spec.is_gathered()):
            mode = 'row'
        elif weight.spec.is_1D_row() and (bias is None or bias.spec.is_1D_row() or bias.spec.is_1D_col()):
            mode = 'col'
        else:
            raise NotImplementedError
        ret_tensor = colo_linear_1d(mode, input_tensor, weight, bias)
    else:
        raise NotImplementedError

    # building the computing graph, op -> output
    if GraphGlobalEnv().graph_building:
        cur_op_node.add_post_tensor(ret_tensor)
    return ret_tensor
