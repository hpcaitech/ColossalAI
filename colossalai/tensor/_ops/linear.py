import torch
import torch.nn.functional as F
from typing import Optional, Union
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, dist_spec
from colossalai.tensor.graph import GraphOpNode, GraphGlobalEnv


def colo_linear_1Drow(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> ColoTensor:
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    input_tensor = input_tensor.to_dist_spec(
        dist_spec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group().size()]))

    # Output:P
    partial_output = F.linear(input_tensor, weight)
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # Bias
    if bias is not None:
        assert not bias.has_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias

    output = ColoTensor.from_torch_tensor(output, spec=TensorSpec(dist_spec.replicate(weight.spec.get_process_group())))
    return output


def colo_linear_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: Optional[ColoTensor]) -> ColoTensor:
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    input_tensor = input_tensor.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))
    input_parallel = reduce_grad(input_tensor, parallel_action.parallel_mode)

    output_parallel = F.linear(input_parallel, weight, bias)

    output = ColoTensor.from_torch_tensor(
        output_parallel,
        spec=TensorSpec(
            dist_spec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group().size()]),
            [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)]))
    if parallel_action.gather_out:
        # All-Gather(Output)
        output = output.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))
    return output


@colo_op_impl(F.linear)
def colo_linear(input_tensor: Union[ColoTensor, torch.Tensor], weight: ColoTensor, bias: Optional[ColoTensor] = None):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    if not isinstance(input_tensor, ColoTensor):
        input_tensor = ColoTensor.from_torch_tensor(input_tensor)

    # building the computing graph, inputs -> op
    if GraphGlobalEnv().graph_building:
        cur_op_node = GraphOpNode('linear', [weight, bias])
        cur_op_node.add_prev_tensor(input_tensor)
    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_spec():    # No Model Parallel Applied
        assert bias.spec.is_gathered(), 'Invalid bias spec for native Linear op'
        assert bias.spec.is_gathered(), 'Invalid bias spec for native Linear op'
        return F.linear(input_tensor, weight, bias)
    elif weight.spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.spec.is_1D_col() and (bias is None or bias.spec.is_gathered()):
            ret_tensor = colo_linear_1Drow(input_tensor, weight, bias)
        elif weight.spec.is_1D_row() and (bias is None or bias.spec.is_1D_row() or bias.spec.is_1D_col()):
            ret_tensor = colo_linear_1Dcol(input_tensor, weight, bias)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # building the computing graph, op -> output
    if GraphGlobalEnv().graph_building:
        cur_op_node.add_post_tensor(ret_tensor)

    return ret_tensor
