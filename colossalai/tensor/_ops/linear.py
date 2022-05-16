import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, dist_spec
from colossalai.tensor.graph import GraphOpNode, GraphGlobalEnv


def colo_linear_1Drow(input_tensor: ColoTensor, weight: ColoTensor, bias: ColoTensor) -> ColoTensor:
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]
    input_tensor.to_dist_spec(
        dist_spec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group().size()]))

    # Output:P
    partial_output = torch.nn.functional.linear(input_tensor.torch_tensor(), weight.torch_tensor())
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # Bias
    if bias is not None:
        assert not bias.has_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias.torch_tensor()
    output = ColoTensor.init_from_torch_tensor(output,
                                               spec=TensorSpec(dist_spec.replicate(weight.spec.get_process_group())))
    return output


def colo_linear_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, bias: ColoTensor) -> ColoTensor:
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    input_tensor.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))
    input_parallel = reduce_grad(input_tensor.torch_tensor(), parallel_action.parallel_mode)
    if bias is not None:
        bias = bias.torch_tensor()
    output_parallel = torch.nn.functional.linear(input_parallel, weight.torch_tensor(), bias)

    output = ColoTensor.init_from_torch_tensor(
        output_parallel,
        spec=TensorSpec(
            dist_spec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group().size()]),
            [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)]))
    if parallel_action.gather_out:
        # All-Gather(Output)
        output.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))
    return output


@colo_op_impl(torch.nn.functional.linear)
def colo_linear(types, args, kwargs, pg):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor = args[0]
    weight = args[1]

    if version.parse(torch.__version__) > version.parse("1.11.0"):
        if len(args) == 3:
            bias = args[2]
        else:
            bias = None
    else:
        bias = kwargs.get('bias', None)

    if not isinstance(input_tensor, ColoTensor):
        input_tensor = ColoTensor.init_from_torch_tensor(input_tensor)

    if not isinstance(weight, ColoTensor):
        weight = ColoTensor.init_from_torch_tensor(weight)

    if bias is not None and not isinstance(bias, ColoTensor):
        bias = ColoTensor.init_from_torch_tensor(bias)

    # building the computing graph, inputs -> op
    if GraphGlobalEnv().graph_building:
        cur_op_node = GraphOpNode('linear', [weight, bias])
        cur_op_node.add_prev_tensor(input_tensor)

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not weight.has_spec():    # No Model Parallel Applied
        assert bias.spec.is_gathered(), 'Invalid bias spec for native Linear op'
        assert bias.spec.is_gathered(), 'Invalid bias spec for native Linear op'
        input_tensor = input_tensor.torch_tensor()
        weight = weight.torch_tensor()
        if bias is not None:
            bias = bias.torch_tensor()
        ret_tensor = ColoTensor.init_from_torch_tensor(torch.nn.functional.linear(input_tensor, weight, bias))
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
