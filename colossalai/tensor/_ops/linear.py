import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, \
    gather_forward_split_backward, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ShardPattern


def colo_linear_1Drow(input_tensor: ColoTensor, weight: ColoTensor, bias:ColoTensor) -> ColoTensor:
    parallel_action = weight.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DRow)
    # Input:S[1] x Weight:S[0] = Output:P
    # All-Reduce(Output) + bias = res
    # Input:S[1]                
    if input_tensor.is_gathered():
        # Not splited yet.
        assert divide(input_tensor.shape[-1], gpc.tensor_parallel_size) == weight.size(-1), \
        'Invalid shapes in 1Drow forward: input={}, weight={}. Expected last dim of input {}.'.format(
        input_tensor.shape, weight.size, weight.size(-1) * gpc.tensor_parallel_size)
        input_per_partition = split_forward_gather_backward(input_tensor.torch_tensor(), parallel_action.parallel_mode, dim=-1)
    elif input_tensor.shard_pattern == ShardPattern.Col:
        # Splited by 1Dcol
        assert input_tensor.shape[-1] == weight.size(-1), \
        'Invalid shapes in 1Drow forward: input={}, weight={}. Expected last dim of input {}.'.format(
        input_tensor.shape, weight.size, weight.size(-1))
        input_per_partition = input_tensor.torch_tensor()
    else:
        raise NotImplementedError

    # Output:P
    partial_output = torch.nn.functional.linear(input_per_partition, weight.torch_tensor())
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # Bias
    if bias is not None:
        assert not bias.has_spec(), 'Invalid bias spec for 1Drow Linear op'
        output = output + bias.torch_tensor()
    output = ColoTensor.init_from_torch_tensor(output)
    return output

def colo_linear_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, bias:ColoTensor) -> ColoTensor:
    # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
    # All-Gather(Output)
    # Input:B
    parallel_action = weight.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DCol)
    if input_tensor.is_gathered():
        # Not splited yet.
        assert input_tensor.shape[-1] == weight.size(-1), \
            'Invalid shapes in 1Dcol forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_tensor.shape, weight.size, weight.size(-1))
        input_parallel = reduce_grad(input_tensor.torch_tensor(), parallel_action.parallel_mode)

    # Bias:S[1]
    if bias is not None:
        assert bias.has_spec() and bias.shard_spec.num_action == 1 and \
            bias.shard_pattern in [ShardPattern.Col, ShardPattern.Row], \
                'Invalid bias spec for 1Dcol Linear op'

    output_parallel = torch.nn.functional.linear(input_parallel, weight.torch_tensor(), bias.torch_tensor())
    
    output = ColoTensor.init_from_torch_tensor(output_parallel)
    out_parallel_action_list = [
        ParallelAction(
            priority=1, compute_pattern=ComputePattern.Activation, 
            parallel_mode=parallel_action.parallel_mode
        )
    ]
    output_spec = TensorSpec(out_parallel_action_list)
    output.set_spec(output_spec, shard=False)
    output.set_shard_pattern(ShardPattern.Col)
    if parallel_action.gather_out:
        # All-Gather(Output)
        output.gather()
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
                
    # Add communication logic before and after linear call.
    if not weight.has_spec(): # No Model Parallel Applied
        assert not bias.has_spec(), 'Invalid bias spec for native Linear op'
        input_tensor = input_tensor.torch_tensor()
        weight = weight.torch_tensor()
        bias = bias.torch_tensor()
        return ColoTensor.init_from_torch_tensor(torch.nn.functional.linear(input_tensor, weight, bias))
    elif weight.shard_spec.num_action == 1: # Single Model Parallel Applied
        compute_patterns = weight.shard_spec.compute_patterns
        if ComputePattern.TP1DRow in compute_patterns:
            return colo_linear_1Drow(input_tensor, weight, bias)
        elif ComputePattern.TP1DCol in compute_patterns:
            return colo_linear_1Dcol(input_tensor, weight, bias)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
