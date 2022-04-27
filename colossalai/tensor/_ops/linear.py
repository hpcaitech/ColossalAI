import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, \
    gather_forward_split_backward, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor


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

    bias_spec = None
    if isinstance(bias, ColoTensor):
        bias_spec = bias.shard_spec
        bias = bias.torch_tensor()
                
    # Add communication logic before and after linear call.
    if isinstance(weight, ColoTensor):
        if weight.shard_spec == None or weight.shard_spec.num_action == 0:
            assert bias_spec == None or bias_spec.num_action == 0, 'Invalid bias spec for native Linear op'
            if isinstance(input_tensor, ColoTensor):
                input_tensor = input_tensor.torch_tensor()
            if isinstance(weight, ColoTensor):
                weight = weight.torch_tensor()
            return ColoTensor.init_from_torch_tensor(torch.nn.functional.linear(input_tensor, weight, bias))
        elif weight.shard_spec.num_action == 1:
            parallel_action = weight.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DRow)
            compute_patterns = weight.shard_spec.compute_patterns
            if ComputePattern.TP1DRow in compute_patterns:
                # Input:S[1] x Weight:S[0] = Output:P
                # All-Reduce(Output) + bias = res
                # Input:S[1]
                input_spec = None
                if isinstance(input_tensor, ColoTensor):
                    input_spec = input_tensor.shard_spec
                    input_tensor = input_tensor.torch_tensor()

                if input_spec == None or input_spec.num_action == 0:
                    # Not splited yet.
                    assert divide(input_tensor.shape[-1], gpc.tensor_parallel_size) == weight.size(-1), \
                    'Invalid shapes in 1Drow forward: input={}, weight={}. Expected last dim of input {}.'.format(
                    input_tensor.shape, weight.size, weight.size(-1) * gpc.tensor_parallel_size)
                    input_per_partition = split_forward_gather_backward(input_tensor, parallel_action.parallel_mode, dim=-1)
                elif input_tensor.shard_spec.num_action == 1:
                    if ComputePattern.TP1DCol in input_spec.compute_patterns:
                        # Splited by 1Dcol
                        assert input_tensor.shape[-1] == weight.size(-1), \
                        'Invalid shapes in 1Drow forward: input={}, weight={}. Expected last dim of input {}.'.format(
                        input_tensor.shape, weight.size, weight.size(-1))
                        input_per_partition = input_tensor
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                # Output:P
                weight_ = weight.torch_tensor()
                partial_output = torch.nn.functional.linear(input_per_partition, weight_)
                # Reduce(Output)
                output = reduce_input(partial_output, parallel_action.parallel_mode)
                # Bias
                if bias is not None:
                    assert bias_spec == None or bias_spec.num_action == 0, 'Invalid bias spec for 1Drow Linear op'
                    output = output + bias
                output = ColoTensor.init_from_torch_tensor(output)
                return output
            elif ComputePattern.TP1DCol in compute_patterns:
                # Input:B x Weight:S[1] + Bias:S[1] = Output:S[1]
                # All-Gather(Output)
                # Input:B
                input_spec = None
                output_spec = None
                parallel_action = weight.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DCol)
                if isinstance(input_tensor, ColoTensor):
                    input_spec = input_tensor.shard_spec
                    input_tensor = input_tensor.torch_tensor()
                
                if input_spec == None or input_spec.num_action == 0:
                    # Not splited yet.
                    assert input_tensor.shape[-1] == weight.size(-1), \
                        'Invalid shapes in 1Dcol forward: input={}, weight={}. Expected last dim of input {}.'.format(
                            input_tensor.shape, weight.size, weight.size(-1))
                    input_parallel = reduce_grad(input_tensor, parallel_action.parallel_mode)
                else:
                    raise NotImplementedError
                # Bias:S[1]
                if bias is not None:
                    assert bias_spec is not None and bias_spec.num_action == 1 and \
                        ComputePattern.TP1DCol in bias_spec.compute_patterns, \
                            'Invalid bias spec for 1Dcol Linear op'
                
                weight_ = weight.torch_tensor()
                output_parallel = torch.nn.functional.linear(input_parallel, weight_, bias)
               
                if parallel_action.gather_out:
                    # All-Gather(Output)
                    output = gather_forward_split_backward(output_parallel, parallel_action.parallel_mode, dim=-1)
                    output = ColoTensor.init_from_torch_tensor(output)
                else:
                    output = ColoTensor.init_from_torch_tensor(output_parallel)
                    out_parallel_action_list = [
                        ParallelAction(
                            priority=1, compute_pattern=ComputePattern.TP1DCol, 
                            parallel_mode=parallel_action.parallel_mode
                        )
                    ]
                    output_spec = TensorSpec(out_parallel_action_list)
                # set ColoTensor spec
                if output_spec is not None:
                    output.set_spec(output_spec)
                return output

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        return torch.nn.functional.linear(input_tensor, weight, bias)
