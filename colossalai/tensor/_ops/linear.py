import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor.colo_tensor import ColoTensor
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.utils.cuda import get_current_device


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

    if isinstance(bias, ColoTensor):
        bias = bias.torch_tensor()

    # Add communication logic before and after linear call.
    if isinstance(weight, ColoTensor):
        if weight.shard_spec == None:
            if isinstance(input_tensor, ColoTensor):
                input_tensor = input_tensor.torch_tensor()
            if isinstance(weight, ColoTensor):
                weight = weight.torch_tensor()
            return torch.nn.functional.linear(input_tensor, weight, bias)
        elif weight.shard_spec == '1Drow':
            # Input:S[1] x Weight:S[0] = Output:P
            # All-Reduce(Output) + bias = res
            assert divide(input_tensor.shape[-1], gpc.tensor_parallel_size) == weight.size(-1), \
            'Invalid shapes in 1Drow forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_tensor.shape, weight.size, weight.size[-1] * gpc.tensor_parallel_size)
            # Input:S[1]
            input_per_partition = split_forward_gather_backward(input_tensor, ParallelMode.PARALLEL_1D, dim=-1)
            # Output:P
            device = get_current_device()    # TODO where to put to(deivce)?
            weight_ = weight.torch_tensor().to(device)
            partial_output = torch.nn.functional.linear(input_per_partition, weight_)
            # Reduce(Output)
            output = reduce_input(partial_output, ParallelMode.PARALLEL_1D)
            # Bias
            if bias is not None:
                bias_ = bias.to(device)
                output = output + bias_
            return output

        else:
            raise NotImplementedError
    else:
        return torch.nn.functional.linear(input_tensor, weight, bias)
