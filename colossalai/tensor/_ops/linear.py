import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor.colo_tensor import ColoTensor
from packaging import version


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
        return torch.nn.functional.linear(input_tensor, weight.torch_tensor(), bias)
    else:
        return torch.nn.functional.linear(input_tensor, weight, bias)
