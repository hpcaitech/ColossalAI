import torch
from colossalai.gemini.tensor import stateful_op_impl
from ..stateful_tensor import StatefulTensorV2
from packaging import version


@stateful_op_impl(torch.nn.functional.linear)
def stateful_linear(types, args, kwargs, pg):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    print(f'inside sharded_linear {len(args)}')
    print(f'inside sharded_linear kwargs {kwargs}')
    # print(args)
    input = args[0]
    weight = args[1]

    if version.parse(torch.__version__) > version.parse("1.11.0"):
        if len(args) == 3:
            bias = args[2]
        else:
            bias = None
    else:
        bias = kwargs.get('bias', None)
        if isinstance(bias, StatefulTensorV2):
            bias = bias.torch_tensor()

    print(bias)

    # Add communication logic before and after linear call.
    if isinstance(weight, StatefulTensorV2):
        return torch.nn.functional.linear(input, weight.torch_tensor(), bias)
    else:
        return torch.nn.functional.linear(input, weight, bias)
