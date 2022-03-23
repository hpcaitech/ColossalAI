import torch
import torch.distributed as dist

from colossalai.nn.layer.parallel_ops.wrapper import sharded_op_impl
from colossalai.zero.sharded_param.sharded_param import ShardedTensor

from packaging import version


@sharded_op_impl(torch.nn.functional.linear)
def sharded_linear(types, args, kwargs, pg):
    rank = dist.get_rank(pg)
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
        if isinstance(bias, ShardedTensor):
            bias = bias.payload

    print(bias)
    if isinstance(weight, ShardedTensor):
        return torch.nn.functional.linear(input, weight.payload.t(), bias)
    else:
        return torch.nn.functional.linear(input, weight, bias)
