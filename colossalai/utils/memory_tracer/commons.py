from colossalai.zero.sharded_param import ShardedTensor
from typing import Union
import torch


def col_tensor_mem_usage(t: Union[torch.Tensor, ShardedTensor]) -> int:
    if isinstance(t, ShardedTensor):
        target = t.payload
    else:
        target = t
    return target.numel() * target.element_size()
