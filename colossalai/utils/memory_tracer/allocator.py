import torch
from colossalai.zero.sharded_param import ShardedTensor
from typing import Union


def col_tensor_mem_usage(t: Union[torch.Tensor, ShardedTensor]) -> int:
    if isinstance(t, ShardedTensor):
        target = t.payload
    else:
        target = t
    return target.numel() * target.element_size()


def col_allocate_payload(device: torch.device) -> torch.Tensor:
    pass


def col_release_payload(t: torch.Tensor):
    pass
