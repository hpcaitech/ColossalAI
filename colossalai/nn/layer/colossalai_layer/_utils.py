import torch.nn as nn
from torch import Tensor

from ..parallel_2d._operation import split_batch_2d
from ..parallel_2p5d._operation import split_batch_2p5d
from ..parallel_3d._operation import split_batch_3d
from ..utils import get_tensor_parallel_mode

_parallel_split_batch = {'2d': split_batch_2d, '2.5d': split_batch_2p5d, '3d': split_batch_3d}


def partition_batch(input_) -> Tensor:
    tensor_parallel_mode = get_tensor_parallel_mode()
    if tensor_parallel_mode in _parallel_split_batch:
        if isinstance(input_, dict):
            return {k: _parallel_split_batch[tensor_parallel_mode](v) for k, v in input_.items()}
        else:
            return _parallel_split_batch[tensor_parallel_mode](input_)
    else:
        return input_


class ColossalaiModule(nn.Module):

    def __init__(self, module: nn.Module, **kwargs):
        super().__init__()
        # copy values
        self.__dict__ = module.__dict__.copy()
        # copy methods
        for name, attr in module.__class__.__dict__.items():
            if name not in ['__init__', 'forward'] and callable(attr):
                setattr(self, name, getattr(module, name))
        self._forward_func = module.forward
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, *args):
        return self._forward_func(*args)
