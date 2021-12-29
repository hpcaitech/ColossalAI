from torch import Tensor

from ..parallel_2d._operation import split_tensor_2d
from ..parallel_2p5d._operation import split_tensor_2p5d
from ..parallel_3d._operation import split_tensor_3d
from ..utils import get_tensor_parallel_mode

_parallel_split_batch = {'2d': split_tensor_2d, '2.5d': split_tensor_2p5d, '3d': split_tensor_3d}


def split_batch(input_) -> Tensor:
    tensor_parallel_mode = get_tensor_parallel_mode()
    if tensor_parallel_mode in _parallel_split_batch:
        if isinstance(input_, (tuple, list)):
            return tuple(map(_parallel_split_batch[tensor_parallel_mode], input_))
        else:
            return _parallel_split_batch[tensor_parallel_mode](input_)
    else:
        return input_
