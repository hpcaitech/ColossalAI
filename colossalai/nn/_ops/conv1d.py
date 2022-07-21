import torch.nn.functional as F
from typing import Optional
from ._utils import GeneralTensor, convert_to_colo_tensor
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import reduce_input, reduce_grad
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ComputeSpec, ColoTensor, distspec
from colossalai.nn.graph import register_colo_graph, GraphOpNode, GraphGlobalEnv
from typing import Union, List, Tuple

import torch
import builtins

_int = builtins.int
_size = Union[torch.Size, List[_int], Tuple[_int, ...]]


@colo_op_impl(F.conv2d)
def colo_conv_2d(input: GeneralTensor,
                 weight: GeneralTensor,
                 bias: Optional[GeneralTensor] = None,
                 stride: Union[_int, _size] = 1,
                 padding: Union[_int, _size] = 0,
                 dilation: Union[_int, _size] = 1,
                 groups: _int = 1) -> 'ColoTensor':
    # implement a parallel 2d conv here
    # weight in shape (N, C, H, H)
    # supports Tensor Parallelism on dim H
    pass
