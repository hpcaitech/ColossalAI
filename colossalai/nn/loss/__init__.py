from torch import nn
from torch.nn.modules.loss import *
from torch.nn.modules.loss import _Loss

from colossalai.nn.layer.utils import get_tensor_parallel_mode 
from .loss_2d import CrossEntropyLoss2D
from .loss_2p5d import CrossEntropyLoss2p5D
from .loss_3d import CrossEntropyLoss3D
from .loss_moe import MoeCrossEntropyLoss, MoeLoss

_parallel_cross_entropy = {
    '2d': CrossEntropyLoss2D,
    '2.5d': CrossEntropyLoss2p5D,
    '3d': CrossEntropyLoss3D
}


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction: bool = True, *args, **kwargs):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel in ['None', '1d']:
            reduction = 'mean' if reduction else 'none'
            self.loss = nn.CrossEntropyLoss(reduction=reduction, *args, **kwargs)
        else:
            self.loss = _parallel_cross_entropy[tensor_parallel](reduction=reduction, *args, **kwargs)

    def forward(self, *args):
        return self.loss(*args)
