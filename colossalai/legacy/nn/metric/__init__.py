from torch import nn

from colossalai.legacy.nn.layer.utils import get_tensor_parallel_mode

from ._utils import calc_acc
from .accuracy_2d import Accuracy2D
from .accuracy_2p5d import Accuracy2p5D
from .accuracy_3d import Accuracy3D

_parallel_accuracy = {
    "2d": Accuracy2D,
    "2.5d": Accuracy2p5D,
    "3d": Accuracy3D,
}


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel not in _parallel_accuracy:
            self.acc = calc_acc
        else:
            self.acc = _parallel_accuracy[tensor_parallel]()

    def forward(self, *args):
        return self.acc(*args)
