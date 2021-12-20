from torch import nn

from ._utils import calc_acc
from .accuracy_2p5d import Accuracy2p5D
from .accuracy_3d import Accuracy3D

_parallel_accuracy = {
    '3d': Accuracy3D,
}


class Accuracy(nn.Module):
    def __init__(self, tensor_parallel: str = None):
        super().__init__()
        if tensor_parallel is not None:
            self.acc = _parallel_accuracy[tensor_parallel]()
        else:
            self.acc = calc_acc

    def forward(self, *args):
        return self.acc(*args)
