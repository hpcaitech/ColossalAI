from .base_loss import BaseLoss
from .cross_entropy_1d import CrossEntropyLoss1D
from .cross_entropy_2d import CrossEntropyLoss2D
from .cross_entropy_2p5d import CrossEntropyLoss2p5D
from .cross_entropy_3d import CrossEntropyLoss3D

__all__ = ['CrossEntropyLoss1D', 'CrossEntropyLoss2D', 'CrossEntropyLoss2p5D', 'CrossEntropyLoss3D']
