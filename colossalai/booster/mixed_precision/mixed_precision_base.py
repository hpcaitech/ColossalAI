from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from ..interface import OptimizerWrapper


class MixedPrecision(ABC):
    """
    An abstract class for mixed precision training.
    """

    @abstractmethod
    def configure(self,
                  model: nn.Module,
                  optimizer: Optimizer,
                  criterion: Callable = None) -> Tuple[nn.Module, OptimizerWrapper, Callable]:
        # TODO: implement this method
        pass
