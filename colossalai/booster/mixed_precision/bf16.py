from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper

from .mixed_precision_base import MixedPrecision


class BF16Module(ModelWrapper):
    def forward(self, *args, **kwargs):
        with get_accelerator().autocast(dtype=torch.bfloat16):
            return self.module(*args, **kwargs)


class BF16MixedPrecision(MixedPrecision):
    """BF16 autocast with FP32 parameters/optimizer states and no loss scaling."""

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable]:
        model = BF16Module(model)
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer = OptimizerWrapper(optimizer)
        if criterion is not None:
            criterion = BF16Module(criterion)
        return model, optimizer, criterion
