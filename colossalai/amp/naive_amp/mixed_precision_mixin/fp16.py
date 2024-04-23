from abc import abstractmethod
from enum import Enum

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler

from .base import MixedPrecisionMixin


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class FP16MixedPrecisionMixin(MixedPrecisionMixin):
    dtype = torch.float16

    def __init__(
        self,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__()
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )
        self.optim_state = OptimState.UNSCALED
        self.found_overflow = torch.zeros(1, dtype=torch.float, device=get_accelerator().get_current_device())

    @property
    def loss_scale(self) -> float:
        return self.grad_scaler.scale.item()

    @abstractmethod
    def check_local_overflow(self) -> bool:
        """Check whether there is overflow in the local process. This method should be implemented by subclasses.

        Returns:
            bool: Whether there is overflow in the local process.
        """

    def check_overflow(self) -> bool:
        # clear previous overflow record
        self.found_overflow.fill_(0.0)
        if self.check_local_overflow():
            self.found_overflow.fill_(1.0)
        dist.all_reduce(self.found_overflow, op=dist.ReduceOp.MAX)
        return self.found_overflow.item() > 0

    def pre_backward(self, loss: Tensor) -> Tensor:
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        return loss

    def pre_backward_by_grad(self, tensor: Tensor, grad: Tensor) -> Tensor:
        self.optim_state = OptimState.SCALED
        return grad

    def should_skip_step(self) -> bool:
        found_inf = self.check_overflow()
        self.grad_scaler.update(found_inf)
        if found_inf:
            self.optim_state = OptimState.UNSCALED
        return found_inf

    def pre_zero_grad(self) -> None:
        pass

    def get_grad_div_scale(self) -> float:
        assert self.optim_state == OptimState.SCALED, "grads should be scaled before clipping"
        self.optim_state = OptimState.UNSCALED
        return self.loss_scale
