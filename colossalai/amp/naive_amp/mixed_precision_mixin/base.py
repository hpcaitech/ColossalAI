from abc import ABC, abstractmethod

import torch
from torch import Tensor


class MixedPrecisionMixin(ABC):
    """A helper class for mixed precision training. This mixin is used in mixed precision optimizers.

    Attributes:
        dtype (torc.dtype): The expected dtype of the gradients.

    Examples:
        ```python
        class MyMixedPrecisionOptimizer(OptimizerWrapper):
            def __init__(self, optim: Optimizer):
                super().__init__(optim)
                self.mixed_precision = MixedPrecisionMixin()

            def backward(self, loss):
                loss = self.mixed_precision.pre_backward(loss)
                loss.backward()

            def backward_by_grad(self, tensor, grad):
                grad = self.mixed_precision.pre_backward_by_grad(tensor, grad)
                tensor.backward(grad)

            def step(self):
                if self.mixed_precision.should_skip_step():
                    self.zero_grad()
                    return
                div_scale = self.mixed_precision.get_grad_div_scale()
                # maybe clip grad here
                # maybe scale grad here
                self.optim.step()

            def zero_grad(self):
                self.mixed_precision.pre_zero_grad()
                return self.optim.zero_grad()
        ```
    """

    dtype: torch.dtype

    @abstractmethod
    def pre_backward(self, loss: Tensor) -> Tensor:
        """Called before backward.

        Args:
            loss (Tensor): Loss value.

        Returns:
            Tensor: Loss value (possibly scaled).
        """

    @abstractmethod
    def pre_backward_by_grad(self, tensor: Tensor, grad: Tensor) -> Tensor:
        """Called before backward by grad. This is helpful for pipeline parallelism.

        Args:
            tensor (Tensor): Tensor to backward.
            grad (Tensor): Gradient of the tensor.

        Returns:
            Tensor: Gradient of the tensor (possibly scaled).
        """

    @abstractmethod
    def should_skip_step(self) -> bool:
        """Called before step.

        Returns:
            bool: Whether to skip the step.
        """

    @abstractmethod
    def pre_zero_grad(self) -> None:
        """Called before zero_grad."""

    @abstractmethod
    def get_grad_div_scale(self) -> float:
        """Called before step or clip_grad. To keep computation efficiency, this method does not (maybe) unscale grads.

        Returns:
            float: A divisor for gradient clipping or step.
        """
