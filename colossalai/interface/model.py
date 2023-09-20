import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    A wrapper class to define the common interface used by booster.

    Args:
        module (nn.Module): The model to be wrapped.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def unwrap(self):
        """
        Unwrap the model to return the original model for checkpoint saving/loading.
        """
        if isinstance(self.module, ModelWrapper):
            return self.module.unwrap()
        return self.module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class AMPModelMixin:
    """This mixin class defines the interface for AMP training."""

    def update_master_params(self):
        """
        Update the master parameters for AMP training.
        """
