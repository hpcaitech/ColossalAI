import torch.nn as nn
from peft import PeftModel


class ModelWrapper(nn.Module):
    """
    A wrapper class to define the common interface used by booster.

    Args:
        module (nn.Module): The model to be wrapped.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def unwrap(self, unwrap_peft: bool = True):
        """
        Unwrap the model to return the original model for checkpoint saving/loading.
        """
        if isinstance(self.module, ModelWrapper):
            model = self.module.unwrap()
        else:
            model = self.module
        if unwrap_peft and isinstance(model, PeftModel):
            model = model.get_base_model()
        return model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class AMPModelMixin:
    """This mixin class defines the interface for AMP training."""

    def update_master_params(self):
        """
        Update the master parameters for AMP training.
        """
