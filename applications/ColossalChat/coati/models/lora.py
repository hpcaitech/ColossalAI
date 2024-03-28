"""
LORA utils
"""

import dataclasses
import math
import warnings
from typing import Optional

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


@dataclasses.dataclass
class LoRAManager:
    merge_weights: bool = False


LORA_MANAGER = LoRAManager()


class LoraLinear(lora.LoRALayer, nn.Module):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear."""

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        self.weight = weight
        self.bias = bias

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # Initialize A with the default values for nn.Linear and set B to zero.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """
        This function runs when model.train() is invoked. It is used to prepare the linear layer for training
        """

        def T(w):
            return w.T if self.fan_in_fan_out else w

        self.training = mode
        if LORA_MANAGER.merge_weights:
            if mode and self.merged:
                warnings.warn("Invoke module.train() would unmerge LoRA weights.")
                raise NotImplementedError("LoRA unmerge is not tested.")
                # Make sure that the weights are not merged
                if self.r > 0:
                    if not hasattr(self, "lora_A") or not hasattr(self, "lora_B"):
                        # FIXME(csric): temporary fix
                        self.lora_A = nn.Parameter(self.weight.new_empty((self.r, self.in_features)))
                        self.lora_B = nn.Parameter(self.weight.new_empty((self.out_features, self.r)))
                        self.reset_parameters()
                    else:
                        self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
            elif not mode and not self.merged:
                warnings.warn("Invoke module.eval() would merge LoRA weights.")
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                    delattr(self, "lora_A")
                    delattr(self, "lora_B")
                self.merged = True

        return self

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def _lora_linear_wrapper(linear: nn.Linear, lora_rank: int) -> LoraLinear:
    """
    Wraps a linear layer with LoRA functionality.

    Args:
        linear (nn.Linear): The linear layer to be wrapped.
        lora_rank (int): The rank of the LoRA decomposition.

    Returns:
        LoraLinear: The wrapped linear layer with LoRA functionality.
    """
    assert (
        lora_rank <= linear.in_features
    ), f"LoRA rank ({lora_rank}) must be less than or equal to in features ({linear.in_features})"
    lora_linear = LoraLinear(linear.weight, linear.bias, r=lora_rank)
    return lora_linear


def _convert_to_lora_recursively(module: nn.Module, lora_rank: int) -> None:
    """
    Recursively converts the given module and its children to LoRA (Low-Rank Approximation) form.

    Args:
        module (nn.Module): The module to convert to LoRA form.
        lora_rank (int): The rank of the LoRA approximation.

    Returns:
        None
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _lora_linear_wrapper(child, lora_rank))
        else:
            _convert_to_lora_recursively(child, lora_rank)


def convert_to_lora_module(module: nn.Module, lora_rank: int, lora_train_bias: str = "none") -> nn.Module:
    """Convert a torch.nn.Module to a LoRA module.

    Args:
        module (nn.Module): The module to convert.
        lora_rank (int): LoRA rank.

    Returns:
        nn.Module: The converted module.
    """
    if lora_rank <= 0:
        return module
    _convert_to_lora_recursively(module, lora_rank)
    lora.mark_only_lora_as_trainable(module, lora_train_bias)
    return module
