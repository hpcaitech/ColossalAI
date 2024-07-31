"""
LORA utils
"""

import dataclasses
import math
import warnings
from typing import List, Optional, Union

import loralib as lora
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


@dataclasses.dataclass
class LoraManager:
    able_to_merge: bool = True


lora_manager = LoraManager()


@dataclasses.dataclass
class LoraConfig:
    r: int = 0
    lora_alpha: int = 32
    linear_lora_dropout: float = 0.1
    embedding_lora_dropout: float = 0.0
    lora_train_bias: str = "none"
    lora_initialization_method: str = "kaiming_uniform"
    target_modules: List = None

    @classmethod
    def from_file(cls, config_file: str):
        import json

        with open(config_file, "r") as f:
            config = json.load(f)
        return cls(**config)


class LoraBase(lora.LoRALayer, nn.Module):
    def __init__(
        self,
        r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_initialization_method: str = "kaiming_uniform",
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.merged = False
        self.lora_initialization_method = lora_initialization_method
        self.weight = None
        self.bias = None
        self.lora_A = None
        self.lora_B = None

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            if self.lora_initialization_method == "kaiming_uniform" or self.weight.size() != (
                self.out_features,
                self.in_features,
            ):
                # Initialize A with the default values for nn.Linear and set B to zero.
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif self.lora_initialization_method == "PiSSA":
                # PiSSA method in this paper: https://arxiv.org/abs/2404.02948
                # Assume the SVD of the original weights is W = USV^T
                # Initialize a frozen weight to U[:,r:]S[r:,r:]V^T[:,r:] to store less significent part of W
                # Only A, B are trainable, which are initialized to S[r:,:r]^0.5V^T[:,:r] and U[:,:r]S[r:,:r] respectively
                # self.scaling = 1.
                # SVD
                U, S, Vh = torch.svd_lowrank(
                    self.weight.to(torch.float32).data, self.r, niter=4
                )  # U: [out_features, in_features], S: [in_features], V: [in_features, in_features]
                # weight_backup = self.weight.clone()

                # Initialize A, B
                S = S / self.scaling
                self.lora_B.data = (U @ torch.diag(torch.sqrt(S))).to(torch.float32).contiguous()
                self.lora_A.data = (torch.diag(torch.sqrt(S)) @ Vh.T).to(torch.float32).contiguous()
                # Initialize weight
                # To reduce floating point error, we use residual instead of directly using U[:, :self.r] @ S[:self.r] @ Vh[:self.r, :]
                self.weight.data = (
                    ((self.weight - self.scaling * self.lora_B @ self.lora_A)).contiguous().to(self.weight.dtype)
                )
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
            else:
                raise ValueError(f"Unknown LoRA initialization method {self.lora_initialization_method}")

    def train(self, mode: bool = True):
        """
        This function runs when model.train() is invoked. It is used to prepare the linear layer for training
        """

        self.training = mode
        if mode and self.merged:
            warnings.warn("Invoke module.train() would unmerge LoRA weights.")
            raise NotImplementedError("LoRA unmerge is not tested.")
        elif not mode and not self.merged and lora_manager.able_to_merge:
            warnings.warn("Invoke module.eval() would merge LoRA weights.")
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
                delattr(self, "lora_A")
                delattr(self, "lora_B")
            self.merged = True

        return self


class LoraLinear(LoraBase):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear."""

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Union[nn.Parameter, bool],
        r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_initialization_method: str = "kaiming_uniform",
    ):
        super().__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_initialization_method=lora_initialization_method
        )
        self.weight = weight
        self.bias = bias
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(weight.shape[0]))
        if bias is not None:
            self.bias.requires_grad = True

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        assert lora_initialization_method in ["kaiming_uniform", "PiSSA"]
        self.lora_initialization_method = lora_initialization_method
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn((r, in_features)))
            self.lora_B = nn.Parameter(torch.randn((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)


class LoraEmbedding(LoraBase):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear."""

    def __init__(
        self,
        weight: nn.Parameter,
        r: int = 0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        num_embeddings: int = None,
        embedding_dim: int = None,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        lora_initialization_method: str = "kaiming_uniform",
    ):
        super().__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_initialization_method=lora_initialization_method
        )
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = weight

        in_features, out_features = num_embeddings, embedding_dim
        self.in_features = in_features
        self.out_features = out_features
        assert lora_initialization_method in ["kaiming_uniform", "PiSSA"]
        self.lora_initialization_method = lora_initialization_method

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn((r, in_features)))
            self.lora_B = nn.Parameter(torch.randn((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        # reset parameters
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def _embed(self, x: torch.Tensor, weight) -> torch.Tensor:
        return F.embedding(
            x,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor):
        base_embedding = self._embed(x, self.weight)
        # base_embedding.requires_grad = True   # force the embedding layer to be trainable for gradient checkpointing
        if self.r > 0 and not self.merged:
            lora_A_embedding = self._embed(x, self.lora_A.t())
            embedding = base_embedding + (lora_A_embedding @ self.lora_B.t()) * self.scaling
            return embedding
        else:
            return base_embedding

    def train(self, mode: bool = True):
        """
        This function runs when model.train() is invoked. It is used to prepare the linear layer for training
        """

        self.training = mode
        if mode and self.merged:
            warnings.warn("Invoke module.train() would unmerge LoRA weights.")
            raise NotImplementedError("LoRA unmerge is not tested.")
        elif not mode and not self.merged and lora_manager.able_to_merge:
            warnings.warn("Invoke module.eval() would merge LoRA weights.")
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.lora_A.t() @ self.lora_B.t() * self.scaling
                delattr(self, "lora_A")
                delattr(self, "lora_B")
            self.merged = True

        return self


def _lora_linear_wrapper(linear: nn.Linear, lora_config: LoraConfig) -> LoraLinear:
    """
    Wraps a linear layer with LoRA functionality.

    Args:
        linear (nn.Linear): The linear layer to be wrapped.
        lora_rank (int): The rank of the LoRA decomposition.
        lora_train_bias (str): Whether to train the bias. Can be "none", "all", "lora".
        lora_initialization_method (str): The initialization method for LoRA. Can be "kaiming_uniform" or "PiSSA".

    Returns:
        LoraLinear: The wrapped linear layer with LoRA functionality.
    """
    assert (
        lora_config.r <= linear.in_features
    ), f"LoRA rank ({lora_config.r}) must be less than or equal to in features ({linear.in_features})"
    bias = None
    if lora_config.lora_train_bias in ["all", "lora"]:
        bias = linear.bias
        if bias is None:
            bias = True
    lora_linear = LoraLinear(
        linear.weight, bias, r=lora_config.r, lora_initialization_method=lora_config.lora_initialization_method
    )
    return lora_linear


def _convert_to_lora_recursively(module: nn.Module, parent_name: str, lora_config: LoraConfig) -> None:
    """
    Recursively converts the given module and its children to LoRA (Low-Rank Approximation) form.

    Args:
        module (nn.Module): The module to convert to LoRA form.
        lora_rank (int): The rank of the LoRA approximation.
        lora_train_bias (str): Whether to train the bias. Can be "none", "all", "lora".
        parent_name (str): The name of the parent module.
        lora_initialization_method (str): The initialization method for LoRA. Can be "kaiming_uniform" or "PiSSA".

    Returns:
        None
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if lora_config.target_modules is None or any(
                [name in target_module for target_module in lora_config.target_modules]
            ):
                if dist.is_initialized() and dist.get_rank() == 0:
                    logger.info(f"Converting {parent_name}.{name} to LoRA")
                setattr(module, name, _lora_linear_wrapper(child, lora_config))
        elif isinstance(child, nn.Embedding):
            if lora_config.target_modules is None or any(
                [name in target_module for target_module in lora_config.target_modules]
            ):
                if dist.is_initialized() and dist.get_rank() == 0:
                    logger.info(f"Converting {parent_name}.{name} to LoRA")
                setattr(
                    module,
                    name,
                    LoraEmbedding(
                        child.weight,
                        r=lora_config.r,
                        lora_alpha=lora_config.lora_alpha,
                        lora_dropout=lora_config.embedding_lora_dropout,
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        padding_idx=child.padding_idx,
                        max_norm=child.max_norm,
                        norm_type=child.norm_type,
                        scale_grad_by_freq=child.scale_grad_by_freq,
                        sparse=child.sparse,
                        lora_initialization_method=lora_config.lora_initialization_method,
                    ),
                )
        else:
            _convert_to_lora_recursively(child, f"{parent_name}.{name}", lora_config)


def convert_to_lora_module(module: nn.Module, lora_config: LoraConfig) -> nn.Module:
    """Convert a torch.nn.Module to a LoRA module.

    Args:
        module (nn.Module): The module to convert.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): Whether to train the bias. Can be "none", "all", "lora".
        lora_initialization_method (str): The initialization method for LoRA. Can be "kaiming_uniform" or "PiSSA".

    Returns:
        nn.Module: The converted module.
    """
    if lora_config.r <= 0:
        return module
    # make all parameter not trainable, if lora_train_bias is "all", set bias to trainable
    total_parameter_size = 0
    for name, p in module.named_parameters():
        p.requires_grad = False
        if "bias" in name and lora_config.lora_train_bias == "all":
            p.requires_grad = True
        total_parameter_size += p.numel()
    _convert_to_lora_recursively(module, "", lora_config)
    trainable_parameter_size = 0
    for name, p in module.named_parameters():
        if p.requires_grad == True:
            trainable_parameter_size += p.numel()
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(
            f"Trainable parameter size: {trainable_parameter_size/1024/1024:.2f}M\nOriginal trainable parameter size: {total_parameter_size/1024/1024:.2f}M\nPercentage: {trainable_parameter_size/total_parameter_size*100:.2f}%"
        )
    return module
