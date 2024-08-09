#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import warnings
from abc import ABC, abstractmethod

import torch.nn as nn

from colossalai.lazy import LazyInitContext

from ._operation import hook_parameter_in_backward
from .utils import SeqParallelUtils

__all__ = ["FusedLayerNorm", "FusedRMSNorm", "LayerNorm", "RMSNorm", "BaseLayerNorm"]

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm

    EnableFastLayerNorm = True
except ImportError:
    EnableFastLayerNorm = False

try:
    from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
    from apex.normalization import FusedRMSNorm as ApexFusedRMSNorm

    class FusedLayerNormWithHook(ApexFusedLayerNorm):
        def __init__(self, normalized_shape, eps=0.00001, elementwise_affine=True):
            super().__init__(normalized_shape, eps, elementwise_affine)

        def forward(self, input):
            output = super().forward(input)
            output = hook_parameter_in_backward(output, self.weight, self.bias)
            return output

    class FusedRMSNormWithHook(ApexFusedRMSNorm):
        def __init__(self, normalized_shape, eps=0.00001, elementwise_affine=True):
            super().__init__(normalized_shape, eps, elementwise_affine)

        def forward(self, input):
            output = super().forward(input)
            output = hook_parameter_in_backward(output, self.weight)
            return output

except ImportError:
    warnings.warn("Please install apex from source (https://github.com/NVIDIA/apex) to use the fused RMSNorm kernel")

FAST_LAYERNORM_SUPPORTED_SIZE = [
    1024,
    1536,
    2048,
    2304,
    3072,
    3840,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    12800,
    15360,
    16384,
    18432,
    20480,
    24576,
    25600,
    30720,
    32768,
    40960,
    49152,
    65536,
]

if EnableFastLayerNorm:

    class FastLayerNormWithHook(FastLayerNorm):
        def __init__(self, hidden_size, eps=0.00001):
            super().__init__(hidden_size, eps)

        def forward(self, input):
            output = super().forward(input)
            output = hook_parameter_in_backward(output, self.weight, self.bias)
            return output


class BaseLayerNorm(ABC):
    @abstractmethod
    def from_native_module(module: nn.Module, sp_partial_derived: bool = False):
        """
        Convert a native PyTorch layer normalization module to a specific layer normalization module,
        and optionally mark parameters for gradient aggregation.

        Args:
            module (nn.Module): The native PyTorch layer normalization module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The specific layer normalization module.

        Raises:
            AssertionError: If the provided module is not an instance of the supported layer normalization type.
        """


class RMSNorm(BaseLayerNorm):
    r"""
    This is a wrapper around the RMSNorm. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedLayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to convert a native RMSNorm module to colossalai layer norm module."
        )

    @staticmethod
    def from_native_module(module: nn.Module, sp_partial_derived: bool = False, *args, **kwargs) -> nn.Module:
        """
        Convert a native RMSNorm module to colossalai layer norm module,
        and optionally mark parameters for gradient aggregation.

        Args:
            module (nn.Module): The native RMSNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The RMSNorm module.
        """

        LazyInitContext.materialize(module)

        if sp_partial_derived:
            # Since gradients are computed using only a subset of the data,
            # aggregation of these gradients is necessary during backpropagation.
            # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.weight)

        return module


class LayerNorm(BaseLayerNorm):
    r"""
    This is a wrapper around native LayerNorm. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "LayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to convert a native LayerNorm module to colossalai layer norm module."
        )

    @staticmethod
    def from_native_module(module: nn.Module, sp_partial_derived: bool = False, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native LayerNorm module to colossalai layer norm module,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.Module): The native LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The colossalai LayerNorm module.

        """

        LazyInitContext.materialize(module)

        if sp_partial_derived:
            # Since gradients are computed using only a subset of the data,
            # aggregation of these gradients is necessary during backpropagation.
            # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.weight)
            if module.bias is not None:
                SeqParallelUtils.marked_as_sp_partial_derived_param(module.bias)

        return module


class FusedLayerNorm(BaseLayerNorm):
    r"""
    This is a wrapper around the apex fused layernorm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedLayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface convert a native LayerNorm module to FusedLayerNorm module provided by apex."
        )

    @staticmethod
    def from_native_module(module: nn.LayerNorm, sp_partial_derived: bool = False, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native LayerNorm module to FusedLayerNorm module provided by apex,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.Module): The native LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: Union[FastLayerNorm, FusedLayerNorm].

        """

        LazyInitContext.materialize(module)
        # get the attributes of the module
        normalized_shape = getattr(module, "normalized_shape", module.weight.shape[0])
        eps = module.variance_epsilon if hasattr(module, "variance_epsilon") else module.eps
        elementwise_affine = getattr(module, "elementwise_affine", True)
        dtype = module.weight.dtype
        device = module.weight.device

        # pick the suitable layernorm implementation
        use_fast_ln = normalized_shape in FAST_LAYERNORM_SUPPORTED_SIZE

        if use_fast_ln:
            if EnableFastLayerNorm:
                ApexFusedLayerNorm = FastLayerNormWithHook
            else:
                # fall back to the normal fused layernorm is not built
                ApexFusedLayerNorm = FusedLayerNormWithHook
        else:
            try:
                ApexFusedLayerNorm = FusedLayerNormWithHook
            except NameError:
                warnings.warn(
                    "Please install Apex from source to use fused kernels, or set self.enable_fused_normalization = False. Using native layernorm instead."
                )
                return module

        layernorm = (
            ApexFusedLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype).to(device)
        )
        layernorm.weight = module.weight
        if module.bias is not None:
            layernorm.bias = module.bias

        if sp_partial_derived:
            # Since gradients are computed using only a subset of the data,
            # aggregation of these gradients is necessary during backpropagation.
            # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
            SeqParallelUtils.marked_as_sp_partial_derived_param(layernorm.weight)
            SeqParallelUtils.marked_as_sp_partial_derived_param(layernorm.bias)

        return layernorm


class FusedRMSNorm(BaseLayerNorm):
    """
    This is a wrapper around the apex fused rms norm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedRMSNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native RMSNorm module to FusedRMSNorm module provided by apex."
        )

    @staticmethod
    def from_native_module(module: nn.Module, sp_partial_derived: bool = False, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native RMSNorm module module to FusedRMSNorm module provided by apex,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.LayerNorm): The native PyTorch LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: FusedRMSNorm module.
        """

        LazyInitContext.materialize(module)

        # try to get normalized_shape, eps, elementwise_affine from the module
        normalized_shape = getattr(module, "normalized_shape", module.weight.shape[0])
        eps = module.variance_epsilon if hasattr(module, "variance_epsilon") else module.eps
        elementwise_affine = getattr(module, "elementwise_affine", True)

        try:
            rmsnorm = FusedRMSNormWithHook(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        except ImportError:
            warnings.warn(
                "Module replacement failed.\
                Please install apex from source (https://github.com/NVIDIA/apex) to use the fused RMS normalization kernel"
            )
            return module

        rmsnorm.weight = module.weight

        if sp_partial_derived:
            # Since gradients are computed using only a subset of the data,
            # aggregation of these gradients is necessary during backpropagation.
            # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
            SeqParallelUtils.marked_as_sp_partial_derived_param(rmsnorm.weight)

        return rmsnorm
