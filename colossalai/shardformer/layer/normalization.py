#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import warnings
import torch.nn as nn
from colossalai.lazy import LazyInitContext
from ._operation import hook_paramter_in_backward

__all__ = ["FusedLayerNorm", "FusedRMSNorm"]

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    EnableFastLayerNorm = True
except ImportError:
    EnableFastLayerNorm = False

try:
    from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
    from apex.normalization import FusedRMSNorm as ApexFusedRMSNorm
except ImportError:
    warnings.warn(
        "Please install apex from source (https://github.com/NVIDIA/apex) to use the fused layernorm kernel"
    )

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
            output = hook_paramter_in_backward(output, self.weight, self.bias)
            return output
        
class FusedLayerNormWithHook(ApexFusedLayerNorm):
    def __init__(self, normalized_shape, eps=0.00001, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input):
        output = super().forward(input)
        output = hook_paramter_in_backward(output, self.weight, self.bias)
        return output

class FusedRMSNormWithHook(ApexFusedRMSNorm):
    def __init__(self, normalized_shape, eps=0.00001, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input):
        output = super().forward(input)
        output = hook_paramter_in_backward(output, self.weight)
        return output
        

class FusedLayerNorm:
    r"""
    This is a wrapper around the apex fused layernorm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedLayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to wrap the fused layernorm implementation provided by apex."
        )

    @staticmethod
    def from_native_module(module: nn.LayerNorm, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native pytorch layer norm module to colossalai layer norm module
        """

        LazyInitContext.materialize(module)
        # get the attributes of the module
        normalized_shape = module.normalized_shape
        eps = module.eps
        elementwise_affine = module.elementwise_affine
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
            ApexFusedLayerNorm = FusedLayerNormWithHook

        layernorm = (
            ApexFusedLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype).to(device)
        )
        layernorm.weight = module.weight
        layernorm.bias = module.bias

        return layernorm


class FusedRMSNorm:
    """
    This is a wrapper around the apex fused rms norm implementation. It is meant to be used only with the from_native_module interface.
    """
    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedRMSNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to wrap the fused rms norm implementation provided by apex."
        )
    
    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        LazyInitContext.materialize(module)
        # to check if it is huggingface LlamaRMSNorm
        if module.__class__.__name__ == "LlamaRMSNorm":
            normalized_shape = module.weight.shape[0]
            eps = module.variance_epsilon
            elementwise_affine = True
        else:
            # get the attributes of the module
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine

        rmsnorm = FusedRMSNormWithHook(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

        rmsnorm.weight = module.weight

        return rmsnorm
