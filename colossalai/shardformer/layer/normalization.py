#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, Module
from typing import Tuple, Iterator, Set, Optional
from torch.nn import Parameter

from colossalai.lazy import LazyInitContext
from ._operation import hook_paramter_in_backward

__all__ = ["FusedLayerNorm", "FusedRMSNorm"]

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

class FusedLayerNorm(nn.Module):
    r"""
    This is a wrapper around the apex fused layernorm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self, layernorm=None) -> None:
        super().__init__()
        self.layernorm = layernorm
        self._parameters = layernorm._parameters
        self._buffers = layernorm._buffers

    @staticmethod
    def from_native_module(module: nn.LayerNorm, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native pytorch layer norm module to colossalai layer norm module
        """
        # check if apex is installed
        try:
            pass
        except ImportError:
            raise ImportError(
                "Please install apex from source (https://github.com/NVIDIA/apex) to use the fused layernorm kernel"
            )

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
            try:
                from apex.contrib.layer_norm.layer_norm import FastLayerNorm as ApexFusedLayerNorm
            except ImportError:
                # fall back to the normal fused layernorm is not built
                from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
        else:
            from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm

        layernorm = (
            ApexFusedLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine).to(dtype).to(device)
        )

        layernorm.weight = module.weight
        layernorm.bias = module.bias

        return FusedLayerNorm(layernorm=layernorm)

    def forward(self, input):
        weight = self.layernorm.weight
        bias = self.layernorm.bias
        layernorm_output = self.layernorm(input)
        output = hook_paramter_in_backward(layernorm_output, weight, bias)
        return output

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=False)
        for elem in gen:
            yield elem

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self


class FusedRMSNorm(nn.Module):
    """
    This is a wrapper around the apex fused rms norm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self, rmsnorm=None) -> None:
        super().__init__()
        self.rmsnorm = rmsnorm
        self._parameters = rmsnorm._parameters
        self._buffers = rmsnorm._buffers

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        try:
            from apex.normalization import FusedRMSNorm as ApexFusedRMSNorm
        except ImportError:
            raise ImportError(
                "Please install apex from source (https://github.com/NVIDIA/apex) to use the fused RMS normalization kernel"
            )

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

        rmsnorm = ApexFusedRMSNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

        rmsnorm.weight = module.weight

        return FusedRMSNorm(rmsnorm=rmsnorm)
    
    def forward(self, input):
        weight = self.rmsnorm.weight
        rmsnorm_output = self.rmsnorm(input)
        output = hook_paramter_in_backward(rmsnorm_output, weight)
        return output
    

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=False)
        for elem in gen:
            yield elem

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
