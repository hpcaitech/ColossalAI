#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings
import torch.nn as nn
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, Module
from typing import Tuple, Iterator, Set, Optional, Mapping, List, Any
from torch.nn import Parameter
from collections import OrderedDict
from torch.nn.modules.module import _IncompatibleKeys

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

class LayerNormBase(nn.Module):

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

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        r"""Don't recursive process self._module
        """

        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
    
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        r"""Don't recursive process self._module
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

class FusedLayerNorm(LayerNormBase):
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


class FusedRMSNorm(LayerNormBase):
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
