#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.kernel import LayerNorm
from colossalai.nn import init as init

from .parallel_module import ParallelModule

__all__ = ['LayerNorm1D']

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


class LayerNorm1D(ParallelModule):
    r"""
    Layer Normalization for colossalai

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability, defaults to 1e-05.
        bias (bool, optional): Whether to add a bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    _fast_ln_supported_sizes = [
        1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
        24576, 25600, 30720, 32768, 40960, 49152, 65536
    ]

    def __init__(self,
                 normalized_shape: int,
                 eps: int = 1e-05,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        if Fast_LN is not None and normalized_shape in self._fast_ln_supported_sizes:
            norm = Fast_LN(normalized_shape, eps=eps).to(dtype)
        else:
            norm = None
            try:
                from apex.normalization import FusedLayerNorm
                norm = FusedLayerNorm(normalized_shape, eps=eps).to(dtype)
            except ImportError:
                norm = LayerNorm(normalized_shape, eps=eps, device=device, dtype=dtype)
        self.norm = norm

    @staticmethod
    def from_native_module(module: nn.LayerNorm, process_group: Union[ProcessGroup, List[ProcessGroup]], *args,
                           **kwargs) -> ParallelModule:
        r"""
        Convert a native pytorch layer norm module to colossalai layer norm module
        """
        normalized_shape = module.normalized_shape
        eps = module.eps
        bias = module.bias is not None
        dtype = module.weight.dtype
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, \
                f'Expected only one process group, got {len(process_group)}.'
            process_group = process_group[0]

        # create layer norm
        layer_norm = LayerNorm1D(normalized_shape, eps=eps, bias=bias, device=device, dtype=dtype).norm

        with torch.no_grad():
            # copy weight and bias
            layer_norm.weight.copy_(module.weight)
            if bias:
                layer_norm.bias.copy_(module.bias)
        return layer_norm
