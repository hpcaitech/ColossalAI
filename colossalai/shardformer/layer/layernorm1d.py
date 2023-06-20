#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from collections import OrderedDict

from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.kernel import LayerNorm
from colossalai.nn import init as init
from colossalai.nn.layer.colossalai_layer._utils import ColossalaiModule
from colossalai.utils.checkpointing import broadcast_state_dict

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


class LayerNorm1D(ColossalaiModule):
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

    def __init__(self, normalized_shape: int, eps=1e-05, bias=True, dtype=None):
        if Fast_LN is not None and normalized_shape in self._fast_ln_supported_sizes:
            norm = Fast_LN(normalized_shape, eps=eps).to(dtype)
        else:
            norm = None
            try:
                from apex.normalization import FusedLayerNorm
                norm = FusedLayerNorm(normalized_shape, eps=eps).to(dtype)
            except ImportError:
                norm = LayerNorm(normalized_shape, eps=eps).to(dtype)
        super().__init__(norm)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias

        local_state = broadcast_state_dict(local_state, ParallelMode.PARALLEL_1D)
        super()._load_from_state_dict(local_state, prefix, *args)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            super()._save_to_state_dict(destination, prefix, keep_vars)
