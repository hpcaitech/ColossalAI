# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .fmha import (
    AttentionBias,
    AttentionOp,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
    MemoryEfficientAttentionCutlassFwdFlashBwOp,
    MemoryEfficientAttentionCutlassOp,
    MemoryEfficientAttentionFlashAttentionOp,
    MemoryEfficientAttentionOp,
    #MemoryEfficientAttentionTritonFwdFlashBwOp,
    #TritonFlashAttentionOp,
    memory_efficient_attention,
    memory_efficient_attention_backward,
    memory_efficient_attention_forward,
    memory_efficient_attention_forward_requires_grad,
)
from .indexing import index_select_cat, scaled_index_add
from .swiglu_op import (
    SwiGLU,
    SwiGLUEagerOp,
    SwiGLUFusedOp,
    SwiGLUOp,
    SwiGLUOpDispatch,
    SwiGLUPackedFusedOp,
    swiglu,
)
from .unbind import get_stack_strides, stack_or_none, unbind

# BW compatibility
AttentionMask = AttentionBias


def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(
            masked_matmul, (a, b, mask), a, b, mask
        )

    att = a @ b

    if mask is None:
        return att

    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        att += mask 
    return att


__all__ = [
    "memory_efficient_attention",
    "AttentionBias",
    "AttentionMask",
    "AttentionOp",
    "AttentionOpBase",
    "AttentionOpDispatch",
    "LowerTriangularMask",
    "MemoryEfficientAttentionCutlassFwdFlashBwOp",
    "MemoryEfficientAttentionCutlassOp",
    "MemoryEfficientAttentionFlashAttentionOp",
    "MemoryEfficientAttentionOp",
    #"MemoryEfficientAttentionTritonFwdFlashBwOp",
    "memory_efficient_attention_backward",
    "memory_efficient_attention_forward",
    "memory_efficient_attention_forward_requires_grad",
    "SwiGLU",
    "SwiGLUEagerOp",
    "SwiGLUFusedOp",
    "SwiGLUOp",
    "SwiGLUOpDispatch",
    "SwiGLUPackedFusedOp",
    "swiglu",
    #"TritonFlashAttentionOp",
    "unbind",
    "stack_or_none",
    "get_stack_strides",
    "masked_matmul",
    "scaled_index_add",
    "index_select_cat",
]
