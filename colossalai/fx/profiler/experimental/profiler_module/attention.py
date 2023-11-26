from typing import Optional, Tuple

import torch

from ..registry import meta_profiler_module


# TODO: This is hard to compute memory cost
@meta_profiler_module.register(torch.nn.MultiheadAttention)
def torch_nn_msa(
    self: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[int, int]:
    if getattr(self, "batch_first", False):
        batch_size = query.shape[0]
        len_idx = 1
    else:
        batch_size = query.shape[1]
        len_idx = 0
    dim_idx = 2

    qdim = query.shape[dim_idx]
    kdim = key.shape[dim_idx]
    vdim = value.shape[dim_idx]

    qlen = query.shape[len_idx]
    klen = key.shape[len_idx]
    vlen = value.shape[len_idx]

    num_heads = self.num_heads
    assert qdim == self.embed_dim

    if self.kdim is None:
        assert kdim == qdim
    if self.vdim is None:
        assert vdim == qdim

    flops = 0
    macs = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += 2 * ((qlen * qdim * qdim) + (klen * kdim * kdim) + (vlen * vdim * vdim))  # QW  # KW  # VW

    macs += (qlen * qdim * qdim) + (klen * kdim * kdim) + (vlen * vdim * vdim)  # QW  # KW  # VW

    if self.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        2 * (qlen * klen * qk_head_dim) + (qlen * klen) + 2 * (qlen * klen * v_head_dim)  # QK^T  # softmax  # AV
    )
    head_macs = (qlen * klen * qk_head_dim) + 2 * (qlen * klen * v_head_dim)  # QK^T  # AV

    flops += num_heads * head_flops
    macs += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    macs *= batch_size
    return flops, macs
