import warnings

HAS_MEM_EFF_ATTN = False
try:
    from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp, memory_efficient_attention
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask,
        BlockDiagonalMask,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
    )
    HAS_MEM_EFF_ATTN = True
except ImportError:
    warnings.warn('please install xformers from https://github.com/facebookresearch/xformers')
    HAS_MEM_EFF_ATTN = False

if HAS_MEM_EFF_ATTN:
    """
    A general attention module using the flash attention kernels from xformers:
    https://github.com/facebookresearch/xformers/tree/main/xformers/ops/fmha
    """
    from typing import Optional

    import torch

    from .utils import SeqLenInfo

    allow_alibi = True
    for op in MemoryEfficientAttentionCutlassOp:
        allow_alibi = allow_alibi & (LowerTriangularMaskWithTensorBias in op.SUPPORTED_ATTN_BIAS_TYPES)

    def mem_eff_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          seq_len_info_q: SeqLenInfo,
                          seq_len_info_kv: SeqLenInfo,
                          bias: Optional[torch.Tensor] = None,
                          dropout_p: float = 0.,
                          scale: float = None,
                          causal: bool = False,
                          padded: bool = False):

        attn_bias = None
        if padded:    # bert style
            if not causal:
                attn_bias = BlockDiagonalMask.from_seqlens(seq_len_info_q.seqlens, seq_len_info_kv.seqlens)
            else:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_len_info_q.seqlens, seq_len_info_kv.seqlens)
        elif causal:    # gpt style
            attn_bias = LowerTriangularMask()

        if bias is not None:    # alibi / relative position embedding
            assert allow_alibi, "flash attention with bias is not supported in this system."
            assert causal, \
                "attention with bias is only supported for causal attention so far."
            attn_bias = attn_bias.add_bias(bias)

        if padded:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p, scale=scale)

        # shape: (b*s, n, d)
        if padded:
            out = out.squeeze(0)

        return out
