from .version_available import HAS_MEM_EFF_ATTN

if HAS_MEM_EFF_ATTN:
    """
    A general attention module using the flash attention kernels from xformers:
    https://github.com/facebookresearch/xformers/tree/main/xformers/ops/fmha
    """
    import math
    from typing import Optional

    import torch
    from einops import rearrange
    from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp, memory_efficient_attention
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask,
        BlockDiagonalMask,
        LowerTriangularMask,
        LowerTriangularMaskWithTensorBias,
    )

    from ..scaled_softmax import AttnMaskType
    from .padding import Repad, Unpad

    allow_alibi = True
    for op in MemoryEfficientAttentionCutlassOp:
        allow_alibi = allow_alibi & (LowerTriangularMaskWithTensorBias in op.SUPPORTED_ATTN_BIAS_TYPES)

    def mem_eff_attention(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          q_seqlen,
                          kv_seqlen,
                          attn_mask_type: Optional[AttnMaskType] = None,
                          bias: Optional[torch.Tensor] = None,
                          dropout=0.0,
                          scale=None):

        # batch_size, tgt_len, src_len = query.shape[0], query.shape[1], key.shape[1]
        attn_bias = None
        if attn_mask_type and attn_mask_type.value % 2 == 1:    # bert style
            if attn_mask_type == AttnMaskType.padding:
                attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
            elif attn_mask_type == AttnMaskType.paddedcausal:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(q_seqlen, kv_seqlen)
        elif attn_mask_type == AttnMaskType.causal:    # gpt style
            attn_bias = LowerTriangularMask()

        if bias is not None:    # alibi / relative position embedding
            assert allow_alibi, "flash attention with bias is not supported in this system."
            assert attn_mask_type == AttnMaskType.causal, \
                "attention with bias is only supported for causal attention so far."
            attn_bias = attn_bias.add_bias(bias)

        out = memory_efficient_attention(query, key, value, attn_bias=attn_bias, p=dropout, scale=scale)

        return out
