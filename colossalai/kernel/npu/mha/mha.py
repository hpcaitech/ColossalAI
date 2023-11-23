import math
from typing import Optional

import torch
from einops import rearrange

from .fused_attn import HAS_NPU_FUSED_ATTN

if HAS_NPU_FUSED_ATTN:
    from .fused_attn import npu_fused_attention


class NPUColoAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, scale=None):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"the embed dim ({embed_dim}) is not divisible by the number of attention heads ({num_heads})."
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(embed_dim // num_heads)
        self.dropout = dropout

        if not HAS_NPU_FUSED_ATTN:
            raise Exception("npu attention kernel can not support!")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_mask_type: int = None,
        bias: Optional[torch.Tensor] = None,
    ):
        if HAS_NPU_FUSED_ATTN and query.dtype in [torch.float16, torch.bfloat16] and bias == None:
            attn = npu_fused_attention
        else:
            raise Exception("npu attention kernel can not support!")

        out = attn(
            query,
            key,
            value,
            attention_mask=attn_mask,
            dropout_p=self.dropout,
            scale=self.scale,
        )
        out = rearrange(out, "b s h d -> b s (h d)")
        return out
