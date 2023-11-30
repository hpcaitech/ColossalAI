import math
from typing import Optional

import torch

from .sdpa_attn import npu_sdpa_attention
from .triangle_attn import HAS_NPU_TRIANGLE_ATTENTION


class NPUColoAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, scale: float = None):
        super().__init__()

        try:
            import torch_npu  # noqa
        except ImportError:
            raise Exception("torch_npu is not installed.")

        assert (
            embed_dim % num_heads == 0
        ), f"the embed dim ({embed_dim}) is not divisible by the number of attention heads ({num_heads})."
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(embed_dim // num_heads)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        origin_attn_mask: Optional[torch.Tensor] = None,
        attn_mask_type: int = None,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Implement the scaled dot product attention with softmax.

        Arguments:
            q: (batch, q_seqlen, nheads, headdim)
            k: (batch, kv_seqlen, nheads, headdim)
            v: (batch, kv_seqlen, nheads, headdim)
            batch_size: int.
            seq_len: int.
            dropout_p: float. Dropout probability.
            scale: float. The scaling of QK^T before applying softmax.
                Default to 1.
        Return:
            attn_out: (batch, q_seqlen, nheads, headdim).
        """
        assert (
            len(query.shape) == 4 and len(key.shape) == 4 and len(value.shape) == 4
        ), f"query, key, value should be 4D tensors, but got {query.shape}, {key.shape}, {value.shape}"
        assert (
            query.device.type == "npu" and key.device.type == "npu" and value.device.type == "npu"
        ), f"query, key, value should be on npu device, but got {query.device}, {key.device}, {value.device}"
        assert bias is None, "bias is not supported in npu colo attention"

        causal = attn_mask_type is not None and attn_mask_type.value > 1

        if HAS_NPU_TRIANGLE_ATTENTION:
            from .triangle_attn import npu_triangle_attention

            attn_fn = npu_triangle_attention
        else:
            attn_fn = npu_sdpa_attention

        out = attn_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            origin_attn_mask=origin_attn_mask,
            dropout_p=self.dropout,
            scale=self.scale,
            is_causal=causal,
        )
        return out
