import torch
from einops import rearrange


def npu_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor = None,
    origin_attn_mask: torch.Tensor = None,
    scale: float = 1.0,
    dropout_p: float = 0.0,
    is_causal: bool = True,
):
    """
    The scaled dot product attention.

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
    q, k, v = [rearrange(x, "b s h d -> b h s d").contiguous() for x in (q, k, v)]
    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=origin_attn_mask,
        dropout_p=dropout_p,
        is_causal=origin_attn_mask is None,
        scale=scale,
    )
    output = rearrange(output, "b h s d -> b s (h d)")
    return output
