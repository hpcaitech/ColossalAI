"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf; Triton https://github.com/openai/triton)
"""

import torch
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func
except ImportError:
    raise ImportError('please install flash_attn from https://github.com/HazyResearch/flash-attention')



def flash_attention_qkv(qkv, sm_scale, batch_size, seq_len):
    """
    Arguments:
        qkv: (batch*seq, 3, nheads, headdim)
        batch_size: int.
        seq_len: int.
        sm_scale: float. The scaling of QK^T before applying softmax.
    Return:
        out: (total, nheads, headdim).
    """
    max_s = seq_len
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32,
        device=qkv.device)
    out = flash_attn_unpadded_qkvpacked_func(
        qkv, cu_seqlens, max_s, 0.0,
        softmax_scale=sm_scale, causal=False
    )
    return out


def flash_attention_q_kv(q, kv, sm_scale, batch_size, q_seqlen, kv_seqlen):
    """
    Arguments:
        q: (batch*seq, nheads, headdim)
        kv: (batch*seq, 2, nheads, headdim)
        batch_size: int.
        seq_len: int.
        sm_scale: float. The scaling of QK^T before applying softmax.
    Return:
        out: (total, nheads, headdim).
    """
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_seqlen, step=q_seqlen, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * kv_seqlen, step=kv_seqlen, dtype=torch.int32, device=kv.device)
    out = flash_attn_unpadded_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, q_seqlen, kv_seqlen, 0.0, sm_scale)
    return out
