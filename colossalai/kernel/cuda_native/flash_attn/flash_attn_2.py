from .version_available import HAS_FLASH_ATTN

if HAS_FLASH_ATTN:
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    # from flash_attn.flash_attention import FlashAttention
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )

    def flash_attention_qkv(qkv, sm_scale, batch_size, seq_len, dropout_p=0., causal=False):
        """
        Arguments:
            qkv: (batch * seqlen, 3, nheads, headdim)
            batch_size: int.
            seq_len: int.
            sm_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            dropout_p: float.
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        Return:
            out: (total, nheads, headdim).
        """
        if cu_seqlens == None:
            max_s = seq_len
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_s, dropout_p, softmax_scale=sm_scale, causal=causal)
        return out

    def flash_attention_q_kv(q, kv, sm_scale, batch_size, q_seqlen, kv_seqlen, dropout_p=0., causal=False):
        """
        Arguments:
            q: (batch * q_seqlen, nheads, headdim)
            kv: (batch * kv_seqlen, 2, nheads, headdim)
            batch_size: int.
            seq_len: int.
            sm_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            dropout_p: float.
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        Return:
            out: (total, nheads, headdim).
        """
        if cu_seqlens_q == None:
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_seqlen,
                                        step=q_seqlen,
                                        dtype=torch.int32,
                                        device=q.device)
        if cu_seqlens_k == None:
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * kv_seqlen,
                                        step=kv_seqlen,
                                        dtype=torch.int32,
                                        device=kv.device)
        out = flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, q_seqlen, kv_seqlen, dropout_p,
                                              sm_scale, causal)
        return out

    def flash_attention_q_k_v(q, k, v, sm_scale, cu_seqlens_q, cu_seqlens_kv, dropout_p=0., causal=False):
        """
        Arguments:
            q: (batch, q_seqlen, nheads, headdim)
            k: (batch, kv_seqlen, nheads, headdim)
            v: (batch, kv_seqlen, nheads, headdim)
            batch_size: int.
            seq_len: int.
            dropout_p: float. Dropout probability.
            sm_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        Return:
            attn_out: (batch, q_seqlen, nheads, headdim).
        """
        if cu_seqlens_q == None:
            return flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=sm_scale, causal=causal)

        q = rearrange(q, 'b s ... -> (b s) ...')
        k = rearrange(k, 'b s ... -> (b s) ...')
        v = rearrange(v, 'b s ... -> (b s) ...')
        cu_seqlens_q = torch.Tensor(cu_seqlens_q).to(torch.int32).to(q.device)
        max_seqlen_q = cu_seqlens_q.max().item()
        cu_seqlens_q = F.pad(torch.cumsum(cu_seqlens_q, dim=0, dtype=torch.torch.int32), (1, 0))
        cu_seqlens_kv = torch.Tensor(cu_seqlens_kv).to(torch.int32).to(k.device)
        max_seqlen_kv = cu_seqlens_kv.max().item()
        cu_seqlens_kv = F.pad(torch.cumsum(cu_seqlens_kv, dim=0, dtype=torch.torch.int32), (1, 0))

        attn_out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, dropout_p,
                                          sm_scale, causal)
        attn_out = rearrange(attn_out, '(b s) ... -> b s ...', b=1)
        return attn_out
