from .version_available import HAS_FLASH_ATTN

if HAS_FLASH_ATTN:
    import torch
    # from flash_attn.flash_attention import FlashAttention
    from flash_attn.flash_attn_interface import (
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
            q: (batch * q_seqlen, nheads, headdim)
            k: (batch * kv_seqlen, nheads, headdim)
            v: (batch * kv_seqlen, nheads, headdim)
            batch_size: int.
            seq_len: int.
            dropout_p: float. Dropout probability.
            sm_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        Return:
            out: (total, nheads, headdim).
        """
        import pathlib
        pathlib.Path(
            "/home/lcjmy/code/personal/ColossalAI/colossalai/kernel/cuda_native/ops/flash_attention_2.txt").write_text(
                str(q))
        batch_size, q_seqlen, _, _ = q.shape
        _, kv_seqlen, _, _ = k.shape
        if cu_seqlens_q == None:
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_seqlen,
                                        step=q_seqlen,
                                        dtype=torch.int32,
                                        device=q.device)
        if cu_seqlens_kv == None:
            cu_seqlens_kv = torch.arange(0, (batch_size + 1) * kv_seqlen,
                                         step=kv_seqlen,
                                         dtype=torch.int32,
                                         device=k.device)
        return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_seqlen, kv_seqlen, dropout_p, sm_scale,
                                      causal)

    # def flash_attention(query, key, value, cu_seqlens_q, cu_seqlens_kv, sm_scale, dropout_p, causal):
    #     return flash_attention_q_k_v(query, key, value, sm_scale, cu_seqlens_q, cu_seqlens_kv, dropout_p, causal),
