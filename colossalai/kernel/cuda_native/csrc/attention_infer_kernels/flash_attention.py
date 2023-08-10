try:
    from col_flash_attn_2_lib import flash_fwd, varlen_flash_fwd
    HAS_FLASH_CUDA = True
except:
    HAS_FLASH_CUDA = False
    print("in order to use flash-attention, make sure you install cuda kernels in op directory")


if HAS_FLASH_CUDA:
    def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_fwd(
            q, k, v, None, dropout_p, softmax_scale, causal, return_softmax, None
        )
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
    
    def _flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               dropout_p, softmax_scale, causal, return_softmax):
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = varlen_flash_fwd(
            q, k, v, None, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale, False, causal, return_softmax, None
        )
        
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


    def flash_attention_fwd(qkv, scale, causal = True, return_softmax = False):
        assert qkv.is_contiguous()
        batches = qkv.shape[0]
        d_model = qkv.shape[-1] // 3
        num_of_heads = d_model // head_size

        q = qkv[:, :, :d_model]
        k = qkv[:, :, d_model:d_model * 2]
        v = qkv[:, :, d_model * 2:]
        q = q.view(batches, -1, num_of_heads, head_size)
        k = k.view(batches, -1, num_of_heads, head_size)
        v = v.view(batches, -1, num_of_heads, head_size)

        out_flash, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(q, k, v, 0, 
                                softmax_scale = scale, 
                                causal = causal, 
                                return_softmax = return_softmax
                                )
        
        if return_softmax:
            return out_flash, softmax_lse
        else:
            return out_flash
