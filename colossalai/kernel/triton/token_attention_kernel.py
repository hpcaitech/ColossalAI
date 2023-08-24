# Adapted from ModelTC https://github.com/ModelTC/lightllm

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

if HAS_TRITON:

    @triton.jit
    def _token_attn_1_kernel(Q, K, sm_scale, kv_cache_loc, kv_cache_start_loc, kv_cache_seqlen, max_kv_cache_len,
                             attn_out, kv_cache_loc_b_stride, kv_cache_loc_s_stride, q_batch_stride, q_head_stride,
                             q_head_dim_stride, k_batch_stride, k_head_stride, k_head_dim_stride, attn_head_stride,
                             attn_batch_stride, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)
        start_n = tl.program_id(2)

        offs_d = tl.arange(0, HEAD_DIM)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        current_batch_start_index = max_kv_cache_len - current_batch_seq_len
        current_batch_end_index = max_kv_cache_len

        off_q = current_batch * q_batch_stride + current_head * q_head_stride + offs_d * q_head_dim_stride

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        block_stard_index = start_n * BLOCK_N
        block_mask = tl.where(block_stard_index < current_batch_seq_len, 1, 0)

        for start_mark in range(0, block_mask, 1):
            q = tl.load(Q + off_q + start_mark)
            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                            mask=offs_n_new < current_batch_end_index,
                            other=0)
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    @triton.jit
    def _token_attn_1_alibi_kernel(Q, K, sm_scale, alibi, kv_cache_loc, kv_cache_start_loc, kv_cache_seqlen,
                                   max_kv_cache_len, attn_out, kv_cache_loc_b_stride, kv_cache_loc_s_stride,
                                   q_batch_stride, q_head_stride, q_head_dim_stride, k_batch_stride, k_head_stride,
                                   k_head_dim_stride, attn_head_stride, attn_batch_stride, HEAD_DIM: tl.constexpr,
                                   BLOCK_N: tl.constexpr):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)
        start_n = tl.program_id(2)

        offs_d = tl.arange(0, HEAD_DIM)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        current_batch_start_index = max_kv_cache_len - current_batch_seq_len
        current_batch_end_index = max_kv_cache_len

        off_q = current_batch * q_batch_stride + current_head * q_head_stride + offs_d * q_head_dim_stride

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        block_stard_index = start_n * BLOCK_N
        block_mask = tl.where(block_stard_index < current_batch_seq_len, 1, 0)

        for start_mark in range(0, block_mask, 1):
            alibi_m = tl.load(alibi + current_head)
            q = tl.load(Q + off_q + start_mark)
            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                            mask=offs_n_new < current_batch_end_index,
                            other=0)
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            att_value -= alibi_m * (current_batch_seq_len - 1 - offs_n)
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    @torch.no_grad()
    def token_attn_fwd_1(q,
                         k,
                         attn_out,
                         kv_cache_loc,
                         kv_cache_start_loc,
                         kv_cache_seqlen,
                         max_kv_cache_len,
                         alibi=None):
        BLOCK = 32
        # shape constraints
        q_head_dim, k_head_dim = q.shape[-1], k.shape[-1]
        assert q_head_dim == k_head_dim
        assert k_head_dim in {16, 32, 64, 128}
        sm_scale = 1.0 / (k_head_dim**0.5)

        batch, head_num = kv_cache_loc.shape[0], q.shape[1]

        grid = (batch, head_num, triton.cdiv(max_kv_cache_len, BLOCK))

        num_warps = 4 if k_head_dim <= 64 else 8
        num_warps = 2

        if alibi is not None:
            _token_attn_1_alibi_kernel[grid](
                q,
                k,
                sm_scale,
                alibi,
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seqlen,
                max_kv_cache_len,
                attn_out,
                kv_cache_loc.stride(0),
                kv_cache_loc.stride(1),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                attn_out.stride(0),
                attn_out.stride(1),
                HEAD_DIM=k_head_dim,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _token_attn_1_kernel[grid](
                q,
                k,
                sm_scale,
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seqlen,
                max_kv_cache_len,
                attn_out,
                kv_cache_loc.stride(0),
                kv_cache_loc.stride(1),
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                attn_out.stride(0),
                attn_out.stride(1),
                HEAD_DIM=k_head_dim,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=1,
            )
        return

    @triton.jit
    def _token_attn_softmax_fwd(softmax_logics, kv_cache_start_loc, kv_cache_seqlen, softmax_prob_out,
                                logics_head_dim_stride, logics_batch_stride, prob_head_dim_stride, prob_batch_stride,
                                BLOCK_SIZE: tl.constexpr):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        row = tl.load(softmax_logics + current_head * logics_head_dim_stride +
                      (current_batch_in_all_start_index + col_offsets) * logics_batch_stride,
                      mask=col_offsets < current_batch_seq_len,
                      other=-float('inf')).to(tl.float32)

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        tl.store(softmax_prob_out + current_head * prob_head_dim_stride +
                 (current_batch_in_all_start_index + col_offsets) * prob_batch_stride,
                 softmax_output,
                 mask=col_offsets < current_batch_seq_len)
        return

    @torch.no_grad()
    def token_attn_softmax_fwd(softmax_logics, kv_cache_start_loc, kv_cache_seqlen, softmax_prob_out, max_kv_cache_len):
        BLOCK_SIZE = triton.next_power_of_2(max_kv_cache_len)
        batch, head_num = kv_cache_start_loc.shape[0], softmax_logics.shape[0]

        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        _token_attn_softmax_fwd[(batch, head_num)](
            softmax_logics,
            kv_cache_start_loc,
            kv_cache_seqlen,
            softmax_prob_out,
            softmax_logics.stride(0),
            softmax_logics.stride(1),
            softmax_prob_out.stride(0),
            softmax_prob_out.stride(1),
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return

    @triton.jit
    def _token_attn_2_kernel(Prob, V, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seqlen, max_kv_cache_len,
                             kv_cache_loc_b_stride, kv_cache_loc_s_stride, prob_head_dim_stride, prob_batch_stride,
                             v_batch_stride, v_head_stride, v_head_dim_stride, attn_out_batch_stride,
                             attn_out_head_stride, attn_out_head_dim_stride, HEAD_DIM: tl.constexpr,
                             BLOCK_N: tl.constexpr):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_start_index = max_kv_cache_len - current_batch_seq_len
        current_batch_end_index = current_batch_seq_len
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        v_loc_off = current_batch * kv_cache_loc_b_stride + (current_batch_start_index + offs_n) * kv_cache_loc_s_stride
        p_offs = current_head * prob_head_dim_stride + (current_batch_in_all_start_index + offs_n) * prob_batch_stride
        v_offs = current_head * v_head_stride + offs_d[None, :] * v_head_dim_stride

        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for start_n in range(0, current_batch_seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            p_value = tl.load(Prob + p_offs + start_n * kv_cache_loc_s_stride,
                              mask=(start_n + offs_n) < current_batch_seq_len,
                              other=0.0)
            v_loc = tl.load(kv_cache_loc + v_loc_off + start_n * kv_cache_loc_s_stride,
                            mask=(start_n + offs_n) < current_batch_seq_len,
                            other=0.0)
            v_value = tl.load(V + v_offs + v_loc[:, None] * v_batch_stride,
                              mask=(start_n + offs_n[:, None]) < current_batch_seq_len,
                              other=0.0)
            acc += tl.sum(p_value[:, None] * v_value, 0)

        acc = acc.to(tl.float16)
        off_o = current_batch * attn_out_batch_stride + current_head * attn_out_head_stride + offs_d * attn_out_head_dim_stride
        out_ptrs = attn_out + off_o
        tl.store(out_ptrs, acc)
        return

    @torch.no_grad()
    def token_attn_fwd_2(prob, v, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seqlen, max_kv_cache_len):
        if triton.__version__ >= "2.1.0":
            BLOCK = 128
        else:
            BLOCK = 64
        batch, head = kv_cache_loc.shape[0], v.shape[1]
        grid = (batch, head)
        num_warps = 4
        dim = v.shape[-1]

        _token_attn_2_kernel[grid](
            prob,
            v,
            attn_out,
            kv_cache_loc,
            kv_cache_start_loc,
            kv_cache_seqlen,
            max_kv_cache_len,
            kv_cache_loc.stride(0),
            kv_cache_loc.stride(1),
            prob.stride(0),
            prob.stride(1),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            attn_out.stride(0),
            attn_out.stride(1),
            attn_out.stride(2),
            HEAD_DIM=dim,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    @torch.no_grad()
    def token_attention_fwd(q,
                            k,
                            v,
                            attn_out,
                            kv_cache_loc,
                            kv_cache_start_loc,
                            kv_cache_seq_len,
                            max_len_in_batch,
                            alibi=None):
        head_num = k.shape[1]
        batch_size = kv_cache_seq_len.shape[0]
        calcu_shape1 = (batch_size, head_num, k.shape[2])
        total_token_num = k.shape[0]

        att_m_tensor = torch.empty((head_num, total_token_num), dtype=q.dtype, device="cuda")

        token_attn_fwd_1(q.view(calcu_shape1),
                         k,
                         att_m_tensor,
                         kv_cache_loc,
                         kv_cache_start_loc,
                         kv_cache_seq_len,
                         max_len_in_batch,
                         alibi=alibi)

        prob = torch.empty_like(att_m_tensor)

        token_attn_softmax_fwd(att_m_tensor, kv_cache_start_loc, kv_cache_seq_len, prob, max_len_in_batch)
        att_m_tensor = None
        token_attn_fwd_2(prob, v, attn_out.view(calcu_shape1), kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len,
                         max_len_in_batch)

        prob = None

        return
