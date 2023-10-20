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
    """
    this functions are modified from https://github.com/ModelTC/lightllm
    """

    # Adapted from  https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
    @triton.jit
    def _context_flash_attention_kernel(
        Q,
        K,
        V,
        q_input_scale,
        k_input_scale,
        v_input_scale,
        pv_output_scale,
        sm_scale,
        B_Start_Loc,
        B_Seqlen,
        TMP,
        alibi_ptr,
        Out,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_tmp_b,
        stride_tmp_h,
        stride_tmp_s,
        # suggtest set-up 64, 128, 256, 512
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

        # get batch info
        cur_batch_seq_len = tl.load(B_Seqlen + batch_id)
        cur_batch_start_index = tl.load(B_Start_Loc + batch_id)
        block_start_loc = BLOCK_M * start_m

        load_p_ptrs = (
            Q
            + (cur_batch_start_index + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(load_p_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
        q = q.to(tl.float16) * q_input_scale.to(tl.float16)

        k_ptrs = K + offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        t_ptrs = TMP + batch_id * stride_tmp_b + cur_head * stride_tmp_h + offs_m * stride_tmp_s

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        if alibi_ptr is not None:
            alibi_m = tl.load(alibi_ptr + cur_head)

        block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            k = tl.load(
                k_ptrs + (cur_batch_start_index + start_n) * stride_kbs,
                mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
                other=0.0,
            )
            k = k.to(tl.float16) * k_input_scale.to(tl.float16)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale

            if alibi_ptr is not None:
                alibi_loc = offs_m[:, None] - (start_n + offs_n[None, :])
                qk -= alibi_loc * alibi_m

            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            tl.store(t_ptrs, acc_scale)
            acc_scale = tl.load(t_ptrs)
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(
                v_ptrs + (cur_batch_start_index + start_n) * stride_vbs,
                mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
                other=0.0,
            )

            v = v.to(tl.float16) * v_input_scale.to(tl.float16)
            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        acc = (acc / pv_output_scale.to(tl.float16)).to(tl.int8)
        off_o = (
            (cur_batch_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
        )
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
        return

    @torch.no_grad()
    def smooth_llama_context_attn_fwd(
        q, k, v, o, q_input_scale, k_input_scale, v_input_scale, pv_output_scale, b_start_loc, b_seq_len, max_input_len
    ):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk, "context process only supports equal query, key, value length"
        assert Lk == Lv, "context process only supports equal query, key, value length"
        assert Lk in {16, 32, 64, 128}
        sm_scale = 1.0 / math.sqrt(Lk)
        batch, head = b_seq_len.shape[0], q.shape[1]
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

        tmp = torch.empty((batch, head, max_input_len + 256), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _context_flash_attention_kernel[grid](
            q,
            k,
            v,
            q_input_scale,
            k_input_scale,
            v_input_scale,
            pv_output_scale,
            sm_scale,
            b_start_loc,
            b_seq_len,
            tmp,
            None,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            tmp.stride(0),
            tmp.stride(1),
            tmp.stride(2),
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    # Adapted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
    @triton.jit
    def _token_attn_1_kernel(
        Q,
        K,
        q_input_scale,
        k_input_scale,
        sm_scale,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seqlen,
        max_kv_cache_len,
        attn_out,
        kv_cache_loc_b_stride,
        kv_cache_loc_s_stride,
        q_batch_stride,
        q_head_stride,
        q_head_dim_stride,
        k_batch_stride,
        k_head_stride,
        k_head_dim_stride,
        attn_head_stride,
        attn_batch_stride,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
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
            q = q.to(tl.float16) * q_input_scale.to(tl.float16)
            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(
                kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                mask=offs_n_new < current_batch_end_index,
                other=0,
            )
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            k = k.to(tl.float16) * k_input_scale.to(tl.float16)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    # Adapted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
    @triton.jit
    def _token_attn_1_alibi_kernel(
        Q,
        K,
        q_input_scale,
        k_input_scale,
        sm_scale,
        alibi,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seqlen,
        max_kv_cache_len,
        attn_out,
        kv_cache_loc_b_stride,
        kv_cache_loc_s_stride,
        q_batch_stride,
        q_head_stride,
        q_head_dim_stride,
        k_batch_stride,
        k_head_stride,
        k_head_dim_stride,
        attn_head_stride,
        attn_batch_stride,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
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
            q = q.to(tl.float16) * q_input_scale.to(tl.float16)

            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(
                kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                mask=offs_n_new < current_batch_end_index,
                other=0,
            )
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            k = k.to(tl.float16) * k_input_scale.to(tl.float16)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            att_value -= alibi_m * (current_batch_seq_len - 1 - offs_n)
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    @torch.no_grad()
    def token_attn_fwd_1(
        q,
        k,
        attn_out,
        q_input_scale,
        k_input_scale,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seqlen,
        max_kv_cache_len,
        alibi=None,
    ):
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
                q_input_scale,
                k_input_scale,
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
                q_input_scale,
                k_input_scale,
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

    # Adapted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_softmax_and_reducev.py
    @triton.jit
    def _token_attn_softmax_fwd(
        softmax_logics,
        kv_cache_start_loc,
        kv_cache_seqlen,
        softmax_prob_out,
        logics_head_dim_stride,
        logics_batch_stride,
        prob_head_dim_stride,
        prob_batch_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        row = tl.load(
            softmax_logics
            + current_head * logics_head_dim_stride
            + (current_batch_in_all_start_index + col_offsets) * logics_batch_stride,
            mask=col_offsets < current_batch_seq_len,
            other=-float("inf"),
        ).to(tl.float32)

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        tl.store(
            softmax_prob_out
            + current_head * prob_head_dim_stride
            + (current_batch_in_all_start_index + col_offsets) * prob_batch_stride,
            softmax_output,
            mask=col_offsets < current_batch_seq_len,
        )
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

    # Adapted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/token_attention_nopad_att1.py
    @triton.jit
    def _token_attn_2_kernel(
        Prob,
        V,
        attn_out,
        v_input_scale,
        pv_output_scale,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seqlen,
        max_kv_cache_len,
        kv_cache_loc_b_stride,
        kv_cache_loc_s_stride,
        prob_head_dim_stride,
        prob_batch_stride,
        v_batch_stride,
        v_head_stride,
        v_head_dim_stride,
        attn_out_batch_stride,
        attn_out_head_stride,
        attn_out_head_dim_stride,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        current_batch = tl.program_id(0)
        current_head = tl.program_id(1)

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        current_batch_seq_len = tl.load(kv_cache_seqlen + current_batch)
        current_batch_start_index = max_kv_cache_len - current_batch_seq_len
        current_batch_in_all_start_index = tl.load(kv_cache_start_loc + current_batch)

        v_loc_off = current_batch * kv_cache_loc_b_stride + (current_batch_start_index + offs_n) * kv_cache_loc_s_stride
        p_offs = current_head * prob_head_dim_stride + (current_batch_in_all_start_index + offs_n) * prob_batch_stride
        v_offs = current_head * v_head_stride + offs_d[None, :] * v_head_dim_stride

        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for start_n in range(0, current_batch_seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            p_value = tl.load(
                Prob + p_offs + start_n * kv_cache_loc_s_stride,
                mask=(start_n + offs_n) < current_batch_seq_len,
                other=0.0,
            )
            v_loc = tl.load(
                kv_cache_loc + v_loc_off + start_n * kv_cache_loc_s_stride,
                mask=(start_n + offs_n) < current_batch_seq_len,
                other=0.0,
            )
            v_value = tl.load(
                V + v_offs + v_loc[:, None] * v_batch_stride,
                mask=(start_n + offs_n[:, None]) < current_batch_seq_len,
                other=0.0,
            )
            v_value = v_value.to(tl.float16) * v_input_scale.to(tl.float16)
            acc += tl.sum(p_value[:, None] * v_value, 0)

        acc = (acc / pv_output_scale.to(tl.float16)).to(tl.int8)
        off_o = (
            current_batch * attn_out_batch_stride
            + current_head * attn_out_head_stride
            + offs_d * attn_out_head_dim_stride
        )
        out_ptrs = attn_out + off_o
        tl.store(out_ptrs, acc)
        return

    @torch.no_grad()
    def token_attn_fwd_2(
        prob,
        v,
        attn_out,
        v_input_scale,
        pv_output_scale,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seqlen,
        max_kv_cache_len,
    ):
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
            v_input_scale,
            pv_output_scale,
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
    def smooth_token_attention_fwd(
        q,
        k,
        v,
        attn_out,
        q_input_scale,
        k_input_scale,
        v_input_scale,
        pv_output_scale,
        kv_cache_loc,
        kv_cache_start_loc,
        kv_cache_seq_len,
        max_len_in_batch,
        alibi=None,
    ):
        head_num = k.shape[1]
        batch_size = kv_cache_seq_len.shape[0]
        calcu_shape1 = (batch_size, head_num, k.shape[2])
        total_token_num = k.shape[0]

        att_m_tensor = torch.empty((head_num, total_token_num), dtype=torch.float32, device="cuda")

        token_attn_fwd_1(
            q.view(calcu_shape1),
            k,
            att_m_tensor,
            q_input_scale,
            k_input_scale,
            kv_cache_loc,
            kv_cache_start_loc,
            kv_cache_seq_len,
            max_len_in_batch,
            alibi=alibi,
        )

        prob = torch.empty_like(att_m_tensor)

        token_attn_softmax_fwd(att_m_tensor, kv_cache_start_loc, kv_cache_seq_len, prob, max_len_in_batch)
        att_m_tensor = None
        token_attn_fwd_2(
            prob,
            v,
            attn_out.view(calcu_shape1),
            v_input_scale,
            pv_output_scale,
            kv_cache_loc,
            kv_cache_start_loc,
            kv_cache_seq_len,
            max_len_in_batch,
        )

        prob = None

        return
