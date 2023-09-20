# Adapted from ModelTC https://github.com/ModelTC/lightllm


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
    def _token_attn_1_kernel(
        Q,
        K,
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
            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(
                kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                mask=offs_n_new < current_batch_end_index,
                other=0,
            )
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    @triton.jit
    def _token_attn_1_alibi_kernel(
        Q,
        K,
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
            offs_n_new = current_batch_start_index + offs_n
            k_loc = tl.load(
                kv_cache_loc + kv_cache_loc_b_stride * current_batch + kv_cache_loc_s_stride * offs_n_new,
                mask=offs_n_new < current_batch_end_index,
                other=0,
            )
            off_k = k_loc[:, None] * k_batch_stride + current_head * k_head_stride + offs_d[None, :] * k_head_dim_stride
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < current_batch_end_index, other=0.0)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            att_value -= alibi_m * (current_batch_seq_len - 1 - offs_n)
            off_o = current_head * attn_head_stride + (current_batch_in_all_start_index + offs_n) * attn_batch_stride
            tl.store(attn_out + off_o, att_value, mask=offs_n_new < current_batch_end_index)
        return

    @torch.no_grad()
    def token_attn_fwd_1(
        q, k, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seqlen, max_kv_cache_len, alibi=None
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

    @triton.jit
    def _token_attn_2_kernel(
        Prob,
        V,
        attn_out,
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
            acc += tl.sum(p_value[:, None] * v_value, 0)

        acc = acc.to(tl.float16)
        off_o = (
            current_batch * attn_out_batch_stride
            + current_head * attn_out_head_stride
            + offs_d * attn_out_head_dim_stride
        )
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
    def token_attention_fwd(
        q, k, v, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_len_in_batch, alibi=None
    ):
        head_num = k.shape[1]
        batch_size = kv_cache_seq_len.shape[0]
        calcu_shape1 = (batch_size, head_num, k.shape[2])
        total_token_num = k.shape[0]

        att_m_tensor = torch.empty((head_num, total_token_num), dtype=q.dtype, device="cuda")

        token_attn_fwd_1(
            q.view(calcu_shape1),
            k,
            att_m_tensor,
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
            prob, v, attn_out.view(calcu_shape1), kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_len_in_batch
        )

        prob = None

        return


class Llama2TokenAttentionForwards:
    @staticmethod
    @triton.jit
    def _fwd_kernel(
        Logics,
        V,
        Out,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        max_input_len,
        stride_logic_h,
        stride_logic_bs,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_b_loc_b,
        stride_b_loc_s,
        other_kv_index,  # avoid nan information
        kv_group_num,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)

        cur_kv_head = cur_head // kv_group_num

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        off_v = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

        v_ptrs = V + off_v

        e_max = float("-inf")
        e_sum = 0.0
        acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, cur_batch_seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            v_index = tl.load(
                B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s,
                mask=(start_n + offs_n) < cur_batch_seq_len,
                other=other_kv_index,
            )

            qk = tl.load(
                Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                mask=start_n + offs_n < cur_batch_seq_len,
                other=float("-inf"),
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            old_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            e_sum = e_sum * old_scale + tl.sum(p, 0)
            v = tl.load(v_ptrs + v_index[:, None] * stride_vbs)
            acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
            e_max = n_e_max

        acc = acc / e_sum
        off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc)
        return

    @staticmethod
    @torch.no_grad()
    def token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
        BLOCK = 64
        batch, head = b_seq_len.shape[0], logics.shape[0]
        grid = (batch, head)
        kv_group_num = logics.shape[0] // v.shape[1]

        num_warps = 1
        Llama2TokenAttentionForwards._fwd_kernel[grid](
            logics,
            v,
            o,
            b_loc,
            b_start_loc,
            b_seq_len,
            max_input_len,
            logics.stride(0),
            logics.stride(1),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            b_loc.stride(0),
            b_loc.stride(1),
            other_kv_index,
            kv_group_num,
            BLOCK_DMODEL=v.shape[-1],
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=3,
        )
        return

    @staticmethod
    @triton.jit
    def _fwd_kernel_token_softmax(
        Logics,
        B_Start_Loc,
        B_Seqlen,
        Prob_Out,
        stride_logic_h,
        stride_logic_bs,
        stride_prob_h,
        stride_prob_bs,
        BLOCK_SIZE: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        row = tl.load(
            Logics + cur_head * stride_logic_h + (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs,
            mask=col_offsets < cur_batch_seq_len,
            other=-float("inf"),
        ).to(tl.float32)

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        tl.store(
            Prob_Out + cur_head * stride_prob_h + (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs,
            softmax_output,
            mask=col_offsets < cur_batch_seq_len,
        )
        return

    @staticmethod
    @torch.no_grad()
    def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
        BLOCK_SIZE = triton.next_power_of_2(max_input_len)
        batch, head_num = B_Start_Loc.shape[0], Logics.shape[0]

        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        Llama2TokenAttentionForwards._fwd_kernel_token_softmax[(batch, head_num)](
            Logics,
            B_Start_Loc,
            B_Seqlen,
            Prob_Out,
            Logics.stride(0),
            Logics.stride(1),
            Prob_Out.stride(0),
            Prob_Out.stride(1),
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return

    @staticmethod
    @triton.jit
    def _fwd_kernel_token_att1(
        Q,
        K,
        sm_scale,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        max_input_len,
        Att_Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_kbs,
        stride_kh,
        stride_kd,
        att_stride_h,
        att_stride_bs,
        kv_group_num,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_n = tl.program_id(2)

        cur_kv_head = cur_head // kv_group_num

        offs_d = tl.arange(0, BLOCK_DMODEL)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        cur_batch_start_index = max_input_len - cur_batch_seq_len
        cur_batch_end_index = max_input_len

        off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        block_stard_index = start_n * BLOCK_N
        block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

        for start_mark in range(0, block_mask, 1):
            q = tl.load(Q + off_q + start_mark)
            offs_n_new = cur_batch_start_index + offs_n
            k_loc = tl.load(
                B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new,
                mask=offs_n_new < cur_batch_end_index,
                other=0,
            )
            off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
            k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
            att_value = tl.sum(q[None, :] * k, 1)
            att_value *= sm_scale
            off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
            tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
        return

    @staticmethod
    @torch.no_grad()
    def token_att_fwd(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
        BLOCK = 32
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        assert Lk in {16, 32, 64, 128}
        sm_scale = 1.0 / (Lk**0.5)

        batch, head_num = B_Loc.shape[0], q.shape[1]

        grid = (batch, head_num, triton.cdiv(max_input_len, BLOCK))
        kv_group_num = q.shape[1] // k.shape[1]

        num_warps = 4 if Lk <= 64 else 8
        num_warps = 2

        Llama2TokenAttentionForwards._fwd_kernel_token_att1[grid](
            q,
            k,
            sm_scale,
            B_Loc,
            B_Start_Loc,
            B_Seqlen,
            max_input_len,
            att_out,
            B_Loc.stride(0),
            B_Loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            att_out.stride(0),
            att_out.stride(1),
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    @staticmethod
    @triton.jit
    def _fwd_kernel_token_att2(
        Prob,
        V,
        Out,
        B_Loc,
        B_Start_Loc,
        B_Seqlen,
        max_input_len,  # B_Start_Loc cumsum of input lens if continuous
        stride_b_loc_b,
        stride_b_loc_s,
        stride_ph,
        stride_pbs,
        stride_vbs,
        stride_vh,
        stride_vd,
        stride_obs,
        stride_oh,
        stride_od,
        kv_group_num,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)

        cur_kv_head = cur_head // kv_group_num

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_start_index = max_input_len - cur_batch_seq_len
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        v_loc_off = cur_batch * stride_b_loc_b + (cur_batch_start_index + offs_n) * stride_b_loc_s
        p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
        v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

        acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
        for start_n in range(0, cur_batch_seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            p_value = tl.load(
                Prob + p_offs + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0
            )
            v_loc = tl.load(
                B_Loc + v_loc_off + start_n * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0
            )
            v_value = tl.load(
                V + v_offs + v_loc[:, None] * stride_vbs,
                mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
                other=0.0,
            )
            acc += tl.sum(p_value[:, None] * v_value, 0)

        acc = acc.to(tl.float16)
        off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc)
        return

    @staticmethod
    @torch.no_grad()
    def token_att_fwd2(prob, v, out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
        if triton.__version__ >= "2.1.0":
            BLOCK = 128
        else:
            BLOCK = 64
        batch, head = B_Loc.shape[0], prob.shape[0]
        grid = (batch, head)
        num_warps = 4
        dim = v.shape[-1]

        kv_group_num = prob.shape[0] // v.shape[1]

        Llama2TokenAttentionForwards._fwd_kernel_token_att2[grid](
            prob,
            v,
            out,
            B_Loc,
            B_Start_Loc,
            B_Seqlen,
            max_input_len,
            B_Loc.stride(0),
            B_Loc.stride(1),
            prob.stride(0),
            prob.stride(1),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=dim,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    # this is the interface of llama2 attn forward
    @staticmethod
    @torch.no_grad()
    def token_attn(
        q, k, v, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_len_in_batch, other_kv_index
    ):
        total_token_num = k.shape[0]
        batch_size, head_num, head_dim = q.shape
        calcu_shape1 = (batch_size, head_num, head_dim)
        att_m_tensor = torch.empty((head_num, total_token_num), dtype=q.dtype, device="cuda")

        Llama2TokenAttentionForwards.token_att_fwd(
            q,
            k,
            att_m_tensor,
            kv_cache_loc,
            kv_cache_start_loc,
            kv_cache_seq_len,
            max_len_in_batch,
        )

        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            Llama2TokenAttentionForwards.token_softmax_fwd(
                att_m_tensor, kv_cache_start_loc, kv_cache_seq_len, prob, max_len_in_batch
            )
            att_m_tensor = None

            Llama2TokenAttentionForwards.token_att_fwd2(
                prob,
                v,
                attn_out.view(calcu_shape1),
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seq_len,
                max_len_in_batch,
            )

            prob = None
            return

        elif triton.__version__ >= "2.1.0":
            Llama2TokenAttentionForwards.token_softmax_reducev_fwd(
                att_m_tensor,
                v,
                attn_out.view(calcu_shape1),
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seq_len,
                max_len_in_batch,
                other_kv_index,
            )
        else:
            raise Exception("not support triton version")
