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
    this function is modified from
    https://github.com/ModelTC/lightllm/blob/f093edc20683ac3ea1bca3fb5d8320a0dd36cf7b/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L10
    """
    if triton.__version__ < "2.1.0":
        @triton.jit
        def _context_flash_attention_kernel(
            Q,
            K,
            V,
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

                p = p.to(v.dtype)
                acc += tl.dot(p, v)
                # update m_i and l_i
                l_i = l_i_new
                m_i = m_i_new

            off_o = (
                (cur_batch_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
            )
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
            return
    else:
        # this function is modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L11
        @triton.jit
        def _context_flash_attention_kernel_2(
            Q, K, V, sm_scale, Alibi, B_Start_Loc, B_Seqlen,
            Out, 
            kv_group_num, 
            stride_qbs, stride_qh, stride_qd,
            stride_kbs, stride_kh, stride_kd,
            stride_vbs, stride_vh, stride_vd,
            stride_obs, stride_oh, stride_od,
            BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            cur_batch = tl.program_id(0)
            cur_head = tl.program_id(1)
            start_m = tl.program_id(2)
            
            if kv_group_num is not None:
                cur_kv_head = cur_head // kv_group_num

            cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
            cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

            block_start_loc = BLOCK_M * start_m

            # initialize offsets
            offs_n = tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_DMODEL)
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
            if kv_group_num is None or kv_group_num == 1:
                off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
                off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
            else:
                off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
                off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

            q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

            k_ptrs = K + off_k
            v_ptrs = V + off_v

            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

            if Alibi is not None:
                alibi_m = tl.load(Alibi + cur_head)

            block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

            for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                # -- compute qk ----
                k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                qk += tl.dot(q, k)
                qk *= sm_scale

                if Alibi is not None:
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
                acc = acc * acc_scale[:, None]
                # update acc
                v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

                p = p.to(v.dtype)
                acc += tl.dot(p, v)
                # update m_i and l_i
                l_i = l_i_new
                m_i = m_i_new
            # initialize pointers to output
            off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
            return

    @torch.no_grad()
    def bloom_context_attn_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, alibi=None):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk, "context process only supports equal query, key, value length"
        assert Lk == Lv, "context process only supports equal query, key, value length"
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / math.sqrt(Lk)
        batch, head = b_seq_len.shape[0], q.shape[1]

        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

        num_warps = 4 if Lk <= 64 else 8
        
        if triton.__version__ < "2.1.0":
            tmp = torch.empty((batch, head, max_input_len + 256), device=q.device, dtype=torch.float32)
            _context_flash_attention_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                b_start_loc,
                b_seq_len,
                tmp,
                alibi,
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
                # manually setting this blcok num, we can use tuning config to futher speed-up
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _context_flash_attention_kernel_2[grid](
                q, k, v, sm_scale, alibi, b_start_loc, b_seq_len,
                o,
                None,
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
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=1,
            )
            
        return

    @torch.no_grad()
    def llama_context_attn_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
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
        # num_warps = 4
        
        if triton.__version__ < "2.1.0":
            _context_flash_attention_kernel[grid](
                q,
                k,
                v,
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
        else:
            kv_group_num = q.shape[1] // k.shape[1]
            _context_flash_attention_kernel_2[grid](                
                q, 
                k, 
                v, 
                sm_scale, 
                None,
                b_start_loc, 
                b_seq_len,
                o,
                kv_group_num,
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
                BLOCK_M=BLOCK,
                BLOCK_DMODEL=Lk,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=1,)
            
        return