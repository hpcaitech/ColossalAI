# Adapted from ModelTC https://github.com/ModelTC/lightllm


import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

try:
    from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2 as lightllm_llama_token_att_fwd2
    from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd as lightllm_llama_token_att_fwd
    from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd as lightllm_llama_token_softmax_fwd
    from lightllm.models.bloom.triton_kernel.token_attention_nopad_att1 import token_att_fwd as lightllm_bloom_token_att_fwd

    HAS_TRITON_TOKEN_ATTENTION = True
except ImportError:
    print("unable to import lightllm kernels")
    HAS_TRITON_TOKEN_ATTENTION = False

if HAS_TRITON:

    @torch.no_grad()
    def token_attention_fwd(
        q, k, v, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_len_in_batch, alibi=None
    ):
        head_num = k.shape[1]
        batch_size = kv_cache_seq_len.shape[0]
        calcu_shape1 = (batch_size, head_num, k.shape[2])
        total_token_num = k.shape[0]

        att_m_tensor = torch.empty((head_num, total_token_num), dtype=q.dtype, device="cuda")

        if alibi is None:
            lightllm_llama_token_att_fwd(
                q.view(calcu_shape1),
                k,
                att_m_tensor,
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seq_len,
                max_len_in_batch,
            )
        else:
            lightllm_bloom_token_att_fwd(
                q.view(calcu_shape1),
                k,
                att_m_tensor,
                alibi,
                kv_cache_loc,
                kv_cache_start_loc,
                kv_cache_seq_len,
                max_len_in_batch,
            )

        prob = torch.empty_like(att_m_tensor)

        lightllm_llama_token_softmax_fwd(att_m_tensor, kv_cache_start_loc, kv_cache_seq_len, prob, max_len_in_batch)
        att_m_tensor = None
        lightllm_llama_token_att_fwd2(
            prob, v, attn_out.view(calcu_shape1), kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_len_in_batch
        )
        prob = None
        return


class Llama2TokenAttentionForwards:
    @staticmethod
    @triton.jit

    # this function is adapted from https://github.com/ModelTC/lightllm/blob/5c559dd7981ed67679a08a1e09a88fb4c1550b3a/lightllm/models/llama2/triton_kernel/token_attention_nopad_softmax.py#L8
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

    # this function is adapted from https://github.com/ModelTC/lightllm/blob/5c559dd7981ed67679a08a1e09a88fb4c1550b3a/lightllm/models/llama2/triton_kernel/token_attention_nopad_softmax.py#L36
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

        lightllm_llama_token_att_fwd(
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
            lightllm_llama_token_softmax_fwd(
                att_m_tensor, kv_cache_start_loc, kv_cache_seq_len, prob, max_len_in_batch
            )
            att_m_tensor = None

            lightllm_llama_token_att_fwd2(
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
