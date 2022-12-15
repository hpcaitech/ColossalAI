"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf; Triton https://github.com/openai/triton)
"""

import math
import os
import subprocess

import torch


def triton_cuda_check():
    cuda_home = os.getenv("CUDA_HOME", default="/usr/local/cuda")
    cuda_version = subprocess.check_output([os.path.join(cuda_home, "bin/nvcc"), "--version"]).decode().strip()
    cuda_version = cuda_version.split('release ')[1]
    cuda_version = cuda_version.split(',')[0]
    cuda_version = cuda_version.split('.')
    if len(cuda_version) == 2 and \
        (int(cuda_version[0]) == 11 and int(cuda_version[1]) >= 4) or \
        int(cuda_version[0]) > 11:
        return True
    return False


try:
    import triton
    import triton.language as tl
    if triton_cuda_check():
        HAS_TRITON = True
    else:
        print("triton requires cuda >= 11.4")
        HAS_TRITON = False
except ImportError:
    print('please install triton from https://github.com/openai/triton')
    HAS_TRITON = False
try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_func,
        flash_attn_unpadded_kvpacked_func,
        flash_attn_unpadded_qkvpacked_func,
    )
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print('please install flash_attn from https://github.com/HazyResearch/flash-attention')

try:
    from xformers.ops.fmha import memory_efficient_attention
    HAS_MEM_EFF_ATTN = True
except ImportError:
    HAS_MEM_EFF_ATTN = False
    print('please install xformers from https://github.com/facebookresearch/xformers')

if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        TMP,
        L,
        M,    # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        Z,
        H,
        N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
        # Initialize pointers to Q, K, V
        q_ptrs = Q + off_q
        k_ptrs = K + off_k
        v_ptrs = V + off_v
        # initialize pointer to m and l
        t_ptrs = TMP + off_hz * N_CTX + offs_m
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load q: it will stay in SRAM throughout
        q = tl.load(q_ptrs)
        # loop over k, v and update accumulator
        for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs + start_n * stride_kn)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, trans_b=True)
            qk *= sm_scale
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
            # -- compute m_ij, p, l_ij
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
            acc_scale = tl.load(t_ptrs)    # BUG: have to store and immediately load
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs + start_n * stride_vk)
            p = p.to(tl.float16)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # rematerialize offsets to save registers
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # write back l and m
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i)
        tl.store(m_ptrs, m_i)
        # initialize pointers to output
        offs_n = tl.arange(0, BLOCK_DMODEL)
        off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc)

    @triton.jit
    def _bwd_preprocess(
        Out,
        DO,
        L,
        NewDO,
        Delta,
        BLOCK_M: tl.constexpr,
        D_HEAD: tl.constexpr,
    ):
        off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, D_HEAD)
        # load
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
        denom = tl.load(L + off_m).to(tl.float32)
        # compute
        do = do / denom[:, None]
        delta = tl.sum(o * do, axis=1)
        # write-back
        tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
        tl.store(Delta + off_m, delta)

    @triton.jit
    def _bwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        M,
        D,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        Z,
        H,
        N_CTX,
        num_block,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H
        # offset pointers for batch/head
        Q += off_z * stride_qz + off_h * stride_qh
        K += off_z * stride_qz + off_h * stride_qh
        V += off_z * stride_qz + off_h * stride_qh
        DO += off_z * stride_qz + off_h * stride_qh
        DQ += off_z * stride_qz + off_h * stride_qh
        DK += off_z * stride_qz + off_h * stride_qh
        DV += off_z * stride_qz + off_h * stride_qh
        for start_n in range(0, num_block):
            lo = start_n * BLOCK_M
            # initialize row/col offsets
            offs_qm = lo + tl.arange(0, BLOCK_M)
            offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_m = tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_DMODEL)
            # initialize pointers to value-like data
            q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            # pointer to row-wise quantities in value-like data
            D_ptrs = D + off_hz * N_CTX
            m_ptrs = M + off_hz * N_CTX
            # initialize dv amd dk
            dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
            # k and v stay in SRAM throughout
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            # loop over rows
            for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
                offs_m_curr = start_m + offs_m
                # load q, k, v, do on-chip
                q = tl.load(q_ptrs)
                # recompute p = softmax(qk, dim=-1).T
                # NOTE: `do` is pre-divided by `l`; no normalization here
                qk = tl.dot(q, k, trans_b=True)
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
                m = tl.load(m_ptrs + offs_m_curr)
                p = tl.exp(qk * sm_scale - m[:, None])
                # compute dv
                do = tl.load(do_ptrs)
                dv += tl.dot(p.to(tl.float16), do, trans_a=True)
                # compute dp = dot(v, do)
                Di = tl.load(D_ptrs + offs_m_curr)
                dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
                dp += tl.dot(do, v, trans_b=True)
                # compute ds = p * (dp - delta[:, None])
                ds = p * dp * sm_scale
                # compute dk = dot(ds.T, q)
                dk += tl.dot(ds.to(tl.float16), q, trans_a=True)
                # # compute dq
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds.to(tl.float16), k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                # # increment pointers
                dq_ptrs += BLOCK_M * stride_qm
                q_ptrs += BLOCK_M * stride_qm
                do_ptrs += BLOCK_M * stride_qm
            # write-back
            dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)

    class _TritonFlashAttention(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k, v, sm_scale):
            BLOCK = 128
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            assert Lq == Lk and Lk == Lv
            assert Lk in {16, 32, 64, 128}
            o = torch.empty_like(q)
            grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
            tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
            L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
            m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
            num_warps = 4 if Lk <= 64 else 8

            _fwd_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                tmp,
                L,
                m,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                BLOCK_M=BLOCK,
                BLOCK_N=BLOCK,
                BLOCK_DMODEL=Lk,
                num_warps=num_warps,
                num_stages=1,
            )
            ctx.save_for_backward(q, k, v, o, L, m)
            ctx.BLOCK = BLOCK
            ctx.grid = grid
            ctx.sm_scale = sm_scale
            ctx.BLOCK_DMODEL = Lk
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, l, m = ctx.saved_tensors
            do = do.contiguous()
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            do_scaled = torch.empty_like(do)
            delta = torch.empty_like(l)
            _bwd_preprocess[(ctx.grid[0] * ctx.grid[1],)](
                o,
                do,
                l,
                do_scaled,
                delta,
                BLOCK_M=ctx.BLOCK,
                D_HEAD=ctx.BLOCK_DMODEL,
            )

            # NOTE: kernel currently buggy for other values of `num_warps`
            num_warps = 8
            _bwd_kernel[(ctx.grid[1],)](
                q,
                k,
                v,
                ctx.sm_scale,
                o,
                do_scaled,
                dq,
                dk,
                dv,
                l,
                m,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                ctx.grid[0],
                BLOCK_M=ctx.BLOCK,
                BLOCK_N=ctx.BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                num_warps=num_warps,
                num_stages=1,
            )
            return dq, dk, dv, None

    def triton_flash_attention(q, k, v, sm_scale):
        """
        Arguments:
            q: (batch, nheads, seq, headdim)
            k: (batch, nheads, seq, headdim)
            v: (batch, nheads, seq, headdim)
            sm_scale: float. The scaling of QK^T before applying softmax.
        Return:
            out: (batch, nheads, seq, headdim)
        """
        if HAS_TRITON:
            return _TritonFlashAttention.apply(q, k, v, sm_scale)
        else:
            raise RuntimeError("Triton kernel requires CUDA 11.4+!")


if HAS_FLASH_ATTN:

    from einops import rearrange

    class MaskedFlashAttention(torch.nn.Module):

        def __init__(self, num_attention_heads: int, attention_head_size: int, attention_dropout: float) -> None:
            super().__init__()
            self.num_attention_heads = num_attention_heads
            self.attention_head_size = attention_head_size
            self.attention_func = FlashAttention(softmax_scale=math.sqrt(attention_head_size),
                                                 attention_dropout=attention_dropout)

        def forward(self, query_key_value: torch.Tensor, attention_mask: torch.Tensor, causal=False):
            if attention_mask.dtype is not torch.bool:
                attention_mask = attention_mask.bool()
            qkv = rearrange(query_key_value, 'b s (three h d) -> b s three h d', three=3, h=self.num_attention_heads)
            context, _ = self.attention_func(qkv, key_padding_mask=attention_mask, causal=causal)
            context = rearrange(context, 'b s h d -> b s (h d)')
            return context

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
        max_s = seq_len
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
        out = flash_attn_unpadded_qkvpacked_func(qkv,
                                                 cu_seqlens,
                                                 max_s,
                                                 dropout_p,
                                                 softmax_scale=sm_scale,
                                                 causal=causal)
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
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_seqlen, step=q_seqlen, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * kv_seqlen,
                                    step=kv_seqlen,
                                    dtype=torch.int32,
                                    device=kv.device)
        out = flash_attn_unpadded_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, q_seqlen, kv_seqlen, dropout_p,
                                                sm_scale, causal)
        return out

    def flash_attention_q_k_v(q, k, v, sm_scale, batch_size, q_seqlen, kv_seqlen, dropout_p=0., causal=False):
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
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_seqlen, step=q_seqlen, dtype=torch.int32, device=q.device)
        cu_seqlens_kv = torch.arange(0, (batch_size + 1) * kv_seqlen,
                                     step=kv_seqlen,
                                     dtype=torch.int32,
                                     device=k.device)
        return flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_seqlen, kv_seqlen, dropout_p, sm_scale,
                                        causal)


if HAS_MEM_EFF_ATTN:

    from einops import rearrange
    from xformers.ops.fmha import LowerTriangularMask

    class MemoryEfficientAttention(torch.nn.Module):

        def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float = 0.0):
            super().__init__()
            attention_head_size = hidden_size // num_attention_heads
            self.scale = 1 / attention_head_size**0.5
            self.dropout = attention_dropout

        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
            context = memory_efficient_attention(query, key, value, attention_mask, self.dropout, self.scale)
            context = rearrange(context, 'b s h d -> b s (h d)')
            return context
