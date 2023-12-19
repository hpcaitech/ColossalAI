# Applying the FlashAttention V2 as described in:
# "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
# by Tri Dao, 2023
# https://github.com/Dao-AILab/flash-attention
#
# Inspired and modified from Triton Tutorial - Fused Attention
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_context_paged_attention_kernel_v2(
    Q,
    K,
    V,
    O,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    context_lengths,  # [num_seqs]
    sm_scale,
    H: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head_size, or head_dim (synonym)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    block_start_m = tl.program_id(2)  # Br, 这个打不满，max_input_len // Block_M
    # Only consider MHA for llama for now
    # cur_kv_head_idx = cur_head_idx

    # get the current sequence length from provided context lengths tensor
    cur_seq_len = tl.load(context_lengths + cur_seq_idx)
    # NOTE when talking to fused QKV and a nopadding context attention,
    # we assume that the input Q/K/V is contiguous, and thus here `prev_seq_len_sum`
    # could be considered as the start index of the current sequence.
    # FIXME might want to explore better way to get the summation of prev seq lengths.
    # `tl.sum(tensor[:end])` is invalid as tensor slice is not supported.
    prev_seq_len_sum = 0
    for i in range(0, cur_seq_idx):
        prev_seq_len_sum += tl.load(context_lengths + i)

    qkv_offset = prev_seq_len_sum.to(tl.int64) * stride_qt + cur_head_idx.to(tl.int64) * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(MAX_SEQ_LEN, BLOCK_DMODEL),  # 大几率超出当前seq范围 需要用boundary_check
        strides=(stride_qt, stride_qd),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(MAX_SEQ_LEN, BLOCK_DMODEL),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, MAX_SEQ_LEN),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(cur_seq_len, BLOCK_DMODEL),
        strides=(stride_ot, stride_od),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    offsets_m = block_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if block_start_m * BLOCK_M >= cur_seq_len:
        return

    Q_i = tl.load(Q_block_ptr, boundary_check=(1, 0))

    for block_start_n in range(0, (block_start_m + 1) * BLOCK_M, BLOCK_N):
        block_start_n = tl.multiple_of(block_start_n, BLOCK_N)

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        S_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        S_ij += tl.dot(Q_i, k)
        S_ij *= sm_scale
        S_ij += tl.where(offsets_m[:, None] >= (block_start_n + offsets_n[None, :]), 0, float("-inf"))

        m_ij = tl.max(S_ij, 1)  # rowmax(Sij)
        m_ij = tl.maximum(m_i, m_ij)  # m_ij
        S_ij -= m_ij[:, None]
        p_ij_hat = tl.exp(S_ij)
        scale = tl.exp(m_i - m_ij)
        l_ij = scale * l_i + tl.sum(p_ij_hat, 1)
        acc = acc * scale[:, None]

        v = tl.load(V_block_ptr, boundary_check=(1, 0))
        p_ij_hat = p_ij_hat.to(v.type.element_ty)

        acc += tl.dot(p_ij_hat, v)
        l_i = l_ij
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(1, 0))

    return


def context_attention_unpadded(
    q: torch.Tensor,  # [num_tokens, num_heads, head_size]
    k: torch.Tensor,  # [num_tokens, num_heads, head_size]
    v: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # k_cache: torch.Tensor, # [num_blocks, num_heads, head_size, block_size]
    # v_cache: torch.Tensor, # [num_blocks, num_heads, head_size, block_size]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_sequence]
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv
    assert Lk in {32, 64, 128, 256}
    assert k.shape[-2] == v.shape[-2]
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert context_lengths.shape[0] == block_tables.shape[0]

    # For now, we focus on MHA and thus num_heads == num_kv_heads
    num_tokens, num_heads, _ = q.shape
    num_seqs, max_blocks_per_seq = block_tables.shape
    max_seq_len = context_lengths.max().item()
    sm_scale = 1.0 / (Lq**0.5)

    output = torch.zeros_like(q)

    # FIXME We might want to add autotune for BLOCK_M and BLOCK_N
    BLOCK_M = BLOCK_N = 32

    grid = (num_seqs, num_heads, triton.cdiv(max_seq_len, BLOCK_M))

    _fwd_context_paged_attention_kernel_v2[grid](
        q,
        k,
        v,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        context_lengths,  # [num_seqs]
        sm_scale,
        num_heads,
        MAX_SEQ_LEN=max_seq_len,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output
