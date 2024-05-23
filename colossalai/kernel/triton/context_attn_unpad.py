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


# Triton 2.1.0
@triton.jit
def _fwd_context_paged_attention_kernel(
    Q,
    K,
    V,
    O,
    KCache,
    VCache,
    BLOCK_TABLES,  # [num_seqs, max_blocks_per_sequence]
    batch_size,
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
    stride_cacheb,
    stride_cacheh,
    stride_cachebs,
    stride_cached,
    stride_bts,
    stride_btb,
    context_lengths,
    sm_scale,
    KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    if cur_seq_idx >= batch_size:
        return
    cur_head_idx = tl.program_id(1)
    block_start_m = tl.program_id(2)  # Br, max_input_len // Block_M
    cur_kv_head_idx = cur_head_idx // KV_GROUPS

    # NOTE It requires BLOCK_M, BLOCK_N, and BLOCK_SIZE to be the same
    tl.static_assert(BLOCK_M == BLOCK_N)
    tl.static_assert(BLOCK_N == BLOCK_SIZE)

    # get the current sequence length from provided context lengths tensor
    cur_seq_len = tl.load(context_lengths + cur_seq_idx)
    # NOTE when talking to fused QKV and a nopadding context attention,
    # we assume that the input Q/K/V is contiguous, and thus here `prev_seq_len_sum`
    # could be considered as the start index of the current sequence.
    # FIXME might want to explore better way to get the summation of prev seq lengths.
    # `tl.sum(tensor[:end])` is invalid as tensor slice is not supported in triton.
    prev_seq_len_sum = 0
    for i in range(0, cur_seq_idx):
        prev_seq_len_sum += tl.load(context_lengths + i)

    offset_q = prev_seq_len_sum * stride_qt + cur_head_idx * stride_qh
    offset_kv = prev_seq_len_sum * stride_kt + cur_kv_head_idx * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_qt, stride_qd),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + offset_kv,
        shape=(HEAD_DIM, cur_seq_len),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + offset_kv,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_ot, stride_od),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # block table for the current sequence
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    # block indexes on block table (i.e. 0, 1, 2, ..., max_blocks_per_seq)
    # Consider `block_start_m` as the logical block idx in the current block table,
    # as we have BLOCK_M the same size as the block size.
    cur_block_table_idx = block_start_m
    cur_block_id = tl.load(block_table_ptr + cur_block_table_idx * stride_btb)
    offset_kvcache = cur_block_id * stride_cacheb + cur_kv_head_idx * stride_cacheh

    offsets_m = block_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(1, 0))

    if cur_head_idx % KV_GROUPS == 0:
        # Copy k to corresponding cache block
        offsets_dmodel = tl.arange(0, HEAD_DIM)
        offsets_kt = block_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = K + offset_kv + offsets_dmodel[None, :] * stride_kd + offsets_kt[:, None] * stride_kt
        k = tl.load(offsets_k, mask=offsets_kt[:, None] < cur_seq_len, other=0.0)
        offsets_kcachebs = tl.arange(0, BLOCK_SIZE)
        offsets_kcache = (
            KCache
            + offset_kvcache
            + offsets_dmodel[None, :] * stride_cached
            + offsets_kcachebs[:, None] * stride_cachebs
        )
        tl.store(offsets_kcache, k, mask=offsets_kcachebs[:, None] < cur_seq_len - block_start_m * BLOCK_SIZE)
        # Copy v to corresponding cache block
        offsets_vd = offsets_dmodel
        offsets_vt = block_start_m * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_v = V + offset_kv + offsets_vt[None, :] * stride_vt + offsets_vd[:, None] * stride_vd
        v = tl.load(offsets_v, mask=offsets_vt[None, :] < cur_seq_len, other=0.0)
        offsets_vcachebs = offsets_kcachebs  # same block size range, just to notify here
        offsets_vcache = (
            VCache
            + offset_kvcache
            + offsets_vcachebs[None, :] * stride_cachebs
            + offsets_dmodel[:, None] * stride_cached
        )
        tl.store(offsets_vcache, v, mask=offsets_vcachebs[None, :] < cur_seq_len - block_start_m * BLOCK_SIZE)

    return


# Triton 2.1.0
# TODO(yuanheng-zhao): This is a temporary dispatch to use the new layout for kcache
# merge `_fwd_context_paged_attention_kernel_v2` with `_fwd_context_paged_attention_kernel` later
# as the kcache layout has been supported in the whole triton flow.
@triton.jit
def _fwd_context_paged_attention_kernel_v2(
    Q,
    K,
    V,
    O,
    KCache,  # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    VCache,  # [num_blocks, num_kv_heads, block_size, head_dim]
    BLOCK_TABLES,  # [num_seqs, max_blocks_per_sequence]
    batch_size,
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
    stride_cacheb,  # v cache stride(0) - num_blocks
    stride_cacheh,  # v cache stride(1) - num_kv_heads
    stride_cachebs,  # v cache stride(2) - block_size
    stride_cached,  # v cache stride(3) - head_dim
    stride_bts,
    stride_btb,
    context_lengths,
    sm_scale,
    KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,  # k stride on the second last dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    if cur_seq_idx >= batch_size:
        return
    cur_head_idx = tl.program_id(1)
    block_start_m = tl.program_id(2)  # Br, max_input_len // Block_M
    cur_kv_head_idx = cur_head_idx // KV_GROUPS

    # NOTE It requires BLOCK_M, BLOCK_N, and BLOCK_SIZE to be the same
    tl.static_assert(BLOCK_M == BLOCK_N)
    tl.static_assert(BLOCK_N == BLOCK_SIZE)

    # get the current sequence length from provided context lengths tensor
    cur_seq_len = tl.load(context_lengths + cur_seq_idx)
    # NOTE when talking to fused QKV and a nopadding context attention,
    # we assume that the input Q/K/V is contiguous, and thus here `prev_seq_len_sum`
    # could be considered as the start index of the current sequence.
    # FIXME might want to explore better way to get the summation of prev seq lengths.
    # `tl.sum(tensor[:end])` is invalid as tensor slice is not supported in triton.
    prev_seq_len_sum = 0
    for i in range(0, cur_seq_idx):
        prev_seq_len_sum += tl.load(context_lengths + i)

    offset_q = prev_seq_len_sum * stride_qt + cur_head_idx * stride_qh
    offset_kv = prev_seq_len_sum * stride_kt + cur_kv_head_idx * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_qt, stride_qd),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + offset_kv,
        shape=(HEAD_DIM, cur_seq_len),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + offset_kv,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_ot, stride_od),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # block table for the current sequence
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    # block indexes on block table (i.e. 0, 1, 2, ..., max_blocks_per_seq)
    # Consider `block_start_m` as the logical block idx in the current block table,
    # as we have BLOCK_M the same size as the block size.
    cur_block_table_idx = block_start_m
    cur_block_id = tl.load(block_table_ptr + cur_block_table_idx * stride_btb)
    offset_kvcache = cur_block_id * stride_cacheb + cur_kv_head_idx * stride_cacheh

    offsets_m = block_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(1, 0))

    if cur_head_idx % KV_GROUPS == 0:
        # Copy k to corresponding cache block
        block_range = tl.arange(0, BLOCK_SIZE)
        X_range = tl.arange(0, KCACHE_X)
        # unroll the loop aggressively
        for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
            offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
            offsets_k = K + offset_kv + offsets_dmodel_x_partition[None, :] * stride_kd + offsets_m[:, None] * stride_kt
            k = tl.load(offsets_k, mask=offsets_m[:, None] < cur_seq_len, other=0.0)
            # HACK: KCache must be contiguous in order to apply the following offsets calculation
            offsets_kcache = (
                KCache
                + offset_kvcache
                + split_x * BLOCK_SIZE * KCACHE_X
                + block_range[:, None] * KCACHE_X
                + X_range[None, :]
            )
            tl.store(offsets_kcache, k, mask=block_range[:, None] < cur_seq_len - block_start_m * BLOCK_SIZE)
        # Copy v to corresponding cache block
        offsets_vd = tl.arange(0, HEAD_DIM)  # offsets_dmodel
        offsets_vt = block_start_m * BLOCK_N + offsets_n
        offsets_v = V + offset_kv + offsets_vt[None, :] * stride_vt + offsets_vd[:, None] * stride_vd
        v = tl.load(offsets_v, mask=offsets_vt[None, :] < cur_seq_len, other=0.0)
        offsets_vcache = (
            VCache + offset_kvcache + block_range[None, :] * stride_cachebs + offsets_vd[:, None] * stride_cached
        )
        tl.store(offsets_vcache, v, mask=block_range[None, :] < cur_seq_len - block_start_m * BLOCK_SIZE)

    return


# Triton 2.1.0
@triton.jit
def _alibi_fwd_context_paged_attention_kernel(
    Q,
    K,
    V,
    O,
    KCache,
    VCache,
    BLOCK_TABLES,  # [num_seqs, max_blocks_per_sequence]
    batch_size,
    alibi_slopes,
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
    stride_cacheb,
    stride_cacheh,
    stride_cachebs,
    stride_cached,
    stride_bts,
    stride_btb,
    context_lengths,
    sm_scale,
    KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    if cur_seq_idx >= batch_size:
        return
    cur_head_idx = tl.program_id(1)
    block_start_m = tl.program_id(2)  # Br, max_input_len // Block_M
    cur_kv_head_idx = cur_head_idx // KV_GROUPS

    global_block_start_offest = block_start_m * BLOCK_M

    # NOTE It requires BLOCK_M, BLOCK_N, and BLOCK_SIZE to be the same
    tl.static_assert(BLOCK_M == BLOCK_N)
    tl.static_assert(BLOCK_N == BLOCK_SIZE)

    # get the current sequence length from provided context lengths tensor
    cur_seq_len = tl.load(context_lengths + cur_seq_idx)
    # NOTE when talking to fused QKV and a nopadding context attention,
    # we assume that the input Q/K/V is contiguous, and thus here `prev_seq_len_sum`
    # could be considered as the start index of the current sequence.
    # FIXME might want to explore better way to get the summation of prev seq lengths.
    # `tl.sum(tensor[:end])` is invalid as tensor slice is not supported in triton.
    prev_seq_len_sum = 0
    for i in range(0, cur_seq_idx):
        prev_seq_len_sum += tl.load(context_lengths + i)

    offset_q = prev_seq_len_sum * stride_qt + cur_head_idx * stride_qh
    offset_kv = prev_seq_len_sum * stride_kt + cur_kv_head_idx * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_qt, stride_qd),
        offsets=(global_block_start_offest, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + offset_kv,
        shape=(HEAD_DIM, cur_seq_len),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + offset_kv,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + offset_q,
        shape=(cur_seq_len, HEAD_DIM),
        strides=(stride_ot, stride_od),
        offsets=(global_block_start_offest, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # block table for the current sequence
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    # block indexes on block table (i.e. 0, 1, 2, ..., max_blocks_per_seq)
    # Consider `block_start_m` as the logical block idx in the current block table,
    # as we have BLOCK_M the same size as the block size.
    cur_block_table_idx = block_start_m
    cur_block_id = tl.load(block_table_ptr + cur_block_table_idx * stride_btb)
    offset_kvcache = cur_block_id * stride_cacheb + cur_kv_head_idx * stride_cacheh

    offsets_m = global_block_start_offest + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # load alibi_slope
    alibi_slope = tl.load(alibi_slopes + cur_head_idx)
    m_alibi_offset = tl.arange(0, BLOCK_M)[:, None] + global_block_start_offest
    n_alibi_offset = tl.arange(0, BLOCK_N)[None, :]

    if global_block_start_offest >= cur_seq_len:
        return

    Q_i = tl.load(Q_block_ptr, boundary_check=(1, 0))

    for block_start_n in range(0, (block_start_m + 1) * BLOCK_M, BLOCK_N):
        block_start_n = tl.multiple_of(block_start_n, BLOCK_N)

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        S_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        S_ij += tl.dot(Q_i, k)
        S_ij *= sm_scale
        S_ij += tl.where(offsets_m[:, None] >= (block_start_n + offsets_n[None, :]), 0, float("-inf"))

        alibi = (n_alibi_offset + block_start_n - m_alibi_offset) * alibi_slope
        alibi = tl.where((alibi <= 0) & (m_alibi_offset < cur_seq_len), alibi, float("-inf"))
        S_ij += alibi

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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(1, 0))

    if cur_head_idx % KV_GROUPS == 0:
        # Copy k to corresponding cache block
        offsets_dmodel = tl.arange(0, HEAD_DIM)
        offsets_kt = global_block_start_offest + tl.arange(0, BLOCK_M)
        offsets_k = K + offset_kv + offsets_dmodel[None, :] * stride_kd + offsets_kt[:, None] * stride_kt
        k = tl.load(offsets_k, mask=offsets_kt[:, None] < cur_seq_len, other=0.0)
        offsets_kcachebs = tl.arange(0, BLOCK_SIZE)
        offsets_kcache = (
            KCache
            + offset_kvcache
            + offsets_dmodel[None, :] * stride_cached
            + offsets_kcachebs[:, None] * stride_cachebs
        )
        tl.store(offsets_kcache, k, mask=offsets_kcachebs[:, None] < cur_seq_len - block_start_m * BLOCK_SIZE)
        # Copy v to corresponding cache block
        offsets_vd = offsets_dmodel
        offsets_vt = block_start_m * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_v = V + offset_kv + offsets_vt[None, :] * stride_vt + offsets_vd[:, None] * stride_vd
        v = tl.load(offsets_v, mask=offsets_vt[None, :] < cur_seq_len, other=0.0)
        offsets_vcachebs = offsets_kcachebs  # same block size range, just to notify here
        offsets_vcache = (
            VCache
            + offset_kvcache
            + offsets_vcachebs[None, :] * stride_cachebs
            + offsets_dmodel[:, None] * stride_cached
        )
        tl.store(offsets_vcache, v, mask=offsets_vcachebs[None, :] < cur_seq_len - block_start_m * BLOCK_SIZE)

    return


def context_attention_unpadded(
    q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_dim]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_sequence],
    block_size: int,
    output: torch.Tensor = None,  # [num_tokens, num_heads, head_dim]
    alibi_slopes: torch.Tensor = None,  # [num_heads]
    max_seq_len: int = None,
    sm_scale: int = None,
    # NOTE(yuanheng-zhao): the following flag is used to determine whether to use the new layout for kcache
    # [num_blocks, num_kv_heads, head_dim // x, block_size, x] - must be contiguous
    use_new_kcache_layout: bool = False,
):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv
    assert Lk in {32, 64, 128, 256}
    assert q.shape[0] == k.shape[0] == v.shape[0]
    k_cache_shape = k_cache.shape
    v_cache_shape = v_cache.shape
    if use_new_kcache_layout:
        assert (
            len(k_cache_shape) == 5
            and k_cache_shape[1] == v_cache_shape[1]
            and k_cache_shape[2] * k_cache_shape[4] == v_cache_shape[3]
        ), f"Invalid KCache shape {k_cache_shape} and VCache shape {v_cache_shape}"
    else:
        assert k_cache_shape == v_cache_shape, f"Invalid KCache shape {k_cache_shape} and VCache shape {v_cache_shape}"
    assert context_lengths.shape[0] == block_tables.shape[0]

    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[-2]
    assert num_kv_heads > 0 and num_heads % num_kv_heads == 0
    num_kv_group = num_heads // num_kv_heads

    num_seqs, max_blocks_per_seq = block_tables.shape
    max_seq_len = context_lengths.max().item() if max_seq_len is None else max_seq_len
    sm_scale = 1.0 / (Lq**0.5) if sm_scale is None else sm_scale
    output = (
        torch.empty((num_tokens, num_heads * head_dim), dtype=q.dtype, device=q.device) if output is None else output
    )

    # NOTE For now, BLOCK_M and BLOCK_N are supposed to be equivalent with
    # the size of physical cache block (i.e. `block_size`)
    assert block_size in {16, 32, 64, 128}
    BLOCK_M = BLOCK_N = block_size

    # NOTE use `triton.next_power_of_2` here to utilize the cache mechanism of triton
    # To optimize, revise batching/scheduling to batch 2^n sequences in a batch (preferred)
    grid = (triton.next_power_of_2(num_seqs), num_heads, triton.cdiv(max_seq_len, BLOCK_M))

    if use_new_kcache_layout:
        # TODO(yuanheng-zhao): Since the alibi kernel is pretty similar to the original one,
        # the code (alibi kernel) will be refactored later to avoid code duplication, when
        # the whole triton flow with new k cache layout has been supported and tested.
        assert (
            alibi_slopes is None
        ), "Alibi Slopes will be supported with new kcache layout later when the whole triton flow is ready"
        x = k_cache_shape[4]  # Intuition: 16 // dtype_size

        _fwd_context_paged_attention_kernel_v2[grid](
            q,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_tables,
            num_seqs,
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
            head_dim,
            1,
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            context_lengths,
            sm_scale,
            KV_GROUPS=num_kv_group,
            BLOCK_SIZE=block_size,
            HEAD_DIM=Lk,
            KCACHE_X=x,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return output

    if alibi_slopes is not None:
        _alibi_fwd_context_paged_attention_kernel[grid](
            q,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_tables,
            num_seqs,
            alibi_slopes,
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
            head_dim,
            1,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            context_lengths,
            sm_scale,
            num_kv_group,
            block_size,
            HEAD_DIM=Lk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        _fwd_context_paged_attention_kernel[grid](
            q,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_tables,
            num_seqs,
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
            head_dim,
            1,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            context_lengths,
            sm_scale,
            num_kv_group,
            block_size,
            HEAD_DIM=Lk,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return output
