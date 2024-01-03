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
    stride_cached,
    stride_cachebs,
    stride_bts,
    stride_btb,
    context_lengths,
    sm_scale,
    KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
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

    q_offset = prev_seq_len_sum * stride_qt + cur_head_idx * stride_qh
    kv_offset = prev_seq_len_sum * stride_kt + cur_kv_head_idx * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(cur_seq_len, BLOCK_DMODEL),
        strides=(stride_qt, stride_qd),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, cur_seq_len),
        strides=(stride_kd, stride_kt),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(cur_seq_len, BLOCK_DMODEL),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + q_offset,
        shape=(cur_seq_len, BLOCK_DMODEL),
        strides=(stride_ot, stride_od),
        offsets=(block_start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    # block table for the current sequence
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    # block indexes on block table (i.e. 0, 1, 2, ..., max_blocks_per_seq)
    # Consider `block_start_m` as the logical block idx in the current block table,
    # as we have BLOCK_M the same size as the block size.
    cur_block_table_idx = block_start_m
    cur_block_id = tl.load(block_table_ptr + cur_block_table_idx * stride_btb)
    kvcache_offset = cur_block_id * stride_cacheb + cur_kv_head_idx * stride_cacheh

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
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O.type.element_ty), boundary_check=(1, 0))

    if cur_head_idx % KV_GROUPS == 0:
        # Copy k to corresponding cache block
        kd_offsets = tl.arange(0, BLOCK_DMODEL)
        kt_offsets = block_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = K + kv_offset + kd_offsets[:, None] * stride_kd + kt_offsets[None, :] * stride_kt
        k = tl.load(k_offsets, mask=kt_offsets[None, :] < cur_seq_len, other=0.0)
        kcached_offsets = tl.arange(0, BLOCK_DMODEL)
        kcachebs_offsets = tl.arange(0, BLOCK_SIZE)
        kcache_offsets = (
            KCache
            + kvcache_offset
            + kcached_offsets[:, None] * stride_cached
            + kcachebs_offsets[None, :] * stride_cachebs
        )
        tl.store(kcache_offsets, k, mask=kcachebs_offsets[None, :] < cur_seq_len - block_start_m * BLOCK_SIZE)
        # Copy v to corresponding cache block
        vd_offsets = kd_offsets
        vt_offsets = block_start_m * BLOCK_N + tl.arange(0, BLOCK_N)
        v_offsets = V + kv_offset + vt_offsets[:, None] * stride_vt + vd_offsets[None, :] * stride_vd
        v = tl.load(v_offsets, mask=vt_offsets[:, None] < cur_seq_len, other=0.0)
        vcached_offsets = kcached_offsets
        vcachebs_offsets = kcachebs_offsets
        vcache_offsets = (
            VCache
            + kvcache_offset
            + vcachebs_offsets[:, None] * stride_cachebs
            + vcached_offsets[None, :] * stride_cached
        )
        tl.store(vcache_offsets, v, mask=vcachebs_offsets[:, None] < cur_seq_len - block_start_m * BLOCK_SIZE)

    return


def context_attention_unpadded(
    q: torch.Tensor,  # [num_tokens, num_heads, head_size]
    k: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    v: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    k_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    v_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_blocks_per_sequence],
    block_size: int,
):
    # q/k in context stage are supposed to be put into k_cache and v_cache.
    # This step can be optimized in future.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv
    assert Lk in {32, 64, 128, 256}
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert k_cache.shape == v_cache.shape
    assert context_lengths.shape[0] == block_tables.shape[0]

    num_tokens, num_heads, _ = q.shape
    num_kv_heads = k.shape[-2]
    assert num_kv_heads > 0 and num_heads % num_kv_heads == 0
    num_kv_group = num_heads // num_kv_heads

    num_seqs, max_blocks_per_seq = block_tables.shape
    max_seq_len = context_lengths.max().item()
    sm_scale = 1.0 / (Lq**0.5)

    output = torch.zeros_like(q)

    # NOTE For now, BLOCK_M and BLOCK_N are supposed to be equivalent with
    # the size of physical cache block (i.e. `block_size`)
    assert block_size in {16, 32, 64, 128}
    BLOCK_M = BLOCK_N = block_size

    grid = (num_seqs, num_heads, triton.cdiv(max_seq_len, BLOCK_M))

    _fwd_context_paged_attention_kernel[grid](
        q,
        k,
        v,
        output,
        k_cache,
        v_cache,
        block_tables,
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
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output
