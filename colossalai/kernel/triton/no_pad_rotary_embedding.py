import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

"""
# Base autotune if needed
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=4),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":8,},num_warps=8),
        triton.Config({'BLOCK_HEAD':8,"BLOCK_TOKENS":8,},num_warps=8),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=16),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=32),
        triton.Config({'BLOCK_HEAD':16,"BLOCK_TOKENS":16,},num_warps=4),
        triton.Config({'BLOCK_HEAD':8,"BLOCK_TOKENS":16,},num_warps=8),
    ],
    key=['HEAD_DIM','q_total_tokens','Q_HEAD_NUM']
)
"""


@triton.jit
def rotary_embedding_kernel(
    q,
    k,
    cos,
    sin,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,  # token range length
):
    cur_head_idx = tl.program_id(0)
    cur_token_block_idx = tl.program_id(1)

    tokens_range = cur_token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )
    loaded_q0 = tl.load(
        q + off_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k0 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range0[None, None, :] * head_dim_stride
        )
        off_k1 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range1[None, None, :] * head_dim_stride
        )
        loaded_k0 = tl.load(
            k + off_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens),
            other=0.0,
        )
        loaded_k1 = tl.load(
            k + off_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens),
            other=0.0,
        )
        out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
        out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
        tl.store(
            k + off_k0,
            out_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens),
        )
        tl.store(
            k + off_k1,
            out_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens),
        )


@triton.jit
def fused_rotary_embedding_kernel(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    block_head_index = tl.program_id(0)
    block_token_index = tl.program_id(1)

    tokens_range = block_token_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_range = block_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + head_range[None, :, None] * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        tokens_range[:, None, None] * q_token_stride
        + head_range[None, :, None] * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )
    off_k0 = (
        tokens_range[:, None, None] * k_token_stride
        + head_range[None, :, None] * k_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_k1 = (
        tokens_range[:, None, None] * k_token_stride
        + head_range[None, :, None] * k_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )

    loaded_q0 = tl.load(
        q + off_q0,
        mask=((head_range[None, :, None] < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((head_range[None, :, None] < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )

    loaded_k0 = tl.load(
        k + off_k0,
        mask=((head_range[None, :, None] < K_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )

    loaded_k1 = tl.load(
        k + off_k1,
        mask=((head_range[None, :, None] < K_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)

    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + tokens_range, mask=(tokens_range < q_total_tokens)) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + tokens_range * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(tokens_range < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids[:, None, None, None] * cacheb_stride
        + head_range[None, :, None, None] * cacheh_stride
        + offsets_in_last_block[:, None, None, None]
        + dim_range0[None, None, None, :] * cached_stride
    )
    kv_range1 = (
        block_ids[:, None, None, None] * cacheb_stride
        + head_range[None, :, None, None] * cacheh_stride
        + offsets_in_last_block[:, None, None, None]
        + dim_range1[None, None, None, :] * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0[:, :, None, :],
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1[:, :, None, :],
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0,
        mask=((head_range[None, :, None] < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((head_range[None, :, None] < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        k + off_k0,
        out_k0,
        mask=((head_range[None, :, None] < K_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        k + off_k1,
        out_k1,
        mask=((head_range[None, :, None] < K_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )


@triton.jit
def fused_rotary_embedding_kernel_v2(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_head_index = tl.program_id(0)
    if block_head_index >= Q_HEAD_NUM:
        return
    block_token_index = tl.program_id(1)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q0 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range0 * head_dim_stride
    off_q1 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range1 * head_dim_stride
    off_k0 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range0 * head_dim_stride
    off_k1 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(
        q + off_q0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
    )

    loaded_k0 = tl.load(
        k + off_k0,
    )

    loaded_k1 = tl.load(
        k + off_k1,
    )

    off_cos_sin = block_token_index * cos_token_stride + dim_range0 * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(block_token_index < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(block_token_index < q_total_tokens), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + block_token_index * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(block_token_index < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range0 * cached_stride
    )
    kv_range1 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range1 * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0,
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1,
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0,
    )
    tl.store(
        q + off_q1,
        out_q1,
    )


@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q,
    k,
    v,
    cos,
    sin,
    k_cache,
    v_cache,
    BLOCK_TABLES,
    context_lengths,
    x,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    kcb_stride,
    kch_stride,
    kcsplit_x_stride,
    kcs_stride,
    kcd_stride,
    vcb_stride,
    vch_stride,
    vcs_stride,
    vcd_stride,
    bts_stride,
    btb_stride,
    block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_head_idx = tl.program_id(0)
    cur_token_idx = tl.program_id(1)

    dim_range = tl.arange(0, HEAD_DIM)
    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * head_dim_stride
    off_q1 = off_q + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(q + off_q0)
    loaded_q1 = tl.load(q + off_q1)
    off_cos_sin = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin)
    loaded_sin = tl.load(sin + off_cos_sin)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0)
    tl.store(q + off_q1, out_q1)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_kv = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_kv + dim_range0 * head_dim_stride
        off_k1 = off_kv + dim_range1 * head_dim_stride
        loaded_k0 = tl.load(k + off_k0)
        loaded_k1 = tl.load(k + off_k1)

        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

        # NOTE The precondition here is that it's only for unpadded inputs during decoding stage,
        # and so that we could directly use the token index as the sequence index
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - 1

        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (
            offsets_cache_base
            + offsets_in_last_block * kcs_stride
            + (dim_range0 // x) * kcsplit_x_stride
            + (dim_range0 % x) * kcd_stride
        )
        k_range1 = (
            offsets_cache_base
            + offsets_in_last_block * kcs_stride
            + (dim_range1 // x) * kcsplit_x_stride
            + (dim_range1 % x) * kcd_stride
        )
        tl.store(k_cache + k_range0, out_k0)
        tl.store(k_cache + k_range1, out_k1)

        off_v = off_kv + dim_range * head_dim_stride
        loaded_v = tl.load(v + off_v)
        v_range = (
            block_ids * vcb_stride
            + cur_k_head_idx * vch_stride
            + offsets_in_last_block * vcs_stride
            + dim_range * vcd_stride
        )
        tl.store(v_cache + v_range, loaded_v)


def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, kv_head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, num_kv_heads, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0)
    BLOCK_TOKENS = 4

    if head_dim >= 512:
        num_warps = 16
    elif head_dim >= 256:
        num_warps = 8
    else:
        num_warps = 4

    k_head_num = k.size(1)
    q_token_stride, q_head_stride, head_dim_stride = q.stride()
    k_token_stride, k_head_stride, _ = k.stride()
    cos_token_stride, cos_stride = cos.stride()

    assert q_head_num % k_head_num == 0
    kv_group_num = q_head_num // k_head_num

    if k_cache == None:
        grid = lambda META: (
            q_head_num,
            triton.cdiv(q_total_tokens, META["BLOCK_TOKENS"]),
        )
        rotary_embedding_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            KV_GROUP_NUM=kv_group_num,
            HEAD_DIM=head_dim,
            BLOCK_TOKENS=BLOCK_TOKENS,
            num_warps=num_warps,
        )
    else:
        warnings.warn("Fused rotary embedding Triton kernel will be deprecated as the new kcache layout is supported")
        grid = (triton.next_power_of_2(q_head_num), q_total_tokens)
        fused_rotary_embedding_kernel_v2[grid](
            q,
            k,
            cos,
            sin,
            k_cache,
            block_tables,
            kv_lengths,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_cache.size(-2),
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
        )
    return


def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, kv_head_num, head_dim]
        v: value tensor, [total tokens, kv_head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, kv_head_num, block_size, head_dim]
        v_cache (torch.Tensor):  Blocked value cache. [num_blocks, kv_head_num, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0) == v.size(0)

    if head_dim >= 512:
        num_warps = 16
    elif head_dim >= 256:
        num_warps = 8
    else:
        num_warps = 4
    k_head_num = k.size(1)
    kv_group_num = q_head_num // k_head_num

    # For KCache and VCache with the same layout
    x = head_dim
    kcsplit_x_stride, kcs_stride, kcd_stride = 0, k_cache.stride(2), k_cache.stride(3)
    # For KCache layout [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    if use_new_kcache_layout:
        assert (
            k_cache.dim() == 5
            and k_cache.shape[1] == v_cache.shape[1]
            and k_cache.shape[2] * k_cache.shape[4] == v_cache.shape[3]
        ), f"Invalid KCache shape {k_cache.shape} and VCache shape {v_cache.shape}"
        x = k_cache.size(-1)
        kcsplit_x_stride, kcs_stride, kcd_stride = k_cache.stride()[-3:]

    grid = (q_head_num, q_total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q,
        k,
        v,
        cos,
        sin,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        x,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        kcsplit_x_stride,
        kcs_stride,
        kcd_stride,
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        k_cache.size(-2),
        KV_GROUP_NUM=kv_group_num,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )
    return
