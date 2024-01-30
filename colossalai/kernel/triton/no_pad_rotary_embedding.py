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
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]

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

    past_kv_seq_len = tl.load(context_lengths + tokens_range) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + tokens_range * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride)
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


@torch.no_grad()
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
        k: key tensor, [total_tokens, head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, num_kv_heads, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0)
    BLOCK_HEAD = 4
    BLOCK_TOKENS = 4
    grid = lambda META: (triton.cdiv(q_head_num, META["BLOCK_HEAD"]), triton.cdiv(q_total_tokens, META["BLOCK_TOKENS"]))

    if head_dim >= 256:
        num_warps = 32
    elif head_dim >= 128:
        num_warps = 16
    else:
        num_warps = 4

    q_token_stride = q.stride(0)
    q_head_stride = q.stride(1)
    head_dim_stride = q.stride(2)

    k_token_stride = k.stride(0)
    k_head_stride = k.stride(1)

    k_head_num = q.shape[1]

    cos_token_stride = cos.stride(0)
    cos_stride = cos.stride(1)
    if k_cache == None:
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
            K_HEAD_NUM=k_head_num,
            HEAD_DIM=head_dim,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_TOKENS=BLOCK_TOKENS,
            num_warps=num_warps,
        )
    else:
        fused_rotary_embedding_kernel[grid](
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
            K_HEAD_NUM=k_head_num,
            HEAD_DIM=head_dim,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_TOKENS=BLOCK_TOKENS,
            num_warps=num_warps,
        )
    return
