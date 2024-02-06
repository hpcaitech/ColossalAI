import torch
import triton
import triton.language as tl


@triton.jit
def fused_rotary_emb(
    q,
    k,
    cos_cache,
    sin_cache,
    cumsum_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_dim_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
):
    block_head_index = tl.program_id(0)
    block_group_index = tl.program_id(1)
    group_token_index = tl.program_id(2)
    idx = block_group_index * BLOCK_SIZE + group_token_index

    # original seq_idx and pos
    cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, N_ELEMENTS))
    ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, 0))
    cos = tl.load(
        cos_cache + ori_seq_idx * cos_token_stride + tl.arange(0, HEAD_DIM // 2) * cos_dim_stride
    )  # [1,HEAD_DIM//2]
    sin = tl.load(sin_cache + ori_seq_idx * cos_token_stride + tl.arange(0, HEAD_DIM // 2) * cos_dim_stride)

    cur_head_range = block_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_q0 = (
        idx * q_token_stride
        + cur_head_range[None, :, None] * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        idx * q_token_stride
        + cur_head_range[None, :, None] * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )

    off_k0 = (
        idx * k_token_stride
        + cur_head_range[None, :, None] * k_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_k1 = (
        idx * q_token_stride
        + cur_head_range[None, :, None] * k_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )

    q_0 = tl.load(
        q + off_q0,
        mask=((cur_head_range[None, :, None] < Q_HEAD_NUM) & (idx < q_total_tokens)),
        other=0.0,
    )

    q_1 = tl.load(
        q + off_q1,
        mask=((cur_head_range[None, :, None] < Q_HEAD_NUM) & (idx < q_total_tokens)),
        other=0.0,
    )

    k_0 = tl.load(
        k + off_k0,
        mask=((cur_head_range[None, :, None] < K_HEAD_NUM) & (idx < q_total_tokens)),
        other=0.0,
    )

    k_1 = tl.load(
        k + off_k1,
        mask=((cur_head_range[None, :, None] < K_HEAD_NUM) & (idx < q_total_tokens)),
        other=0.0,
    )

    out_q0 = q_0 * cos - q_1 * sin
    out_q1 = k_0 * sin + k_1 * cos

    out_k0 = q_0 * cos - q_1 * sin
    out_k1 = k_0 * sin + k_1 * cos
    # concat
    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_range[None, :, None] < Q_HEAD_NUM) & (idx < q_total_tokens)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_range[None, :, None] < Q_HEAD_NUM) & (idx < q_total_tokens)),
    )

    tl.store(
        k + off_k0,
        out_k0,
        mask=((cur_head_range[None, :, None] < K_HEAD_NUM) & (idx < q_total_tokens)),
    )
    tl.store(
        k + off_k1,
        out_k1,
        mask=((cur_head_range[None, :, None] < K_HEAD_NUM) & (idx < q_total_tokens)),
    )


def fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    lengths,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        lengths [num_seqs]
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    assert q.size(0) == k.size(0)
    BLOCK_HEAD = 4
    BLOCK_SIZE = 8
    cumsum_lens = torch.cumsum(lengths, dim=0)

    grid = (triton.cdiv(q_head_num, BLOCK_HEAD), triton.cdiv(q_total_tokens, BLOCK_SIZE), BLOCK_SIZE)

    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    q_token_stride = q.stride(0)
    q_head_stride = q.stride(1)
    head_dim_stride = q.stride(2)

    k_token_stride = k.stride(0)
    k_head_stride = k.stride(1)

    k_head_num = q.shape[1]

    cos_token_stride = cos.stride(0)
    cos_dim_stride = cos.stride(1)

    fused_rotary_emb[grid](
        q,
        k,
        cos,
        sin,
        cumsum_lens,
        q_token_stride,
        q_head_stride,
        k_token_stride,
        k_head_stride,
        head_dim_stride,
        cos_token_stride,
        cos_dim_stride,
        q_total_tokens,
        Q_HEAD_NUM=q_head_num,
        K_HEAD_NUM=k_head_num,
        HEAD_DIM=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SIZE=BLOCK_SIZE,
        N_ELEMENTS=triton.next_power_of_2(q_total_tokens),
        num_warps=num_warps,
    )
