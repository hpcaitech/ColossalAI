import torch
import triton
import triton.language as tl


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

    rotary_data = q
    HEAD_NUM = Q_HEAD_NUM
    head_stride = q_head_stride
    token_stride = q_token_stride

    if block_token_index * BLOCK_TOKENS >= q_total_tokens:
        block_token_index = block_token_index - tl.cdiv(q_total_tokens, BLOCK_TOKENS)
        rotary_data = k
        HEAD_NUM = K_HEAD_NUM
        head_stride = k_head_stride
        token_stride = k_token_stride

    tokens_range = block_token_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_range = block_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    off_data0 = (
        tokens_range[:, None, None] * token_stride
        + head_range[None, :, None] * head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_data1 = (
        tokens_range[:, None, None] * token_stride
        + head_range[None, :, None] * head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )

    loaded_data0 = tl.load(
        rotary_data + off_data0,
        mask=((head_range[None, :, None] < HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )
    loaded_data1 = tl.load(
        rotary_data + off_data1,
        mask=((head_range[None, :, None] < HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
        other=0.0,
    )

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)
    loaded_sin = tl.load(sin + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens), other=0.0)

    out0 = loaded_data0 * loaded_cos[:, None, :] - loaded_data1 * loaded_sin[:, None, :]
    out1 = loaded_data0 * loaded_sin[:, None, :] + loaded_data1 * loaded_cos[:, None, :]

    # concat
    tl.store(
        rotary_data + off_data0,
        out0,
        mask=((head_range[None, :, None] < HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )
    tl.store(
        rotary_data + off_data1,
        out1,
        mask=((head_range[None, :, None] < HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens)),
    )


@torch.no_grad()
def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
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
    BLOCK_TOKENS = 8
    grid = (triton.cdiv(q_head_num, BLOCK_HEAD), 2 * triton.cdiv(q_total_tokens, BLOCK_TOKENS))

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
    cos_stride = cos.stride(1)

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
        num_stages=1,
    )

    return
