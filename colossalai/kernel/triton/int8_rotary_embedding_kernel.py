# Adapted from ModelTC https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    q,
    input_scale,
    output_scale,
    Cos,
    Sin,
    q_bs_stride,
    q_h_stride,
    q_d_stride,
    cos_bs_stride,
    cos_d_stride,
    total_len,
    HEAD_NUM: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    current_head_index = tl.program_id(0)
    current_seq_index = tl.program_id(1)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    current_head_range = current_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    current_seq_range = current_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    off_q0 = (
        current_seq_range[:, None, None] * q_bs_stride
        + current_head_range[None, :, None] * q_h_stride
        + dim_range0[None, None, :] * q_d_stride
    )
    off_q1 = (
        current_seq_range[:, None, None] * q_bs_stride
        + current_head_range[None, :, None] * q_h_stride
        + dim_range1[None, None, :] * q_d_stride
    )

    off_dimcos_sin = current_seq_range[:, None, None] * cos_bs_stride + dim_range0[None, None, :] * cos_d_stride

    q0 = tl.load(
        q + off_q0,
        mask=(current_seq_range[:, None, None] < total_len) & (current_head_range[None, :, None] < HEAD_NUM),
        other=0.0,
    )
    q1 = tl.load(
        q + off_q1,
        mask=(current_seq_range[:, None, None] < total_len) & (current_head_range[None, :, None] < HEAD_NUM),
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=current_seq_range[:, None, None] < total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=current_seq_range[:, None, None] < total_len, other=0.0)

    q0 = q0.to(tl.float32) * input_scale
    q1 = q1.to(tl.float32) * input_scale

    out0 = (q0 * cos - q1 * sin) / output_scale
    out1 = (q0 * sin + q1 * cos) / output_scale

    out0 = out0.to(tl.int8)
    out1 = out1.to(tl.int8)

    tl.store(
        q + off_q0,
        out0,
        mask=(current_seq_range[:, None, None] < total_len) & (current_head_range[None, :, None] < HEAD_NUM),
    )
    tl.store(
        q + off_q1,
        out1,
        mask=(current_seq_range[:, None, None] < total_len) & (current_head_range[None, :, None] < HEAD_NUM),
    )

    return


@torch.no_grad()
def int8_rotary_embedding_fwd(q, cos, sin, input_scale, output_scale):
    total_len = q.shape[0]
    head_num = q.shape[1]
    head_dim = q.shape[2]
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    BLOCK_HEAD = 4
    BLOCK_SEQ = 32
    grid = (triton.cdiv(head_num, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    _rotary_kernel[grid](
        q,
        input_scale,
        output_scale,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        total_len,
        HEAD_NUM=head_num,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
