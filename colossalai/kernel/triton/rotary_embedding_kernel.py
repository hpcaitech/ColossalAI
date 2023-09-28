# Adapted from ModelTC https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    q,
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

    current_head_range = current_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    current_seq_range = current_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, HEAD_DIM // 2)
    dim_range1 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

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

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

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
def rotary_embedding_fwd(q, cos, sin):
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


class Llama2Forwards:
    @staticmethod
    @triton.jit
    def _rotary_kernel(
        Q,
        Cos,
        Sin,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_cosbs,
        stride_cosd,
        stride_sinbs,
        stride_sind,
        max_total_len,
        H,  # N_CTX
        BLOCK_HEAD: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        cur_head_index = tl.program_id(0)
        cur_seq_index = tl.program_id(1)

        cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
        cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

        dim_range0 = tl.arange(0, BLOCK_DMODEL // 2) * 2
        dim_range1 = dim_range0 + 1
        off_q0 = (
            cur_seq_range[:, None, None] * stride_qbs
            + cur_head_range[None, :, None] * stride_qh
            + dim_range0[None, None, :] * stride_qd
        )
        off_q1 = (
            cur_seq_range[:, None, None] * stride_qbs
            + cur_head_range[None, :, None] * stride_qh
            + dim_range1[None, None, :] * stride_qd
        )

        cos_range = tl.arange(0, BLOCK_DMODEL // 2)
        off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + cos_range[None, None, :] * stride_cosd

        q0 = tl.load(
            Q + off_q0,
            mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H),
            other=0.0,
        )
        q1 = tl.load(
            Q + off_q1,
            mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H),
            other=0.0,
        )

        cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
        sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

        out0 = q0 * cos - q1 * sin
        out1 = q0 * sin + q1 * cos

        tl.store(
            Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H)
        )
        tl.store(
            Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < H)
        )

        return

    @staticmethod
    @torch.no_grad()
    def rotary_emb_fwd(q, cos, sin):
        total_len = q.shape[0]
        head_num = q.shape[1]
        head_dim = q.shape[2] // 2
        assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
        BLOCK_HEAD = 4
        BLOCK_SEQ = 32
        grid = (triton.cdiv(head_num, BLOCK_HEAD), triton.cdiv(total_len, BLOCK_SEQ))
        if head_dim >= 128:
            num_warps = 8
        else:
            num_warps = 4

        Llama2Forwards._rotary_kernel[grid](
            q,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            total_len,
            head_num,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_SEQ=BLOCK_SEQ,
            BLOCK_DMODEL=head_dim,
            num_warps=num_warps,
            num_stages=1,
        )
        return
