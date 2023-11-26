# Adapted from AutoGPTQ auto_gptq: https://github.com/PanQiWei/AutoGPTQ

import torch
import triton
import triton.language as tl

from .custom_autotune import autotune, matmul248_kernel_config_pruner


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


# a Triton implementation of the most used activations
# See for instance http://arxiv.org/abs/1606.08415 for an overview


# ReLU
@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    return tl.where(x >= 0, x, 0.0)


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_sq = x * x
    return tl.where(x > 0.0, x_sq, 0.0)


@triton.jit
def star_relu(x):
    """
    Star ReLU activation, as proposed in the "MetaFormer Baselines for Vision"_ paper.

    .. _ "MetaFormer Baselines for Vision": https://arxiv.org/pdf/2210.13452.pdf
    """
    x_sq = x * x
    return 0.8944 * tl.where(x > 0.0, x_sq, 0.0) - 0.4472


# Leaky ReLU
@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    return tl.where(x >= 0.0, x, 0.01 * x)


@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def smelu(x):
    """
    SmeLU_ activation -  Smooth ReLU with beta=2.0

    .. _SmeLU: https://arxiv.org/pdf/2202.06499.pdf
    """
    beta = 2.0

    relu = tl.where(x >= beta, x, 0.0)
    return tl.where(tl.abs(x) <= beta, (x + beta) * (x + beta) / (4.0 * beta), relu)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4
        ),
    ],
    key=["M", "N", "K"],
    nearest_power_of_two=True,
    prune_configs_by={
        "early_config_prune": matmul248_kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)
@triton.jit
def cai_gptq_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    bias_ptr,
    residual_ptr,
    M,
    N,
    K,
    bits,
    maxq,
    gptq_group_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    QKV_FUSED: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    NK = K

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(NK, BLOCK_SIZE_K)
    qkv_offset = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_bk = offs_k + qkv_offset * NK
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)

    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = (
        b_ptr
        + qkv_offset * N * NK // infearure_per_bits
        + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + qkv_offset * NK * N // gptq_group_size + offs_bn[None, :]
    zeros_ptrs = (
        zeros_ptr
        + qkv_offset * NK * N // gptq_group_size // infearure_per_bits
        + (offs_bn[None, :] // infearure_per_bits)
    )

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    g_idx_base = tl.arange(0, BLOCK_SIZE_K)
    g_idx_base = g_idx_base // gptq_group_size
    g_idx = g_idx_base
    # tl.device_print("gidx, ", g_idx)

    scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
    zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
    zeros = (zeros >> zeros_shifter[None, :]) & maxq
    zeros = zeros + 1

    for k in range(0, num_pid_k):
        # g_idx = tl.load(g_ptrs)
        # if (k + 1) * BLOCK_SIZE_K > currend_group_end:
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
        b = (b - zeros).to(tl.float16) * scales  # Scale and shift
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_idx = g_idx_base + ((k + 1) * BLOCK_SIZE_K) // gptq_group_size
        # if (k + 2) * BLOCK_SIZE_K > currend_group_end:

    c_ptrs = c_ptr + qkv_offset * M * N + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    if ADD_BIAS:
        bias_mask = offs_bn < N
        offs_bn += qkv_offset * N
        bias_ptrs = bias_ptr + stride_cn * offs_bn
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        accumulator += bias[None, :]

    if ACT_TYPE == 1:
        accumulator = relu(accumulator)
    elif ACT_TYPE == 2:
        accumulator = gelu(accumulator)
    elif ACT_TYPE == 3:
        accumulator = silu(accumulator)

    if ADD_RESIDUAL:
        residual_ptrs = residual_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        res = tl.load(residual_ptrs, mask=c_mask, other=0.0)
        accumulator += res

    tl.store(c_ptrs, accumulator, mask=c_mask)


# Adapted from AutoGPTQ auto_gptq: https://github.com/PanQiWei/AutoGPTQ
@autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=2, num_warps=4
        ),
    ],
    key=["M", "N", "K"],
    nearest_power_of_two=True,
    prune_configs_by={
        "early_config_prune": matmul248_kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)
@triton.jit
def cai_gptq_idx_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    idx_ptr,
    bias_ptr,
    residual_ptr,
    M,
    N,
    K,
    bits,
    maxq,
    gptq_group_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    QKV_FUSED: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    """
    infearure_per_bits = 32 // bits

    pid = tl.program_id(axis=0)
    NK = K

    # if QKV_FUSED:
    #     NK = K//3
    # else:
    #     NK = K
    # NK = K

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(NK, BLOCK_SIZE_K)
    qkv_offset = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_bk = offs_k + qkv_offset * NK
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)

    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = (
        b_ptr
        + qkv_offset * N * NK // infearure_per_bits
        + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + qkv_offset * NK * N // gptq_group_size + offs_bn[None, :]
    zeros_ptrs = (
        zeros_ptr
        + qkv_offset * NK * N // gptq_group_size // infearure_per_bits
        + (offs_bn[None, :] // infearure_per_bits)
    )

    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    g_ptrs = idx_ptr + offs_k
    g_idx = tl.load(g_ptrs)
    # tl.device_print("gidx, ", g_idx)
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

    scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        zeros = (zeros >> zeros_shifter[None, :]) & maxq
        zeros = zeros + 1

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        # Now we need to unpack b (which is N-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
        b = (b - zeros).to(tl.float16) * scales  # Scale and shift
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + qkv_offset * M * N + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)

    if ADD_BIAS:
        bias_mask = offs_bn < N
        offs_bn += qkv_offset * N
        bias_ptrs = bias_ptr + stride_cn * offs_bn
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        accumulator += bias[None, :]

    if ACT_TYPE == 1:
        accumulator = relu(accumulator)
    elif ACT_TYPE == 2:
        accumulator = gelu(accumulator)
    elif ACT_TYPE == 3:
        accumulator = silu(accumulator)

    if ADD_RESIDUAL:
        residual_ptrs = residual_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        res = tl.load(residual_ptrs, mask=c_mask, other=0.0)
        accumulator += res

    tl.store(c_ptrs, accumulator, mask=c_mask)


def gptq_fused_linear_triton(
    input,
    qweight,
    scales,
    qzeros,
    bias,
    residual,
    bits,
    maxq,
    gptq_group_size,
    qkv_fused,
    add_bias,
    add_residual,
    g_idx=None,
    act_type=0,
):
    # print("gptq fused ", qkv_fused, add_bias, add_residual)
    assert input.is_cuda, "input is not in cuda"
    assert qweight.is_cuda, "qweight is not in cuda"
    assert scales.is_cuda, "scales is not in cuda"
    assert qzeros.is_cuda, "qzeros is not in cuda"

    with torch.cuda.device(input.device):
        if qkv_fused:
            grid = lambda META: (
                triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"])
                * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"])
                * 3,
            )
            output = torch.empty((input.shape[0] * 3, qweight.shape[1]), device=input.device, dtype=torch.float16)
        else:
            grid = lambda META: (
                triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"]),
            )
            output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        # print("dtype, ", qweight.dtype, output.dtype, scales.dtype, qzeros.dtype, bias.dtype, residual.dtype)
        if g_idx is None:
            cai_gptq_matmul_248_kernel[grid](
                input,
                qweight,
                output,
                scales,
                qzeros,
                bias,
                residual,
                input.shape[0],
                qweight.shape[1],
                input.shape[1],
                bits,
                maxq,
                gptq_group_size,
                input.stride(0),
                input.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                output.stride(0),
                output.stride(1),
                scales.stride(0),
                qzeros.stride(0),
                QKV_FUSED=qkv_fused,
                ADD_BIAS=add_bias,
                ADD_RESIDUAL=add_residual,
                ACT_TYPE=act_type,
            )
        else:
            cai_gptq_idx_matmul_248_kernel[grid](
                input,
                qweight,
                output,
                scales,
                qzeros,
                g_idx,
                bias,
                residual,
                input.shape[0],
                qweight.shape[1],
                input.shape[1],
                bits,
                maxq,
                gptq_group_size,
                input.stride(0),
                input.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                output.stride(0),
                output.stride(1),
                scales.stride(0),
                qzeros.stride(0),
                QKV_FUSED=qkv_fused,
                ADD_BIAS=add_bias,
                ADD_RESIDUAL=add_residual,
                ACT_TYPE=act_type,
            )
        if qkv_fused:
            return output.view(3, input.shape[0], qweight.shape[1])
        else:
            return output
