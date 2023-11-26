try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")


if HAS_TRITON:
    """
    this kernel function is modified from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    """

    @triton.jit
    def qkv_gemm_4d_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_ab,
        stride_ah,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cm,
        stride_cn,
        scale,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr = 64,
        BLOCK_SIZE_N: tl.constexpr = 32,
        BLOCK_SIZE_K: tl.constexpr = 32,
        GROUP_SIZE_M: tl.constexpr = 8,
    ):
        r"""A kernel function which is used to do batch-matmul for Q*K^T or score_matrix * V for attention layer,
            where score_matrix is softmax(Q*V^T/sqrt(hidden_size))
        Args:
            a_ptr(torch.Tensor): pointer to input tensor array (bs, M, h, K) or (bs, h, M, K)
            b_ptr(torch.Tensor): pointer to input tensor array (bs, N, h, K) or (bs, h, N, K)
            c_ptr(torch.Tensor): pointer to output tensor array (bs, M, h, N) or (bs, h, M, N)
            stride_ab(tl.constexpr): stride for bs-dimention for tensor array A
            stride_ah(tl.constexpr): stride for h-dimention for tensor array A
            stride_am(tl.constexpr): stride for m-dimention for tensor array A
            stride_ak(tl.constexpr): stride for k-dimention for tensor array A
            stride_bb(tl.constexpr): stride for bs-dimention for tensor array B
            stride_bh(tl.constexpr): stride for h-dimention for tensor array B
            stride_bk(tl.constexpr): stride for k-dimention for tensor array B
            stride_bn(tl.constexpr): stride for n-dimention for tensor array B
            stride_cb(tl.constexpr): stride for bs-dimention for tensor array output
            stride_ch(tl.constexpr): stride for h-dimention for tensor array output
            stride_cm(tl.constexpr): stride for m-dimention for tensor array output
            stride_cn(tl.constexpr): stride for n-dimention for tensor array output
            BLOCK_SIZE_M : tiling size for M-dimension of tensor Array a
            BLOCK_SIZE_N : tiling size for N-dimension of tensor Array b
            BLOCK_SIZE_K : tiling size for K-dimension of a and b
            GROUP_SIZE_M : group size for reducing cache miss, more details:
        """

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        batch = tl.program_id(axis=0)
        head = tl.program_id(axis=1)
        pid = tl.program_id(axis=2)

        # the following is from tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = (
            a_ptr + batch * stride_ab + head * stride_ah + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        )
        b_ptrs = (
            b_ptr + batch * stride_bb + head * stride_bh + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k < K)
            b_mask = (offs_k[:, None] + k < K) & (offs_bn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        accumulator = accumulator.to(c_ptr.dtype.element_ty)
        if scale > 0:
            accumulator = accumulator * scale.to(c_ptr.dtype.element_ty)

        offs_accumu_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_accumu_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + batch * stride_cb
            + head * stride_ch
            + stride_cm * offs_accumu_m[:, None]
            + stride_cn * offs_accumu_n[None, :]
        )
        accumulator_mask = (offs_accumu_m[:, None] < M) & (offs_accumu_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=accumulator_mask)
