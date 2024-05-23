try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

if HAS_TRITON:
    # CREDITS: These functions are adapted from the Triton tutorial
    # https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

    @triton.jit
    def _rmsnorm_kernel(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # This triton kernel implements Root Mean Square Layer Norm (RMSNorm).

        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = x * rstd
            y = x_hat * w
            # Write output
            tl.store(Y + cols, y.to(tl.float16), mask=mask)

    @triton.jit
    def _rmsnorm_with_residual_kernel(
        X,  # pointer to the input
        Y,  # pointer to the output
        R,  # pointer to the residual
        W,  # pointer to the weights
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # This triton kernel implements Root Mean Square Layer Norm (RMSNorm).

        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        R += row * stride
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x, 0.0)
            r = tl.load(R + cols, mask=cols < N, other=0.0).to(tl.float32)
            r = tl.where(cols < N, r, 0.0)
            x = x + r
            _var += x * x
            mask = cols < N
            tl.store(X + cols, x.to(tl.float16), mask=mask)
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = x * rstd
            y = x_hat * w
            # Write output
            tl.store(Y + cols, y.to(tl.float16), mask=mask)

    def rms_layernorm(x, weight, eps, norm_output=None, residual=None):
        # allocate output
        y = (
            x * 0 if norm_output is None else norm_output
        )  # to make the operation non-functional, store y as the intermediate activation
        M, N = x.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()

        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > MAX_FUSED_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # heuristics for number of warps
        num_warps = min(max(triton.next_power_of_2(N) // 256, 8), 32)

        # enqueue kernel
        if residual is None:
            _rmsnorm_kernel[(M,)](x, y, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        else:
            _rmsnorm_with_residual_kernel[(M,)](
                x, y, residual, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
        return y, x
