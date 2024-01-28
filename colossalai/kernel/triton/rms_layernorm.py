import torch

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
    def layer_norm_fw(X, Y, W, stride, N, eps, affine: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # fmt: on
        """
        Fused layernorm kernel over a 3d tensor.
        The layer norm is applied over the last dimension.

        Compute
            y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma
        """

        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N

        # Move to this row
        x_ptrs = X + row * stride + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute mean and variance
        mean = tl.sum(x, axis=0) / N
        x_zm = tl.where(mask, x - mean, 0.0)

        x_var = tl.sum(x_zm * x_zm, axis=0) / N
        rstd = 1.0 / tl.sqrt(x_var + eps)

        # Normalize, optionally affine
        y = x_zm * rstd

        mask = cols < N
        if affine:
            w = tl.load(W + cols, mask=mask, other=1.0)
            y = y * w

        y_ptrs = Y + row * stride + cols
        tl.store(y_ptrs, y, mask=mask)

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

    @torch.no_grad()
    def rms_layernorm(x, weight, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor, (total token, hidden_size)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # # Less than 64KB per feature: enqueue fused kernel
        # MAX_FUSED_SIZE = 65536 // x.element_size()

        # BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        # if N > MAX_FUSED_SIZE:
        #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # # heuristics for number of warps
        # num_warps = min(max(triton.next_power_of_2(N) // 256, 8), 32)

        # # enqueue kernel
        # _rmsnorm_kernel[(M,)](x_arg, y, weight, x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        # return y

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                print(
                    "Non-contiguous input tensor found. Making it contiguous,"
                    + " but could have perf or trainer implications"
                )

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        layer_norm_fw[(M,)](
            x_arg, y, weight,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            affine=weight is not None
        )
