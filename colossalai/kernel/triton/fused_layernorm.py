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
    def _layer_norm_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y.to(tl.float16), mask=mask)

    @torch.no_grad()
    def layer_norm(x, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M,)](
            x_arg, y, weight, bias, x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        return y
