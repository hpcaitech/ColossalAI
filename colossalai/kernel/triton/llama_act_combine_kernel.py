from functools import reduce
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

if HAS_TRITON:
    PRECISION_MAP = {
        "fp32": (0, torch.float32),
        "fp16": (1, torch.float16),
        "bf16": (2, torch.bfloat16),
    }

    @triton.jit
    def _llama_act_combine_forward(
        X_GATE1,
        X_GATE2,
        X_UP,
        Y,
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        X_GATE1 += row * stride
        X_GATE2 += row * stride
        X_UP += row * stride
        Y += row * stride

        # do activation and combine, and store in y
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x_gate1 = tl.load(X_GATE1 + cols, mask=mask, other=0.0)
            x_gate2 = tl.load(X_GATE2 + cols, mask=mask, other=0.0)
            x_up = tl.load(X_UP + cols, mask=mask, other=0.0)
            x_gate2_sigmoid = tl.sigmoid(x_gate2.to(tl.float32)).to(x_gate2.dtype)
            y = x_gate1 * x_gate2 * x_gate2_sigmoid * x_up
            # Write output
            tl.store(Y + cols, y, mask=mask)

    @triton.jit
    def _llama_act_combine_backward(
        X_GATE1,
        X_GATE2,
        X_UP,
        X_GATE1_GRAD,
        X_GATE2_GRAD,
        X_UP_GRAD,
        Y_GRAD,
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        X_GATE1 += row * stride
        X_GATE2 += row * stride
        X_UP += row * stride
        X_GATE1_GRAD += row * stride
        X_GATE2_GRAD += row * stride
        X_UP_GRAD += row * stride
        Y_GRAD += row * stride

        # do activation and combine, and store in y
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x_gate1 = tl.load(X_GATE1 + cols, mask=mask, other=0.0)
            x_gate2 = tl.load(X_GATE2 + cols, mask=mask, other=0.0)
            x_up = tl.load(X_UP + cols, mask=mask, other=0.0)
            y_grad = tl.load(Y_GRAD + cols, mask=mask, other=0.0)

            # forward: y = x_gate1 * x_gate2 * tl.sigmoid(x_gate2) * x_up
            x_gate2_sigmoid = tl.sigmoid(x_gate2.to(tl.float32)).to(x_gate2.dtype)
            x_gate2_act = y_grad * x_gate2 * x_gate2_sigmoid
            x_up_grad = x_gate2_act * x_gate1
            x_gate1_grad = x_gate2_act * x_up
            # grad(x*sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * [1 − sigmoid(x)]
            #                    = sigmoid(x) * {1 + x * [(1 − sigmoid(x)]}
            x_gate2_grad = (y_grad * x_gate1 * x_up) * x_gate2_sigmoid * (1 + x_gate2 * (1 - x_gate2_sigmoid))

            # Write output
            tl.store(X_GATE1_GRAD + cols, x_gate1_grad, mask=mask)
            tl.store(X_GATE2_GRAD + cols, x_gate2_grad, mask=mask)
            tl.store(X_UP_GRAD + cols, x_up_grad, mask=mask)

    class LlamaActCombine(torch.autograd.Function):
        """
        act(x_gate) * x_up

        Args:
            x_gate (torch.Tensor): (b, l, 2d) x_gate
            x_up (torch.Tensor): (b, l, d) x_up
            activation (str): only support swiglu
            precision (str): fp32, fp16, bf16
        """

        @staticmethod
        @custom_fwd
        def forward(ctx: Any, x_gate: torch.Tensor, x_up: torch.Tensor, activation: str = "swiglu") -> torch.Tensor:
            """
            act(x_gate) * x_up

            Args:
                x_gate (torch.Tensor): (b, l, 2d) x gate
                x_up (torch.Tensor): (b, l, d) x up
                activation (str): only support swiglu
            """
            assert activation == "swiglu", "Only swiglu is supported"

            # split x gate
            assert x_gate.shape[-1] % 2 == 0, "axis size must be divisible by 2"
            x_gate1, x_gate2 = torch.split(x_gate, x_gate.shape[-1] // 2, -1)
            x_gate1 = x_gate1.contiguous()
            x_gate2 = x_gate2.contiguous()
            if not x_up.is_contiguous():
                x_up = x_up.contiguous()
            # assert shape
            assert x_gate1.shape == x_gate2.shape == x_up.shape

            # add ctx for backward
            if x_gate.requires_grad:
                ctx.save_for_backward(x_gate1, x_gate2, x_up)

            # allocate output
            y = torch.empty_like(x_up)
            M, N = reduce(lambda x, y: x * y, x_up.shape[:-1]), x_up.shape[-1]

            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x_gate.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # restore setting
            ctx.M, ctx.N, ctx.BLOCK_SIZE, ctx.num_warps = M, N, BLOCK_SIZE, num_warps
            # enqueue kernel
            _llama_act_combine_forward[(M,)](
                x_gate1, x_gate2, x_up, y, x_up.stride(-2), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
            return y

        @staticmethod
        @custom_bwd
        def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, Tensor, None, None]:
            # restore from ctx
            (x_gate1, x_gate2, x_up) = ctx.saved_tensors
            M, N, BLOCK_SIZE, num_warps = ctx.M, ctx.N, ctx.BLOCK_SIZE, ctx.num_warps

            # init grad
            y_grad = grad_outputs[0]
            x_gate1_grad, x_gate2_grad, x_up_grad = (
                torch.empty_like(x_gate1),
                torch.empty_like(x_gate2),
                torch.empty_like(x_up),
            )

            # enqueue kernel
            _llama_act_combine_backward[(M,)](
                x_gate1,
                x_gate2,
                x_up,
                x_gate1_grad,
                x_gate2_grad,
                x_up_grad,
                y_grad,
                x_up.stride(-2),
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            x_gate_grad = torch.cat([x_gate1_grad, x_gate2_grad], dim=-1)
            return x_gate_grad, x_up_grad, None, None
