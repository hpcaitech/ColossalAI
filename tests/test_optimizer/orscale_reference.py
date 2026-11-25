"""The official OrScale, as the reference of the alignment tests of ``colossalai.nn.optimizer.OrScale``.

An excerpt of https://github.com/NUS-HPC-AI-Lab/OrScale at commit a45fc2ba19d072ca73dbbce88822631d5c1dd1c7
(``orscale/optim/newton_schulz.py`` and ``orscale/optim/orscale_optimizer.py``), cut down to the two variants of the
paper: the ablation variants, the diagnostics, ``torch.compile`` and the batched orthogonalization are removed and the
branches on the variant simplified accordingly; the arithmetic is the official statements. When it was added, the
excerpt gave the same bits as the full official code run eagerly (``ORSCALE_DISABLE_TORCH_COMPILE=1``) on CPU and
CUDA, for float32, float64 and bfloat16 parameters. ``polar_express_float64`` is not part of the official code: it is
its iteration without the bfloat16 cast, the ground truth for the orthogonalization alone.

MIT License

Copyright (c) 2026 NUS-HPC-AI-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

MOONLIGHT_RMS_CONSTANT = 0.2

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(M: Tensor, num_iters: int = 5) -> Tensor:
    transposed = M.size(-2) > M.size(-1)
    X = M.bfloat16()
    if transposed:
        X = X.mT

    # Scale so spectral norm <= 1 (required for convergence)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    X = X.contiguous()

    # Allocate work buffers
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    for a, b, c in POLAR_EXPRESS_COEFFS[:num_iters]:
        # A = X @ X^T (small square matrix)
        torch.mm(X, X.mT, out=A)
        # B = b*A + c*(A@A)
        torch.addmm(A, A, A, beta=b, alpha=c, out=B)
        # C = a*X + B @ X
        torch.addmm(X, B, X, beta=a, out=C)
        X, C = C, X

    if transposed:
        X = X.mT
    return X


def polar_express_float64(M: Tensor, num_iters: int = 5) -> Tensor:
    transposed = M.size(-2) > M.size(-1)
    X = M.double()
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS[:num_iters]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class OrScaleOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        variant: str = "orscale_lm",
        r_min: float = 0.5,
        r_max: float = 1.5,
        eps: float = 1e-6,
        ns_iters: int = 5,
        c_denom: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if c_denom is not None and c_denom <= 0.0:
            raise ValueError(f"Invalid c_denom: {c_denom} (must be positive or None)")
        if variant not in ("orscale", "orscale_lm"):
            raise ValueError(f"this excerpt has the variants 'orscale' and 'orscale_lm', got {variant!r}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            variant=variant,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            ns_iters=ns_iters,
            c_denom=c_denom,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            variant = group["variant"]
            r_min = group["r_min"]
            r_max = group["r_max"]
            eps = group["eps"]
            ns_iters = group["ns_iters"]
            c_denom_user = group.get("c_denom", None)

            use_shape_norm = variant == "orscale_lm"
            use_grad_matrix_denom = variant == "orscale"
            effective_alpha = 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim < 2:
                    continue

                orig_shape = p.shape
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

                buf = state["momentum_buffer"]

                # Flatten >=3D Conv weights to 2D [C_out, C_in*kH*kW]
                g = grad.float().reshape(orig_shape[0], -1)
                buf_view = buf.view(orig_shape[0], -1)
                p_view = p.view(orig_shape[0], -1)
                p_float = p_view.float()
                m, n = p_view.shape

                # --- Momentum ---
                # Accumulating momentum: M_t = mu * M_{t-1} + G_t
                buf_view.mul_(mu).add_(g)
                # Nesterov lookahead: M_hat = mu * M_t + G_t
                m_hat = buf_view.mul(mu).add(g)

                # --- Orthogonalization ---
                Q = orthogonalize(m_hat, num_iters=ns_iters)

                # --- Shape normalization ---
                shape_scale = MOONLIGHT_RMS_CONSTANT * math.sqrt(max(m, n)) if use_shape_norm else 1.0

                # --- Trust ratio ---
                w_norm = p_view.norm().float()  # ||W||_F  (float32)

                if use_grad_matrix_denom:
                    w_stat = w_norm
                    # p_float is already float32; promote Q for an explicit
                    # float32 sum.  shape_scale and wd are Python floats.
                    grad_matrix = wd * p_float + shape_scale * Q.float()
                    m_stat = grad_matrix.norm()  # ||λW + s·Q||_F
                else:
                    # Coupled denominator c_denom_ℓ * ||λW + s·Q||_F.
                    grad_matrix = wd * p_float + shape_scale * Q.float()
                    grad_matrix_norm = grad_matrix.norm()  # ||λW + s·Q||_F
                    if "c_denom" not in state:
                        if c_denom_user is not None:
                            state["c_denom"] = float(c_denom_user)
                        else:
                            w_init = w_norm.item()
                            gm_init = grad_matrix_norm.item()
                            if w_init < eps or gm_init < eps:
                                # Fallback: pathological tiny init.  Use the
                                # Moonlight shape constant so the optimizer
                                # degenerates to scaled Moonlight rather than
                                # NaNing or producing a runaway ratio.
                                state["c_denom"] = MOONLIGHT_RMS_CONSTANT * math.sqrt(max(m, n))
                            else:
                                state["c_denom"] = w_init / gm_init
                    c_denom_layer = state["c_denom"]
                    w_stat = w_norm
                    m_stat = c_denom_layer * grad_matrix_norm  # c_denom_ℓ · ||λW + s·Q||_F

                ratio_raw = (w_stat / (m_stat + eps)) ** effective_alpha
                ratio_raw_val = ratio_raw.item() if isinstance(ratio_raw, Tensor) else float(ratio_raw)
                ratio_clipped = max(r_min, min(r_max, ratio_raw_val))

                # --- Update ---
                q_update = Q.to(p.dtype)
                if wd != 0.0:
                    p_view.add_(p_view, alpha=-lr * ratio_clipped * wd)
                p_view.add_(q_update, alpha=-lr * ratio_clipped * shape_scale)

        return loss
