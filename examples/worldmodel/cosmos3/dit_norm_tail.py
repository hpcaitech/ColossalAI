"""Fuse the elementwise tail of ``Cosmos3NemotronRMSNorm``: bit-exact, no reduction involved.

The norm's forward is ``h = x.float(); v = h.pow(2).mean(-1); r = rsqrt(v + eps); u = h * r; z = w * u;
y = bf16(z)``. The reduction in the middle cannot be reproduced bitwise (``probe_nemotron_norm.py``: 24 candidate
orders, best still differs by one fp32 ulp on 35% of rows), so the head stays with torch. **The tail is pure
elementwise**, and a kernel that applies the same three steps in fp32 with the same rounding is bit-identical:

    ``y = CAST_RINT(w * (h * r))``   vs   ``(w * (h * r)).to(bf16)``

That collapses three full passes (measured in the step profile at 1.10 + 1.12 + 1.10 ms per call) into one that reads
h once and writes y once. The backward is written out in torch ops in autograd's own order -- the version that
``probe_dit_norm.py`` already measured at **0 mismatches over 16.8M elements** -- and it recomputes ``u = h * r``
instead of saving it, which trades ~1 ms of arithmetic for 800 MB of saved activation per live call.

Enable with ``--dit-norm-tail`` or ``COSMOS3_DIT_NORM_TAIL=1``.
"""

from __future__ import annotations

import types

import torch
import tilelang
import tilelang.language as T

from examples.worldmodel.cosmos3.vae_norm_kernel import PASS_CONFIGS, _locked_compile

UB_BYTES = 160 * 1024
_KERNELS = {}


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _scale_tail(N: int, H: int, rows: int):
    """y[n, :] = bf16(w * (h[n, :] * r[n])), one read of h and one write of y per element."""
    last = N - rows

    @T.prim_func
    def main(h32: T.Tensor((N, H), "float32"), r: T.Tensor((N,), "float32"), w: T.Tensor((H,), "float32"),
             y: T.Tensor((N, H), "bfloat16")):
        with T.Kernel(T.ceildiv(N, rows), threads=1, is_npu=True) as cid:
            n0 = T.min(cid * rows, last)
            x_ub = T.alloc_ub((rows, H), "float32")
            out_ub = T.alloc_ub((rows, H), "bfloat16")
            row = T.alloc_ub((H,), "float32")
            w_ub = T.alloc_ub((H,), "float32")
            r_ub = T.alloc_ub((rows,), "float32")
            T.copy(h32[n0:n0 + rows, 0:H], x_ub)
            T.copy(w, w_ub)
            T.copy(r[n0:n0 + rows], r_ub)
            for i in T.serial(rows):
                # r is per row and w is per column: fill makes each a vector so the binary ops stay vector ops.
                T.tile.fill(row, r_ub[i])
                T.tile.mul(x_ub[i, 0:H], x_ub[i, 0:H], row)
                T.tile.mul(x_ub[i, 0:H], x_ub[i, 0:H], w_ub)
                T.tile.cast(out_ub[i, 0:H], x_ub[i, 0:H], "CAST_RINT", H)
            T.copy(out_ub, y[n0:n0 + rows, 0:H])

    return main


def _plan(h: int, n: int) -> int:
    rows = max(1, min((UB_BYTES - 3 * h * 4) // (6 * h), 16, n))
    return rows


def _kernel(n: int, h: int):
    key = (n, h)
    kernel = _KERNELS.get(key)
    if kernel is None:
        rows = _plan(h, n)
        kernel = _locked_compile(f"dittail:{n}:{h}:{rows}", lambda: _scale_tail(n, h, rows))
        _KERNELS[key] = kernel
    return kernel


class _TailScale(torch.autograd.Function):
    """Forward fused; backward repeats autograd's own ops for the same three steps."""

    @staticmethod
    def forward(ctx, h32, r, weight):
        ctx.save_for_backward(h32, r, weight)
        n, h = h32.shape
        y = _kernel(n, h)(h32, r.reshape(-1), weight.reshape(-1).float().contiguous())
        return y

    @staticmethod
    def backward(ctx, grad_out):
        h32, r, weight = ctx.saved_tensors
        u = h32 * r
        dz = grad_out.float()
        dw = (dz * u).sum(0)
        du = dz * weight.float()
        dh = du * r
        dr = (du * h32).sum(-1, keepdim=True)
        return dh, dr, dw.to(weight.dtype)


def _reference_forward(module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    h = x.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + module.eps)
    return (module.weight.float() * h).to(input_dtype)


def _tail_forward(module, x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[-1] % 128:
        return _reference_forward(module, x)
    h = int(x.shape[-1])
    n = x.numel() // h
    if n == 0 or _plan(h, n) < 1:
        return _reference_forward(module, x)
    flat = x.view(n, h)
    h32 = flat.float()
    variance = h32.pow(2).mean(-1, keepdim=True)
    r = torch.rsqrt(variance + float(module.eps))
    return _TailScale.apply(h32, r, module.weight).view_as(x)


def enable_dit_norm_tail(model, verbose: bool = False):
    """Patch every ``Cosmos3NemotronRMSNorm`` with the reference head and the fused tail."""
    try:
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm
    except ImportError:  # older diffusers without the Cosmos3 transformer
        return model

    patched = 0
    for module in model.modules():
        if isinstance(module, Cosmos3NemotronRMSNorm):
            module.forward = types.MethodType(_tail_forward, module)
            patched += 1
    model.dit_norm_tail_modules = patched
    if verbose:
        print(f"[model] fused nemotron RMSNorm tail in {patched} module(s) (bit-exact)", flush=True)
    return model


__all__ = ["enable_dit_norm_tail", "_TailScale", "_reference_forward", "_scale_tail"]
