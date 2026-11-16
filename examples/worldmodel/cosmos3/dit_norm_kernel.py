"""Fused ``Cosmos3NemotronRMSNorm`` for the DiT, with a backward that is bit-identical to the reference.

The backbone built with ``hidden_act="relu2"`` uses ``Cosmos3NemotronRMSNorm``, which is not the fused NPU op: its
``forward`` upcasts to fp32 and runs ``pow(2) -> mean(-1) -> * rsqrt(v+eps) -> * weight.float() -> to(bf16)`` as
five separate kernels, four times per layer, for 28 layers. Measured on a steady step (``analyze_profile_deep.py``),
394 of those chains run per step and cost **1.33 s (5.3% of the step)**: 6.3 ms for one call on a full-size tile,
of which 0.99 ms is the upcast, 1.31 ms the square, 0.70 ms the mean, 2.2 ms the two multiplies and 1.1 ms the
downcast.

What can and cannot be reproduced exactly, both measured by ``scripts/probe_nemotron_norm.py`` and
``scripts/probe_dit_norm.py``:

* ``mean(-1)`` **cannot** be reproduced bitwise. Twenty-four candidate reductions (1/2/4/8/16/32/64 lanes, 2-32
  blocks, sequential/tree folds) were compared against it; the best is off by one fp32 ulp on 35% of rows. So a
  hand-written kernel that recomputes the variance is out, whatever its speed.
* ``torch_npu.npu_rms_norm``, the vendor's fused op, lands within **12 elements out of 16.8M** (7e-7) of the same
  chain, and ``torch.rsqrt`` is bitwise ``1/sqrt``.
* The backward written here out of plain torch ops -- the ones autograd would have run -- is **bit-identical** to
  the reference gradient for both ``x`` and ``weight`` (0 mismatches over 16.8M and 2048 elements).

So the module patches the forward onto the vendor op and keeps a graph by wrapping it in a ``Function`` whose
backward repeats the reference chain. It is **off by default**: the forward is 7e-7 away from the decomposed chain,
and whether that is acceptable is a precision decision, not a kernel one. ``COSMOS3_DIT_NORM_KERNEL=1`` or
``--dit-norm-kernel`` turns it on.
"""

from __future__ import annotations

import os
import types

import torch


def _reference_forward(module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    h = x.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + module.eps)
    return (module.weight.float() * h).to(input_dtype)


class _FusedNemotronNorm(torch.autograd.Function):
    """Vendor fused forward, reference-exact backward.

    The backward is the chain autograd would have produced, written out: the gradient of ``w * (h * r)`` with
    ``r = rsqrt(mean(h**2) + eps)``, in the same association order, so the result is bit-identical (measured).
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        import torch_npu

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        h = x.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(variance + ctx.eps)
        u = h * r
        dz = grad_out.float()
        dw = (dz * u).sum(0)
        du = dz * weight.float()
        dh = du * r
        dr = (du * h).sum(-1, keepdim=True)
        dv = dr * r.pow(3) * -0.5
        dh = dh + dv * 2.0 * h / h.shape[-1]
        return dh.to(x.dtype), dw.to(weight.dtype), None


_VERIFY = {"on": False, "calls": 0, "elements": 0, "mismatched": 0, "worst_rel": 0.0}


def _fused_forward(module, x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        return _reference_forward(module, x)
    out = _FusedNemotronNorm.apply(x, module.weight, float(module.eps))
    if _VERIFY["on"]:
        # Compare against the fp32 chain on this call's *real* activation. The synthetic probe used N(0, 0.5) inputs;
        # the trained model's activations have a different distribution, and that is what decides whether the 1-ulp
        # forward difference is confined to the rate the probe measured.
        import atexit

        with torch.no_grad():
            reference = _reference_forward(module, x)
        different = int((out != reference).sum())
        _VERIFY["calls"] += 1
        _VERIFY["elements"] += reference.numel()
        _VERIFY["mismatched"] += different
        _VERIFY["worst_rel"] = max(
            _VERIFY["worst_rel"],
            float((out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)),
        )
        if _VERIFY["calls"] == 1:
            atexit.register(
                lambda: print(
                    f"[dit-norm-verify] {_VERIFY['calls']} calls, {_VERIFY['mismatched']}/{_VERIFY['elements']} elements "
                    f"differ ({100 * _VERIFY['mismatched'] / max(_VERIFY['elements'], 1):.5f}%), worst per-call relL2 "
                    f"{_VERIFY['worst_rel']:.2e}",
                    flush=True,
                )
            )
    return out


def enable_dit_norm(model, verbose: bool = False):
    import os as _os

    _VERIFY["on"] = _os.environ.get("COSMOS3_DIT_NORM_VERIFY", "0") == "1"
    """Patch every ``Cosmos3NemotronRMSNorm`` in the DiT with the fused op + exact backward."""
    try:
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm
    except ImportError:  # older diffusers without the Cosmos3 transformer
        return model

    patched = 0
    for module in model.modules():
        if isinstance(module, Cosmos3NemotronRMSNorm):
            module.forward = types.MethodType(_fused_forward, module)
            patched += 1
    model.fused_dit_norm_modules = patched
    if verbose:
        print(
            f"[model] fused nemotron RMSNorm in {patched} module(s) "
            f"(forward 7e-7 from the fp32 chain, backward bit-identical)",
            flush=True,
        )
    return model


def enable_from_env(model):
    if os.environ.get("COSMOS3_DIT_NORM_KERNEL", "0") == "1":
        enable_dit_norm(model, verbose=True)
    return model


# ------------------------------------------------------------------------------------------------------------------
# mean(-1) -> sum(-1) / H: bit-identical, and it moves a full-tensor division out of the backward.
#
# ``mean(-1)`` divides inside the reduce kernel, so its *backward* is ``grad.expand() / numel`` -- a division over the
# whole [tokens, 2048] fp32 activation. Writing ``sum(-1) / H`` instead puts that division on the [tokens, 1] result
# (forward) and on the [tokens, 1] gradient (backward); the expand then feeds ``* 2h``, which autograd has to do
# anyway. H is 2048, a power of two, so dividing by it is exact and ``sum / H == mean`` bit for bit -- verified at
# [8192, 2048] in probe_nemotron_norm.py ("mean == sum/2048: True").
#
# It shows up in the profile: ``aclnnDivs_RealDiv`` runs 706 times on one node (145 ms) and 822 times on two nodes
# (763 ms, with 44 ms outliers) -- about one per norm backward.


def _sum_mean_forward(module, x: torch.Tensor) -> torch.Tensor:
    h = x.float()
    # ``square`` instead of ``pow(2)``: the two are bitwise equal on this stack (probe_pow_square.py, bf16 and fp32,
    # including 1e-3 and 1e3 magnitudes, 0 differing elements), but ``square`` lowers to a multiply while ``pow``
    # runs the transcendental path -- the profile counts 770 ``aclnnPowTensorScalar`` kernels per step, all from the
    # 198 norms. Same arithmetic, one cheaper kernel family.
    variance = h.square().sum(-1, keepdim=True) / h.shape[-1]
    h = h * torch.rsqrt(variance + module.eps)
    return (module.weight.float() * h).to(x.dtype)


def enable_sum_mean_norm(model, verbose: bool = False):
    """Patch every ``Cosmos3NemotronRMSNorm`` to `sum(-1)/H` instead of `mean(-1)`; bit-identical, fewer full divs."""
    try:
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3NemotronRMSNorm
    except ImportError:  # older diffusers without the Cosmos3 transformer
        return model

    patched = 0
    for module in model.modules():
        if isinstance(module, Cosmos3NemotronRMSNorm):
            module.forward = types.MethodType(_sum_mean_forward, module)
            patched += 1
    model.sum_mean_norm_modules = patched
    if verbose:
        print(f"[model] sum/H instead of mean in {patched} norm module(s) (bit-identical)", flush=True)
    return model


__all__ = [
    "enable_dit_norm",
    "enable_from_env",
    "enable_sum_mean_norm",
    "_reference_forward",
    "_FusedNemotronNorm",
    "_sum_mean_forward",
]
