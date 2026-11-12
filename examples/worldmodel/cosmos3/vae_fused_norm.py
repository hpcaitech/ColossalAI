"""Conservative RMSNorm replacement for the frozen Wan VAE.

This is a replacement forward, not a fused kernel: it keeps the reference's FP32
normalization and cast-before-affine order, and the ``--fuse-vae-norm`` flag
keeps its historical name only. Folding ``scale / norm`` into a low precision
multiplier is unsafe: FP16 overflows for near-zero inputs, and saturating that
multiplier changes the normalization. Only the addition of a literal zero bias
is omitted. This helper makes no end-to-end latent accuracy or speed guarantee;
validate experimental recipes with their actual checkpoint, dtype and input
distribution.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
    dim = 1 if self.channel_first else -1
    # Divide in FP32, before casting: scale/norm need not be representable in
    # FP16 even when the normalized output is perfectly representable.
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    # FP32 accumulation can itself overflow for large BF16/FP32 values. Scale
    # only that range before the reduction. Below the overflow bound divisor=1,
    # preserving the reference's rounding and its eps=1e-12 near-zero behavior.
    peak = work.abs().amax(dim=dim, keepdim=True)
    bound = (torch.finfo(work.dtype).max / work.shape[dim]) ** 0.5 / 2
    divisor = torch.where(peak > bound, peak, torch.ones_like(peak))
    normalized = F.normalize(work / divisor, dim=dim).to(x.dtype)
    out = normalized * self.scale * self.gamma
    # Tensor biases, including trainable zero biases, must retain their gradient.
    if isinstance(self.bias, torch.Tensor) or self.bias != 0:
        out = out + self.bias
    return out


def enable_fused_rmsnorm(vae, verbose: bool = False):
    """Patch every ``WanRMS_norm`` with the conservative forward. Returns the VAE.

    The name is kept for the existing flag and call sites; no fused kernel is installed.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    patched = 0
    for module in vae.modules():
        if isinstance(module, WanRMS_norm):
            module.forward = types.MethodType(_fused_forward, module)
            patched += 1
    vae.fused_rmsnorm_modules = patched
    if verbose:
        print(f"[vae] safe RMSNorm in {patched} module(s)", flush=True)
    return vae
