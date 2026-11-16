"""STATUS: measured, bit-exact, but 73% slower than the permute it removes - not enabled.

AvgDown3D without the permute: sum the group's rows in a kernel, let torch average the contiguous pair.

The idea works arithmetically (160->160 is `torch.equal` over 57M elements) but loses on DMA shape: the reference's
permute is one large contiguous copy, while this kernel issues one small row copy per group member, and the setup
cost dominates. Measured 17.30 ms against the reference's 10.01 ms on the production shape (draft F32). The
4- and 2-group channel decode is also wrong here; it was not fixed because the timing already rules the approach out.
Kept as the record of what the four blocked constructs (F25) can be routed around, and at what cost.

diffusers' ``AvgDown3D`` materialises a permuted copy of the whole activation and then averages a group axis:

    view -> permute(0,1,3,5,7,2,4,6) -> contiguous -> view -> mean(dim=2)

The permute exists only to make the group axis contiguous. The group is ``(t, h, w)`` inside one channel, ``(h, w)``
for two output channels per input channel, or just ``(w)`` for four, so the *w pair* is the only axis that is already
contiguous in NCDHW - and it is the only one a plain row add cannot sum.

That split is the whole idea: a TileLang kernel sums the ``(t, h)`` members of each group with contiguous row reads
and writes one **float32** row per output element, and torch finishes with ``.mean(dim=-1)`` over the contiguous w
pair. Both halves use only operations this build supports; a 2-D ``T.reduce_sum``, column slices, stepped slices and
``T.tile.gather`` all fail here (draft_cosmos3_online_opt_zh.md F25 has the four reproductions).

Bit-exactness: a bf16 value converts to fp32 exactly, and summing at most eight of them is exact in fp32, so the
order of the additions cannot change the result - which is also why ``probe_avgdown_order.py`` found sequential,
pairwise and multiply-by-1/n all identical. The kernel accumulates in fp32, torch's mean over the two fp32 halves is
exact, and dividing by ``ft*fs`` is a power of two, so the final bf16 rounding matches the reference.

The kernel keeps the *full* row width (``w`` rather than ``w/fs``) so every load, add and store is contiguous; the
halving happens in torch, where the innermost axis has unit stride.
"""

# No ``from __future__ import annotations``: TileLang reads the prim_func annotations as objects, and postponed
# evaluation hands it a string instead (same reason as vae_norm_kernel.py and video_pad_norm.py).

import types

import torch

import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _avgdown_rowsum_bf16(B: int, C: int, To: int, Ho: int, W: int, FT: int, FS: int,
                         A: int, Bn: int, GS: int):
    """[B, C, To*FT, Ho*FS, W] bf16 -> [B, out_c, To, Ho, W] fp32, summing each group's (t, h) members.

    The output channel decodes to the group's (c, t, h) start; the group is the ``A x Bn`` block that follows, and the
    w pair is left in place for the caller.
    """
    out_c = C * FT * FS * FS // GS
    group = FT * FS * FS

    @T.prim_func
    def main(x: T.Tensor((B, C, To * FT, Ho * FS, W), "bfloat16"),
             y: T.Tensor((B, out_c, To, Ho, W), "float32")):
        with T.Kernel(B * out_c * To, threads=1, is_npu=True) as cid:
            t_out = cid % To
            rest = cid // To
            oc = rest % out_c
            b = rest // out_c

            start = oc * GS
            c = start // group
            rem = start % group
            a0 = rem // (FS * FS)
            b0 = (rem % (FS * FS)) // FS

            row_ub = T.alloc_ub((W,), "bfloat16")
            row32 = T.alloc_ub((W,), "float32")
            acc = T.alloc_ub((W,), "float32")

            for r in T.serial(Ho):
                T.tile.fill(acc, 0.0)
                for a in T.serial(A):
                    for bb in T.serial(Bn):
                        t_in = t_out * FT + a0 + a
                        h_in = r * FS + b0 + bb
                        # One contiguous row per member: no strides, no gather.
                        T.copy(x[b, c, t_in, h_in, 0:W], row_ub)
                        T.tile.cast(row32, row_ub, "CAST_NONE", W)
                        T.tile.add(acc, acc, row32)
                T.copy(acc, y[b, oc, t_out, r, 0:W])

    return main


_KERNELS = {}


def _kernel(key):
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _avgdown_rowsum_bf16(*key)
        _KERNELS[key] = kernel
    return kernel


def supports(x: torch.Tensor, ft: int, fs: int, out_c: int) -> bool:
    if x.dtype != torch.bfloat16 or x.device.type != "npu" or not x.is_contiguous() or x.dim() != 5:
        return False
    if ft * fs * fs == 1:
        return False
    b, c, t, h, w = x.shape
    if t % ft or h % fs or w % fs:
        return False
    if (c * ft * fs * fs) % out_c:
        return False
    return b * out_c * (t // ft) <= 65535


def avgdown(x: torch.Tensor, ft: int, fs: int, out_c: int) -> torch.Tensor:
    b, c, t, h, w = x.shape
    to, ho, wo = t // ft, h // fs, w // fs
    group = ft * fs * fs
    group_size = c * group // out_c
    a_count = ft if group_size == group else 1
    b_count = fs if group_size >= fs * fs else 1
    total = _kernel((b, c, to, ho, w, ft, fs, a_count, b_count, group_size))(x)
    pair = total.view(b, out_c, to, ho, wo, fs).mean(dim=-1)
    return (pair / (ft * fs)).to(torch.bfloat16)


def _kernel_avgdown_forward(self, x):
    """Drop-in for ``_fast_avgdown_forward``: the kernel path when eligible, the reference path otherwise."""
    if self.factor_t == 1 and self.factor_s == 1 and self.in_channels == self.out_channels:
        return x
    if supports(x, self.factor_t, self.factor_s, self.out_channels):
        return avgdown(x, self.factor_t, self.factor_s, self.out_channels)
    from examples.worldmodel.cosmos3.vae_fast_encoder import _fast_avgdown_forward

    return _fast_avgdown_forward(self, x)


def enable_avgdown_kernel(vae, verbose: bool = False):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    patched = 0
    for module in vae.modules():
        if isinstance(module, AvgDown3D):
            module.forward = types.MethodType(_kernel_avgdown_forward, module)
            patched += 1
    if verbose:
        print(f"[vae] AvgDown3D row-sum kernel installed in {patched} module(s)", flush=True)
    return patched
