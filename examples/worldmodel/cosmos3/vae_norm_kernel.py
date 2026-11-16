# No ``from __future__ import annotations``: TileLang reads the prim_func annotations as objects, and postponed
# annotations turn them into strings (see check_tilelang_env.py).

"""Fused channel-first RMSNorm for the frozen Wan VAE: one HBM read and one write.

The reference forward (diffusers 0.40 ``WanRMS_norm``) on bf16 is::

    normalized = F.normalize(x.float(), dim=1).to(x.dtype)      # cast in, LpNorm, BroadcastTo, RealDiv, cast out
    return normalized * self.scale * self.gamma + self.bias      # three more full-tensor bf16 passes

On Ascend that is eight device kernels and roughly fourteen bf16-tensor passes per call. The 2026-11-14
production profile (batch 8, 33x544x656, fast encoder) puts this chain at 549 ms of the encoder's 2.95 s --
18.6% -- split over 198 calls at twelve distinct shapes, and it already runs at ~1.17 TB/s of effective
traffic. The cost is the number of passes, not the bandwidth, so the fix is a fused kernel that keeps a
``[C, chunk]`` tile in UB, reduces over the channel axis in FP32 and writes the result once.

This backend scalarises ``T.Parallel`` elementwise loops (a per-element ``SetValue`` with a full pipe
barrier each), so every arithmetic step is an ``ascend_tile`` vector op on a whole UB row. ``CAST_RINT`` is
the only fp32->bf16 mode that agrees with torch, and ``CAST_NONE`` is the exact bf16->fp32 widening.

The kernel reproduces the reference's rounding points exactly -- bf16 after the normalisation, bf16 after
``* scale``, bf16 after ``* gamma`` -- and the result is bit-identical to the reference on every production
shape (relL2 0.0 in ``probe_norm_kernel.py``).

The channel axis is the *outer* axis of an NCDHW tensor, so a tile is C strided rows rather than one
contiguous block; one 2-D ``T.copy`` still turns it into a single multi-row DMA per tile. Blocks that overlap
on a non-divisible S recompute the same columns from the same input, so the duplicate writes are identical.
"""

import os
import types

import tilelang
import tilelang.language as T
import torch

from examples.worldmodel.cosmos3.tilelang_lock import locked_compile

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# UB is 192 KB; the tile plus four fp32 rows and the gamma copies must fit inside it.  The legacy
# tile-size cap keeps the historical 160 KiB budget so existing production shapes keep the same DMA
# width; the full-footprint check uses the hardware capacity so a small-C shape cannot plan a tile
# that does not fit.
UB_BYTES = 160 * 1024
UB_CAPACITY_BYTES = 192 * 1024
NORM_EPS = 1e-12


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _channel_rmsnorm(B: int, C: int, S: int, chunk: int, nblk: int, scale: float):
    """One launch per call. B is static because the grid size is baked at compile time.

    ``pack_batch`` micro-batches the encode, so the last call of a step is smaller (batch 4 when 32 samples go
    8 at a time). Each (C, S, B) therefore compiles its own kernel on first use -- serialised across ranks by
    ``_compile_locked``, because tilelang's cache only locks in-process and eight ranks compiling the same
    kernel at once can dlopen a half-written ``kernel_lib.so`` (observed as SIGSEGV on four ranks).
    """
    last_start = S - chunk

    @T.prim_func
    def main(x: T.Tensor((B, C, S), "bfloat16"), gamma: T.Tensor((C,), "float32"), y: T.Tensor((B, C, S), "bfloat16")):
        with T.Kernel(B * nblk, threads=1, is_npu=True) as cid:
            b = cid // nblk
            blk = cid % nblk
            x_ub = T.alloc_ub((C, chunk), "bfloat16")
            row = T.alloc_ub((chunk,), "float32")
            sq = T.alloc_ub((chunk,), "float32")
            acc = T.alloc_ub((chunk,), "float32")
            norm = T.alloc_ub((chunk,), "float32")
            g_row = T.alloc_ub((chunk,), "float32")
            g_ub = T.alloc_ub((C,), "float32")
            # The last block slides back to end exactly at S; on divisible shapes this is blk * chunk.
            s0 = T.min(blk * chunk, last_start)
            T.copy(gamma, g_ub)
            T.copy(x[b, 0:C, s0 : s0 + chunk], x_ub)

            T.tile.fill(acc, 0.0)
            for c in T.serial(C):
                T.tile.cast(row, x_ub[c, 0:chunk], "CAST_NONE", chunk)
                T.tile.mul(sq, row, row)
                T.tile.add(acc, acc, sq)
            # F.normalize clamps the norm itself at eps before dividing; on the squared sum that is eps**2.
            # The division is a real division, not a reciprocal multiply: the vector reciprocal is an
            # approximation, and a ~2^-14 multiplier error flips bf16 rounding on a few percent of elements.
            T.tile.max(acc, acc, NORM_EPS * NORM_EPS)
            T.tile.sqrt(norm, acc)

            for c in T.serial(C):
                # gamma is per-channel, i.e. per row of the tile; fill puts it in a vector so the multiply
                # stays a vector op (a scalar operand to binary_op would be read as a vector at that offset
                # unless the expression is a cast that survives simplification).
                T.tile.fill(g_row, g_ub[c])
                T.tile.cast(row, x_ub[c, 0:chunk], "CAST_NONE", chunk)
                T.tile.div(row, row, norm)
                # The reference rounds to bf16 after the normalisation, then multiplies by scale and by
                # gamma in bf16; reproduce those exact rounding points, then round the final result.
                T.tile.cast(x_ub[c, 0:chunk], row, "CAST_RINT", chunk)
                T.tile.cast(row, x_ub[c, 0:chunk], "CAST_NONE", chunk)
                T.tile.mul(row, row, scale)
                T.tile.cast(x_ub[c, 0:chunk], row, "CAST_RINT", chunk)
                T.tile.cast(row, x_ub[c, 0:chunk], "CAST_NONE", chunk)
                T.tile.mul(row, row, g_row)
                T.tile.cast(x_ub[c, 0:chunk], row, "CAST_RINT", chunk)

            T.copy(x_ub, y[b, 0:C, s0 : s0 + chunk])

    return main


# Two-pass variant: the tile is a single row wide, so each DMA is `chunk * 2` contiguous bytes instead of the
# one-pass tile's `chunk` bytes strided by S. The one-pass kernel spends its time on ~1 KB row DMAs (measured
# 356 GB/s on the widest shape against ~1 TB/s for a contiguous copy); the two-pass version reads twice and
# writes once at a much larger row width. The arithmetic per column is unchanged and in the same order, so the
# result stays bit-identical.
TWO_PASS_CHUNK = 8192


@tilelang.jit(pass_configs=PASS_CONFIGS)
def _channel_rmsnorm_reduce(B: int, C: int, S: int, chunk: int, nblk: int):
    last_start = S - chunk

    @T.prim_func
    def main(x: T.Tensor((B, C, S), "bfloat16"), norm_out: T.Tensor((B, S), "float32")):
        with T.Kernel(B * nblk, threads=1, is_npu=True) as cid:
            b = cid // nblk
            blk = cid % nblk
            row = T.alloc_ub((chunk,), "float32")
            sq = T.alloc_ub((chunk,), "float32")
            acc = T.alloc_ub((chunk,), "float32")
            stage = T.alloc_ub((chunk,), "bfloat16")
            s0 = T.min(blk * chunk, last_start)
            T.tile.fill(acc, 0.0)
            for c in T.serial(C):
                T.copy(x[b, c, s0 : s0 + chunk], stage)
                T.tile.cast(row, stage, "CAST_NONE", chunk)
                T.tile.mul(sq, row, row)
                T.tile.add(acc, acc, sq)
            T.tile.max(acc, acc, NORM_EPS * NORM_EPS)
            T.tile.sqrt(acc, acc)
            T.copy(acc, norm_out[b, s0 : s0 + chunk])

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def _channel_rmsnorm_apply(B: int, C: int, S: int, chunk: int, nblk: int, scale: float):
    last_start = S - chunk

    @T.prim_func
    def main(
        x: T.Tensor((B, C, S), "bfloat16"),
        gamma: T.Tensor((C,), "float32"),
        norm_in: T.Tensor((B, S), "float32"),
        y: T.Tensor((B, C, S), "bfloat16"),
    ):
        with T.Kernel(B * nblk, threads=1, is_npu=True) as cid:
            b = cid // nblk
            blk = cid % nblk
            row = T.alloc_ub((chunk,), "float32")
            norm = T.alloc_ub((chunk,), "float32")
            g_row = T.alloc_ub((chunk,), "float32")
            half = T.alloc_ub((chunk,), "bfloat16")
            stage = T.alloc_ub((chunk,), "bfloat16")
            g_ub = T.alloc_ub((C,), "float32")
            s0 = T.min(blk * chunk, last_start)
            T.copy(gamma, g_ub)
            T.copy(norm_in[b, s0 : s0 + chunk], norm)
            for c in T.serial(C):
                T.tile.fill(g_row, g_ub[c])
                T.copy(x[b, c, s0 : s0 + chunk], stage)
                T.tile.cast(row, stage, "CAST_NONE", chunk)
                T.tile.div(row, row, norm)
                T.tile.cast(half, row, "CAST_RINT", chunk)
                T.tile.cast(row, half, "CAST_NONE", chunk)
                T.tile.mul(row, row, scale)
                T.tile.cast(half, row, "CAST_RINT", chunk)
                T.tile.cast(row, half, "CAST_NONE", chunk)
                T.tile.mul(row, row, g_row)
                # Cast cannot target global memory; round into the UB row and copy it out.
                T.tile.cast(half, row, "CAST_RINT", chunk)
                T.copy(half, y[b, c, s0 : s0 + chunk])

    return main


def _plan_two_pass(c: int, s: int):
    """Column block for the two-pass variant: one row wide, bounded by the apply kernel's UB footprint."""
    max_chunk = min(
        (UB_BYTES // 14) // 128 * 128,
        # row + norm + g_row (12 B) + half + stage (4 B) per element, plus the C-wide gamma row.
        ((UB_CAPACITY_BYTES - 4 * c) // 16) // 128 * 128,
    )
    chunk = min(TWO_PASS_CHUNK, max_chunk, s // 128 * 128)
    if chunk < 128:
        return None
    if s % chunk == 0:
        return chunk, s // chunk
    if s < chunk:
        return None
    return chunk, s // chunk + 1


def _plan(c: int, s: int):
    """Pick the UB tile width; returns (chunk, nblk) or None when the shape is not worth a kernel."""
    max_chunk = min(
        (UB_BYTES // (c * 2)) // 128 * 128,
        # x tile (2 B) + five fp32 rows (20 B) per element, plus the C-wide gamma row.
        ((UB_CAPACITY_BYTES - 4 * c) // (c * 2 + 20)) // 128 * 128,
    )
    chunk = min(max_chunk, s // 128 * 128)
    if chunk < 128:
        return None
    if s % chunk == 0:
        return chunk, s // chunk
    if s < chunk:
        return None
    return chunk, s // chunk + 1


_KERNELS = {}
_GAMMA = {}
# The two-pass path only pays off where the one-pass tile's row DMAs are small; below this many columns the
# launch overhead dominates instead (see probe_norm_twopass.py).
TWO_PASS_MIN_S = 65536


# Shared with the video pad/norm kernel; kept under this name for the VAE and dit-tail call sites.
# The helper creates the lock directory next to the shared TileLang cache instead of silently falling
# back to a per-node /tmp lock.
_locked_compile = locked_compile


def _compile_locked(key, plan, b: int, c: int, s: int, scale: float):
    signature = f"one:{b}:{c}:{s}:{plan}:{scale}"
    return _locked_compile(signature, lambda: _KERNELS.get(key) or _channel_rmsnorm(b, c, s, *plan, scale))


def _kernel(b: int, c: int, s: int, scale: float):
    key = (b, c, s, scale)
    kernel = _KERNELS.get(key)
    if kernel is None:
        plan = _plan(c, s)
        if plan is None:
            raise ValueError(f"no fused-norm plan for C={c} S={s}")
        kernel = _compile_locked(key, plan, b, c, s, scale)
        _KERNELS[key] = kernel
    return kernel


def _two_pass_kernels(b: int, c: int, s: int, scale: float):
    key = ("two_pass", b, c, s, scale)
    kernels = _KERNELS.get(key)
    if kernels is None:
        plan = _plan_two_pass(c, s)
        if plan is None:
            raise ValueError(f"no two-pass plan for C={c} S={s}")
        chunk, nblk = plan
        # Same cross-rank compile lock as the one-pass path: the first step compiles both kernels on all eight
        # ranks at once, which crashed with SIGSEGV before the lock was added here.
        reduce_k = _locked_compile(
            f"reduce:{b}:{c}:{s}:{plan}",
            lambda: _channel_rmsnorm_reduce(b, c, s, chunk, nblk),
        )
        apply_k = _locked_compile(
            f"apply:{b}:{c}:{s}:{plan}:{scale}",
            lambda: _channel_rmsnorm_apply(b, c, s, chunk, nblk, scale),
        )
        kernels = (reduce_k, apply_k)
        _KERNELS[key] = kernels
    return kernels


def _gamma_fp32(module) -> torch.Tensor:
    """The bf16 parameter widened to fp32 once (the widening is exact); the VAE is frozen.

    The key pairs the module id with the parameter's current storage address, so a recycled module id or a
    replaced parameter misses the cache instead of serving the wrong gamma.
    """
    source = module.gamma.detach()
    key = (id(module), source.data_ptr())
    gamma = _GAMMA.get(key)
    if gamma is None:
        gamma = source.reshape(-1).float().contiguous()
        _GAMMA[key] = gamma
    return gamma


def _reference_forward(module, x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return normalized * module.scale * module.gamma + module.bias


def _kernel_forward(module, x: torch.Tensor) -> torch.Tensor:
    if (
        x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not module.channel_first
        or isinstance(module.bias, torch.Tensor)
        or x.dim() < 3
    ):
        return _reference_forward(module, x)
    b, c = int(x.shape[0]), int(x.shape[1])
    s = x.numel() // (b * c)
    # Decide the path before the one-pass gate: a shape can be too large for the one-pass tile but
    # still fit the two-pass apply footprint.
    two_pass_plan = None
    if os.environ.get("COSMOS3_NORM_TWOPASS", "1") == "1" and s >= TWO_PASS_MIN_S:
        two_pass_plan = _plan_two_pass(c, s)
    if two_pass_plan is None and _plan(c, s) is None:
        return _reference_forward(module, x)
    scale = float(module.scale)
    flat = x.view(b, c, s)
    gamma = _gamma_fp32(module)
    # Default on: bit-identical to the one-pass kernel and 1.4-2.3x faster on the wide shapes
    # (358 -> 756 GB/s on 8x160x356864). COSMOS3_NORM_TWOPASS=0 restores the one-pass path.
    if two_pass_plan is not None:
        reduce_k, apply_k = _two_pass_kernels(b, c, s, scale)
        norm = torch.empty((b, s), dtype=torch.float32, device=x.device)
        out = torch.empty_like(flat)
        reduce_k(flat, norm)
        apply_k(flat, gamma, norm, out)
        return out.view_as(x)
    return _kernel(b, c, s, scale)(flat, gamma).view_as(x)


def enable_kernel_rmsnorm(vae, verbose: bool = False):
    """Patch every channel-first ``WanRMS_norm`` with the fused forward. Returns the VAE."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    patched = 0
    for module in vae.modules():
        if isinstance(module, WanRMS_norm) and module.channel_first:
            module.forward = types.MethodType(_kernel_forward, module)
            patched += 1
    vae.kernel_rmsnorm_modules = patched
    if verbose:
        print(f"[vae] fused channel RMSNorm in {patched} module(s)", flush=True)
    return vae


__all__ = ["enable_kernel_rmsnorm", "_channel_rmsnorm", "_plan"]
