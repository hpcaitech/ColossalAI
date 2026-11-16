# No ``from __future__ import annotations``: TileLang reads the prim_func annotations as objects, and postponed
# annotations turn them into strings (see check_tilelang_env.py).

"""uint8 video -> normalised bf16, on the device, with one launch per call.

The dataset keeps pixels as uint8 (the same quantisation upstream uses) and the training path must produce the
same values the host path does: ``x.float() / 127.5 - 1`` rounded once to bf16. ``pad_normalize_torch`` gets
there with four aclnn passes over a 1.07 GB fp32 buffer; this kernel does it in one read and one write with the
``ascend_tile`` vector ops. The earlier version of this file was a negative result -- it used ``T.Parallel``,
which this backend scalarises into a per-element ``SetValue`` with a pipe barrier each (~5 GB/s). The tile ops
below compile to real vector instructions, and ``CAST_RINT`` is the only fp32->bf16 mode that matches torch
(``probe_tile_ops.py``).

The reflect pad runs on the uint8 tensor before the fused normalisation (16 columns and 4 rows out of 656x544), so
the flip+cat moves one byte per element instead of two; only the normalisation, which is the expensive part, is
fused.
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from examples.worldmodel.cosmos3.tilelang_lock import locked_compile
from examples.worldmodel.cosmos3.video_uint8 import pad_normalize_torch  # noqa: F401  (reference path / fallback)
from examples.worldmodel.cosmos3.video_uint8 import reflect_pad_last2 as _reflect_pad_last2

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# Nine bytes per element in UB (uint8 + fp16 + fp32 + bf16). The block must divide the call's element count:
# DROID's 8 x 3 x 33 x 540 x 640 = 2^12 * 3^4 * 5^2 * 11, so 4096 is the largest power-of-two divisor. That
# puts an 8-sample call at 66,824 blocks, just over the 65,535 grid cap, so the call is issued as two launches.
BLOCK = 4096
CHUNK_BLOCKS = 65535


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def _uint8_norm_bf16(M: int, block: int = BLOCK):
    """dst[i] = bf16(cast_f32(cast_f16(src[i])) / 127.5 - 1), one launch over the whole batch."""
    if M % block:
        raise ValueError(f"flat element count {M} is not divisible by the kernel block {block}")
    total = M // block

    @T.prim_func
    def main(src: T.Tensor((M,), "uint8"), dst: T.Tensor((M,), "bfloat16")):
        with T.Kernel(total, threads=1, is_npu=True) as cid:
            u8_ub = T.alloc_ub((block,), "uint8")
            h_ub = T.alloc_ub((block,), "float16")
            f_ub = T.alloc_ub((block,), "float32")
            d_ub = T.alloc_ub((block,), "float32")
            o_ub = T.alloc_ub((block,), "bfloat16")
            T.copy(src[cid * block : (cid + 1) * block], u8_ub)
            # The aicore forbids an unsigned-int -> float C++ static cast; CANN's Cast does uint8 -> fp16
            # exactly (0..255 fits) and fp16 -> fp32 is exact, so the arithmetic below is the host path's.
            T.tile.cast(h_ub, u8_ub, "CAST_NONE", block)
            T.tile.cast(f_ub, h_ub, "CAST_NONE", block)
            T.tile.fill(d_ub, 127.5)
            T.tile.div(f_ub, f_ub, d_ub)
            T.tile.add(f_ub, f_ub, -1.0)
            T.tile.cast(o_ub, f_ub, "CAST_RINT", block)
            T.copy(o_ub, dst[cid * block : (cid + 1) * block])

    return main


_KERNELS = {}


def _block_for(m: int) -> int:
    """Largest power-of-two block (up to BLOCK) that divides ``m``.

    A reflect-padded activation is no longer ``8 x 3 x 33 x 540 x 640``, so the fixed 4096 does not always divide
    the element count; choosing a divisor keeps the kernel path instead of silently falling back to torch ops.
    """
    block = BLOCK
    while block > 1 and m % block:
        block //= 2
    return block


def _kernel(m: int, block: int = BLOCK):
    kernel = _KERNELS.get((m, block))
    if kernel is None:
        # Every rank compiles the same padded shape at its first step; serialise the disk-cache write
        # across ranks exactly as the VAE norm kernel does.
        kernel = locked_compile(f"video_pad_norm:{m}:{block}", lambda: _uint8_norm_bf16(m, block))
        _KERNELS[(m, block)] = kernel
    return kernel


def normalize_uint8(video: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] uint8 on the NPU -> the same shape in normalised bf16."""
    if video.dtype != torch.uint8:
        raise TypeError(f"expected uint8 video, got {video.dtype}")
    flat = video.contiguous().view(-1)
    block = _block_for(flat.numel())
    if block < 64:
        return flat.to(torch.float32).div(127.5).sub(1.0).to(torch.bfloat16).view_as(video)
    blocks = flat.numel() // block
    outputs = []
    for start in range(0, blocks, CHUNK_BLOCKS):
        n = min(CHUNK_BLOCKS, blocks - start) * block
        offset = start * block
        outputs.append(_kernel(n, block)(flat[offset : offset + n]))
    dst = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return dst.view_as(video)


def pad_normalize(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """[B, C, T, H, W] uint8 -> [B, C, T, H+pad_h, W+pad_w] normalised bf16.

    The pad runs on the uint8 tensor, before the conversion, so the flip+cat that dominates this path moves one
    byte per element instead of two. The normalisation then covers the padded tensor, which costs less than the pad
    saves (~22 ms of pad against ~4 ms of extra normalise per 8-sample call). Padding only copies values, so this is
    bit-identical to normalising first and padding after.
    """
    return normalize_uint8(_reflect_pad_last2(video, pad_h, pad_w))


def torch_reference(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """The host path in one place: per-sample float32 normalise + 4-D reflect pad on the CPU (fp32)."""
    samples = [F.pad(sample.float() / 127.5 - 1.0, (0, pad_w, 0, pad_h), mode="reflect") for sample in video]
    return torch.stack(samples)
