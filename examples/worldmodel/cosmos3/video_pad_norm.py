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

The reflect pad stays on the bf16 tensor (16 columns and 4 rows out of 656x544); only the normalisation, which
is the expensive part, is fused.
"""

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

from examples.worldmodel.cosmos3.video_uint8 import reflect_pad_last2 as _reflect_pad_last2
from examples.worldmodel.cosmos3.video_uint8 import pad_normalize_torch  # noqa: F401  (reference path / fallback)

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
            T.copy(src[cid * block:(cid + 1) * block], u8_ub)
            # The aicore forbids an unsigned-int -> float C++ static cast; CANN's Cast does uint8 -> fp16
            # exactly (0..255 fits) and fp16 -> fp32 is exact, so the arithmetic below is the host path's.
            T.tile.cast(h_ub, u8_ub, "CAST_NONE", block)
            T.tile.cast(f_ub, h_ub, "CAST_NONE", block)
            T.tile.fill(d_ub, 127.5)
            T.tile.div(f_ub, f_ub, d_ub)
            T.tile.add(f_ub, f_ub, -1.0)
            T.tile.cast(o_ub, f_ub, "CAST_RINT", block)
            T.copy(o_ub, dst[cid * block:(cid + 1) * block])

    return main


_KERNELS = {}


def _kernel(m: int):
    kernel = _KERNELS.get(m)
    if kernel is None:
        kernel = _uint8_norm_bf16(m)
        _KERNELS[m] = kernel
    return kernel


def normalize_uint8(video: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] uint8 on the NPU -> the same shape in normalised bf16."""
    if video.dtype != torch.uint8:
        raise TypeError(f"expected uint8 video, got {video.dtype}")
    flat = video.contiguous().view(-1)
    blocks = flat.numel() // BLOCK
    if flat.numel() % BLOCK:
        # DROID planes are 540*640 elements, so this only trips on a shape nobody has produced yet.
        return flat.to(torch.float32).div(127.5).sub(1.0).to(torch.bfloat16).view_as(video)
    outputs = []
    for start in range(0, blocks, CHUNK_BLOCKS):
        n = min(CHUNK_BLOCKS, blocks - start) * BLOCK
        offset = start * BLOCK
        outputs.append(_kernel(n)(flat[offset:offset + n]))
    dst = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return dst.view_as(video)


def pad_normalize(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """[B, C, T, H, W] uint8 -> [B, C, T, H+pad_h, W+pad_w] normalised bf16."""
    return _reflect_pad_last2(normalize_uint8(video), pad_h, pad_w)


def torch_reference(video: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """The host path in one place: per-sample float32 normalise + 4-D reflect pad on the CPU (fp32)."""
    samples = [F.pad(sample.float() / 127.5 - 1.0, (0, pad_w, 0, pad_h), mode="reflect") for sample in video]
    return torch.stack(samples)
