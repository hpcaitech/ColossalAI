"""uint8 video -> normalised bf16 with the canvas reflect pad, all on the device.

The pixels are the upstream uint8 quantisation and the affine is the same ``x / 127.5 - 1`` the host path applies;
only the float conversion and the pad move to the device. The device division can round one ulp differently from
the host (1.19e-7 in fp32), far below the VAE's own bf16 noise (~2e-2), and the training loss A/B is the gate.

Kept separate from ``video_pad_norm.py`` so the trainer does not import tilelang: on this stack the TileLang
vector kernel for the same op is launch/Cast bound (~5 GB/s) while these aclnn ops do an 8-sample encode batch in
~27 ms (draft_cosmos3_online_opt_zh.md F6).
"""

from __future__ import annotations

import torch


def reflect_pad_last2(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Reflect-pad the last two dims (bottom/right).

    torch_npu's ``F.pad`` rejects this for 5-D bf16 and negative-step slices are unsupported, so the mirror is a
    positive-step slice plus a flip: the first padded row is ``x[H-2]``, never ``x[H-1]`` (reflect does not repeat
    the edge).
    """
    if pad_h:
        bottom = torch.flip(x[..., -1 - pad_h:-1, :], dims=[-2])
        x = torch.cat([x, bottom], dim=-2)
    if pad_w:
        right = torch.flip(x[..., :, -1 - pad_w:-1], dims=[-1])
        x = torch.cat([x, right], dim=-1)
    return x


def pad_normalize_torch(
    video: torch.Tensor, pad_h: int, pad_w: int, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """[B, C, T, H, W] uint8 on the NPU -> [B, C, T, H+pad_h, W+pad_w] normalised ``dtype``."""
    if video.dtype != torch.uint8:
        raise TypeError(f"expected uint8 video, got {video.dtype}")
    value = video.to(torch.float32) / 127.5 - 1.0
    return reflect_pad_last2(value.to(dtype), pad_h, pad_w)
