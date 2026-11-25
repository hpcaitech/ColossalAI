import math
from typing import Literal, Sequence, Tuple

import torch
from torch import Tensor


def make_rotary_3d(
    position_ids: Tensor,
    axes_dims: Sequence[int],
    theta: float = 10000.0,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Build rotary frequencies for explicit temporal, height and width coordinates.

    Args:
        position_ids: Coordinates in ``(time, height, width)`` order, shaped
            ``[B, S, 3]`` for dense attention or ``[N, 3]`` for packed attention.
            Packed samples retain their own coordinates; the packed token index
            is not a position. Use integer, float32 or float64 coordinates, not
            coordinates already rounded to float16 or bfloat16.
        axes_dims: Three positive even feature dimensions, summing to the head
            dimension. Wan uses ``(44, 42, 42)`` for a 128-dimensional head;
            Open-Sora uses ``(16, 56, 56)``. Each axis contributes half as many
            frequencies as feature dimensions, regardless of the pairing mode.
        theta: Positive finite frequency base.
        compute_dtype: Float32 by default; float64 is available for references
            and applications requiring higher position precision.

    Returns:
        Cosines and sines with shape ``[B, S, D / 2]`` or ``[N, D / 2]``, on the
        coordinate device and in ``compute_dtype``. Frequencies are ordered by
        time, height, then width, with one value per feature pair.

    Coordinates are not inferred from a sequence length or a video shape. The
    model adapter supplies its post-patch grid and any text coordinates. This
    function is differentiable for floating-point coordinates and has no cache.
    """
    if position_ids.ndim not in (2, 3) or position_ids.shape[-1] != 3:
        raise ValueError("position_ids must have shape [B, S, 3] or [N, 3]")
    if position_ids.dtype not in (torch.int32, torch.int64, torch.float32, torch.float64):
        raise TypeError("position_ids must have dtype int32, int64, float32 or float64")
    if len(axes_dims) != 3 or any(type(dim) is not int or dim <= 0 or dim % 2 for dim in axes_dims):
        raise ValueError("axes_dims must contain three positive even integers")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be a positive finite number")
    if compute_dtype not in (torch.float32, torch.float64):
        raise TypeError("compute_dtype must be float32 or float64")

    positions = position_ids.to(dtype=compute_dtype)
    angles = []
    for axis, dim in enumerate(axes_dims):
        exponent = torch.arange(0, dim, 2, device=positions.device, dtype=compute_dtype) / dim
        inverse_frequency = 1.0 / (theta**exponent)
        angles.append(positions[..., axis, None] * inverse_frequency)
    angles = torch.cat(angles, dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, pairing: Literal["adjacent", "half"] = "adjacent") -> Tensor:
    """Rotate dense ``[B, H, S, D]`` or packed ``[N, H, D]`` Q/K features.

    Dense cos/sin have shape ``[B, S, D / 2]``; a batch dimension of 1 may
    broadcast to all samples. Packed cos/sin have shape ``[N, D / 2]``. Heads
    share frequencies. Sequence dimensions must match exactly: there is no
    implicit offset, truncation or broadcasting between token positions.

    ``pairing="adjacent"`` rotates ``(x[0], x[1]), (x[2], x[3]), ...``.
    ``pairing="half"`` rotates ``(x[0], x[D/2]), (x[1], x[D/2+1]), ...``
    across the *whole head*, not separately within each axis segment. The
    modes are different checkpoint conventions and are not interchangeable.

    The operation is out of place and supports non-contiguous inputs and
    autograd. Computation uses at least float32, or float64 when x or the
    frequencies use float64. The result preserves x's shape and dtype.
    Calling this function for Q and K leaves V untouched.
    """
    if pairing not in ("adjacent", "half"):
        raise ValueError("pairing must be 'adjacent' or 'half'")
    if x.ndim not in (3, 4) or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("x must have shape [B, H, S, D] or [N, H, D] with positive even D")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("x must have dtype float16, bfloat16, float32 or float64")
    if cos.dtype != sin.dtype or cos.dtype not in (torch.float32, torch.float64):
        raise TypeError("cos and sin must have the same float32 or float64 dtype")
    if cos.device != x.device or sin.device != x.device:
        raise ValueError("x, cos and sin must be on the same device")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape")
    if x.ndim == 4:
        if (
            cos.ndim != 3
            or cos.shape[0] not in (1, x.shape[0])
            or cos.shape[1] != x.shape[2]
            or cos.shape[2] != x.shape[-1] // 2
        ):
            raise ValueError("dense cos and sin must have shape [B or 1, S, D / 2]")
    elif cos.shape != (x.shape[0], x.shape[-1] // 2):
        raise ValueError("packed cos and sin must have shape [N, D / 2]")

    compute_dtype = torch.promote_types(x.dtype, cos.dtype)
    x_float = x.to(dtype=compute_dtype)
    cos = cos.unsqueeze(1).to(dtype=compute_dtype)
    sin = sin.unsqueeze(1).to(dtype=compute_dtype)
    if pairing == "adjacent":
        first, second = x_float[..., 0::2], x_float[..., 1::2]
    else:
        first, second = x_float.chunk(2, dim=-1)
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    if pairing == "adjacent":
        output = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
    else:
        output = torch.cat((rotated_first, rotated_second), dim=-1)
    return output.to(dtype=x.dtype)
