"""``ColorJitter`` with the hue step compiled, for the dataloader workers.

``ColorJitter`` is 90% of the augmentation chain and ``adjust_hue`` is 71% of ``ColorJitter`` (measured in
``scripts/probe_augment_cost.py``: 1.63 s of 2.29 s per sample at 4 threads, 0.98 s of 1.32 s at 8). The hue step is
a full RGB -> HSV -> RGB round trip, i.e. a long chain of 400 MB temporaries on CPU; inductor fuses it.
``scripts/probe_augment_compile.py`` measures the compiled step at **598 ms vs 992 ms eager, bit-identical
(0/102,643,200 elements differ)**.

The replacement overrides ``transform()`` only: ``make_params`` -- and therefore the ``randperm(4)``, the four uniform
draws and the order the adjustments are applied in -- stays torchvision's own, and the other three adjustments stay
eager (together they are 0.33 s).

Enable with ``COSMOS3_FAST_JITTER=1``; the inductor cache must live on local disk (``TORCHINDUCTOR_CACHE_DIR``),
like the other extension caches, or every worker recompiles on the network filesystem.
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

_COMPILED_HUE = None
_COMPILE_UNAVAILABLE = False


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """torchvision's ``_rgb_to_hsv``, copied verbatim (same ops, same in-place order, same values)."""
    r, g, _ = image.unbind(dim=-3)
    minc, maxc = torch.aminmax(image, dim=-3)
    eqc = maxc == minc
    channels_range = maxc - minc
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)
    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g
    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))
    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """torchvision's ``_hsv_to_rgb``, copied verbatim."""
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)
    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)
    vpqt = torch.stack((v, p, q, t), dim=-3)
    select = torch.tensor([[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]], dtype=torch.long)
    select = select.to(device=img.device, non_blocking=True)
    select = select[:, i]
    if select.ndim > 3:
        select = select.moveaxis(0, -3)
    return vpqt.gather(-3, select)


def _hue_math(image: torch.Tensor, hue_factor: torch.Tensor) -> torch.Tensor:
    """``F.adjust_hue`` with the factor as a 0-dim tensor, so inductor never specialises on its value."""
    hsv = _rgb_to_hsv(image)
    h, s, v = hsv.unbind(dim=-3)
    h.add_(hue_factor).remainder_(1.0)
    return _hsv_to_rgb(torch.stack((h, s, v), dim=-3))


def _compiled_adjust_hue(image: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """``F.adjust_hue`` through inductor, compiled once per worker process.

    The factor goes in as a 0-dim tensor: a Python float argument makes dynamo guard on its value, and the jitter
    draws a fresh value every call, which turns the compiled function into a recompile per sample (measured 3.7 s
    per call against 1.2 s eager).
    """
    global _COMPILED_HUE, _COMPILE_UNAVAILABLE
    factor = torch.tensor(hue_factor)
    if _COMPILE_UNAVAILABLE:
        return _hue_math(image, factor)
    if _COMPILED_HUE is None:
        # torch.compile's device probe goes through torch_npu, and a DataLoader worker never calls set_device, so
        # each worker ends up holding a context on card 0 (~110 MB; measured 32 contexts = 3.5 GB on rank 0's card).
        # Hiding the devices with ASCEND_RT_VISIBLE_DEVICES="" here does NOT prevent it -- the context is already
        # created by the time this runs. See F21.
        _COMPILED_HUE = torch.compile(_hue_math, dynamic=False)
    try:
        return _COMPILED_HUE(image, factor)
    except Exception as error:
        # A host without the Ascend triton fork raises here ("module 'triton.language.extra' has no attribute
        # 'ascend'"), and this runs inside a DataLoader worker: raising takes the whole input pipeline down for
        # an optimisation that is only meant to save CPU time. The eager path is the same function, so the
        # values do not change -- only the speed.
        _COMPILE_UNAVAILABLE = True
        print(f"[augment] hue compilation unavailable ({type(error).__name__}: {error}); staying eager", flush=True)
        return _hue_math(image, factor)


class FastColorJitter(T.ColorJitter):
    """``T.ColorJitter`` with only the hue adjustment compiled; draws and values are unchanged."""

    def transform(self, inpt, params):
        output = inpt
        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]
        for fn_id in params["fn_idx"]:
            if fn_id == 0 and brightness_factor is not None:
                output = self._call_kernel(F.adjust_brightness, output, brightness_factor=brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                output = self._call_kernel(F.adjust_contrast, output, contrast_factor=contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                output = self._call_kernel(F.adjust_saturation, output, saturation_factor=saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                output = _compiled_adjust_hue(output, hue_factor)
        return output


def build_fast_compose(crop_hw, out_hw):
    """The dataset's exact chain with the compiled jitter."""
    return T.Compose(
        [
            T.RandomCrop(crop_hw),
            T.Resize(out_hw, antialias=True),
            FastColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ]
    )


__all__ = ["FastColorJitter", "build_fast_compose"]
