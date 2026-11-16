"""What do the encoder's layout ops actually cost, and what would a fused kernel have to beat?

Three families show up in the bf16 encode profile as pure memory traffic:

  aclnnCat_ConcatD                 133 ms   the feat-cache ``cat([cache, x], dim=2)`` before every causal conv
  aclnnConstantPadNd_PadV3          59 ms   ``F.pad`` inside AvgDown3D and the prime call's temporal pad
  aclnnInplaceCopy_TransposeAiCore 109 ms   ``permute(...).contiguous()`` in AvgDown3D / downsampler / attention

For each: the reference chain, the cheapest torch equivalent that keeps the same values, and the bandwidth floor
(a plain contiguous copy of the same tensor). A TileLang kernel only pays off if it lands near the floor.

    python scripts/probe_layout_ops.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def bench(fn, iters=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def gbps(ms, *tensors, factor=1):
    moved = sum(t.numel() * t.element_size() for t in tensors) * factor
    return moved / (ms * 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    vae.requires_grad_(False)

    avgdowns = [m for m in vae.modules() if isinstance(m, AvgDown3D)]
    print("[avgdown] modules")
    for module in avgdowns:
        print(
            f"  factor_t={module.factor_t} factor_s={module.factor_s} group_size={module.group_size} "
            f"in={module.in_channels} out={module.out_channels}"
        )
    deep = max(avgdowns, key=lambda m: m.in_channels * m.factor_s**2)

    # --- cat -----------------------------------------------------------------------------------------------
    # Widest cache concat in the encoder: 512 channels, two cached frames + four new ones.
    cache = torch.randn(args.batch, 512, 2, 68, 82, device=device, dtype=torch.bfloat16)
    fresh = torch.randn(args.batch, 512, 4, 68, 82, device=device, dtype=torch.bfloat16)
    buffer = torch.empty(args.batch, 512, 6, 68, 82, device=device, dtype=torch.bfloat16)
    buffer[:, :, :2].copy_(cache)

    ms_cat = bench(lambda: torch.cat([cache, fresh], dim=2), args.iters)
    ms_into = bench(lambda: buffer[:, :, 2:].copy_(fresh), args.iters)
    ms_copy = bench(lambda: buffer.copy_(buffer), args.iters)
    print(
        f"\n[cat]    cat(cache,fresh) {ms_cat:7.3f} ms  {gbps(ms_cat, cache, fresh, factor=2):6.0f} GB/s"
        f"   copy_into_slice {ms_into:7.3f} ms  {gbps(ms_into, fresh, factor=2):6.0f} GB/s"
        f"   full clone {ms_copy:7.3f} ms  {gbps(ms_copy, buffer, factor=2):6.0f} GB/s"
    )

    # --- pad -----------------------------------------------------------------------------------------------
    small = torch.randn(args.batch, 512, 4, 68, 82, device=device, dtype=torch.bfloat16)
    ms_pad = bench(lambda: F.pad(small, (0, 0, 0, 0, 2, 0)), args.iters)
    ms_pad0 = bench(lambda: F.pad(small, (0, 0, 0, 0, 0, 0)), args.iters)
    ms_pad_into = bench(lambda: buffer[:, :, 2:].copy_(small), args.iters)
    print(
        f"[pad]    F.pad t+2 {ms_pad:7.3f} ms  {gbps(ms_pad, small):6.0f} GB/s"
        f"   F.pad zero-pad {ms_pad0:7.3f} ms   zeros+copy {ms_pad_into:7.3f} ms"
    )

    # --- avgdown transpose + mean --------------------------------------------------------------------------
    x = torch.randn(args.batch, deep.in_channels, 4, 136, 164, device=device, dtype=torch.bfloat16)
    ft, fs = deep.factor_t, deep.factor_s
    ms_perm = bench(
        lambda: x.view(args.batch, deep.in_channels, 4 // ft, ft, 136 // fs, fs, 164 // fs, fs)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .contiguous(),
        args.iters,
    )
    ms_ref = bench(lambda: deep(x), args.iters)
    view = (
        x.view(args.batch, deep.in_channels, 4 // ft, ft, 136 // fs, fs, 164 // fs, fs)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .contiguous()
    )
    ms_mean = bench(
        lambda: view.view(args.batch, deep.out_channels, deep.group_size, 4 // ft, 136 // fs, 164 // fs).mean(dim=2),
        args.iters,
    )

    # Direct formula, no materialised permutation: group the channel dim into (out, group) and take the group
    # mean, with each group member read at its own spatial phase.
    groups = x.view(args.batch, deep.out_channels, deep.group_size, 4, 136, 164)

    def direct():
        acc = None
        for g in range(deep.group_size):
            dt, rest = divmod(g, fs * fs)
            dh, dw = divmod(rest, fs)
            part = groups[:, :, g, dt::ft, dh::fs, dw::fs]
            acc = part.float() if acc is None else acc + part.float()
        return (acc / deep.group_size).to(x.dtype)

    reference = deep(x)
    fused = direct()
    equal = torch.equal(fused, reference)
    rel = float((fused.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
    ms_direct = bench(direct, args.iters)
    print(f"[avgdown] shape {tuple(x.shape)} factor_t={ft} factor_s={fs}")
    print(f"  permute+contiguous {ms_perm:7.3f} ms  {gbps(ms_perm, x, factor=2):6.0f} GB/s")
    print(f"  full reference    {ms_ref:7.3f} ms")
    print(f"  mean only         {ms_mean:7.3f} ms")
    print(f"  direct torch      {ms_direct:7.3f} ms  bit_equal={equal} relL2={rel:.1e}")


if __name__ == "__main__":
    main()
