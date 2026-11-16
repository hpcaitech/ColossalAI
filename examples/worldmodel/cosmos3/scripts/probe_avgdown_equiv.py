"""Is the strided AvgDown3D shortcut bit-identical to diffusers' permute version, and how long does each take?

The encoder calls AvgDown3D four times per pass and each call materialises the whole tensor through an 8-D
permute just to average the factor_s x factor_s block. The strided version computes the same average without the
materialisation. Gate: identical latents (or a stated relL2), then the encode wall.

    python scripts/probe_avgdown_equiv.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, iters=8):
    for _ in range(3):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    import types

    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import AvgDown3D

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder
    from examples.worldmodel.cosmos3.vae_norm_kernel import enable_kernel_rmsnorm

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    vae.requires_grad_(False)
    original = AvgDown3D.forward
    enable_fast_encoder(vae)
    enable_kernel_rmsnorm(vae)

    # Per-module check first: the encoder-level comparison cannot tell which shortcut drifted.
    for index, module in enumerate(m for m in vae.modules() if isinstance(m, AvgDown3D)):
        for temporal in (1, 3, 4, 6):
            shape = (2, module.in_channels, temporal, 16, 16)
            probe = torch.randn(shape, device=device, dtype=torch.bfloat16)
            fast = module(probe)
            reference = types.MethodType(original, module)(probe)
            same = torch.equal(fast, reference)
            rel = float((fast.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
            print(
                f"[module {index}] ft={module.factor_t} fs={module.factor_s} T={temporal}: "
                f"bit_equal={same} relL2={rel:.1e}",
                flush=True,
            )

    x = torch.randn(args.batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        fast = vae.encode(x).latent_dist.mode()
        ms_fast = bench(lambda: vae.encode(x).latent_dist.mode())

        # enable_fast_encoder patches each instance, so restoring the class method is not enough
        for module in vae.modules():
            if isinstance(module, AvgDown3D):
                module.forward = types.MethodType(original, module)
        reference = vae.encode(x).latent_dist.mode()
        ms_ref = bench(lambda: vae.encode(x).latent_dist.mode())

    equal = torch.equal(fast, reference)
    diff = (fast.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
    maxabs = (fast.float() - reference.float()).abs().max()
    print(f"[avgdown] latent bit_equal={equal} relL2={float(diff):.3e} maxabs={float(maxabs):.3e}")
    print(
        f"[avgdown] encode {ms_ref:.1f} ms (permute) -> {ms_fast:.1f} ms (strided), "
        f"{100 * (1 - ms_fast / ms_ref):.1f}% saved"
    )


if __name__ == "__main__":
    main()
