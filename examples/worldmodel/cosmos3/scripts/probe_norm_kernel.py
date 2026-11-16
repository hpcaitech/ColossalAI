"""Validate and time the fused channel-first RMSNorm kernel against the reference chain.

Two gates, cheapest first:

* per shape -- the module's own forward versus the kernel on the twelve shapes the production encoder calls
  (batch 8), reporting relative L2 / max-abs error and the time of each;
* end to end -- the whole frozen VAE encoder with and without the patch, reporting latent relL2 and encode
  wall time.

    python scripts/probe_norm_kernel.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
import types

import torch

SHAPES = (
    ((8, 640, 1, 34, 41), 90),  # 81 + 9 calls; the deepest stage, launch-bound in the reference
    ((8, 160, 4, 272, 328), 32),
    ((8, 320, 4, 136, 164), 24),
    ((8, 640, 2, 68, 82), 24),
    ((8, 160, 4, 136, 164), 8),
    ((8, 160, 1, 272, 328), 4),
    ((8, 320, 2, 68, 82), 8),
    ((8, 320, 1, 136, 164), 3),
    ((8, 640, 1, 68, 82), 3),
    ((8, 160, 1, 136, 164), 1),
    ((8, 320, 1, 68, 82), 1),
)


def bench(fn, iters=20):
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


def per_shape(iters):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    import examples.worldmodel.cosmos3.vae_norm_kernel as nk

    device = torch.device("npu:0")
    total_ref = total_kernel = 0.0
    for shape, calls in SHAPES:
        c = shape[1]
        torch.manual_seed(0)
        x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        module = WanRMS_norm(c, images=False).to(device).to(torch.bfloat16)  # production gamma: (C, 1, 1, 1)
        with torch.no_grad():
            module.gamma.copy_(torch.empty_like(module.gamma).uniform_(0.5, 1.5))
        ref = types.MethodType(nk._reference_forward, module)
        kernel = types.MethodType(nk._kernel_forward, module)
        y_ref = ref(x)
        y_kernel = kernel(x)
        diff = (y_kernel.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-12)
        maxabs = (y_kernel.float() - y_ref.float()).abs().max()
        ms_ref = bench(lambda: ref(x), iters)
        ms_kernel = bench(lambda: kernel(x), iters)
        total_ref += ms_ref * calls
        total_kernel += ms_kernel * calls
        print(
            f"  {str(shape):26s} ref {ms_ref:8.3f} ms  kernel {ms_kernel:8.3f} ms  "
            f"relL2 {diff.item():9.2e}  maxabs {maxabs.item():8.2e}  x{calls}",
            flush=True,
        )
        del x, y_ref, y_kernel, module
        torch.npu.empty_cache()
    print(
        f"  [per-encode estimate] {total_ref:.1f} ms -> {total_kernel:.1f} ms "
        f"({100.0 * (1 - total_kernel / total_ref):.1f}% saved)"
    )


def end_to_end(checkpoint, iters, batch):
    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder
    from examples.worldmodel.cosmos3.vae_norm_kernel import enable_kernel_rmsnorm

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    enable_fast_encoder(vae)
    for parameter in vae.parameters():
        parameter.requires_grad = False

    torch.manual_seed(0)
    x = torch.randn(batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent_ref = vae.encode(x).latent_dist.mode()
        ms_ref = bench(lambda: vae.encode(x).latent_dist.mode(), iters)
        enable_kernel_rmsnorm(vae)
        latent_kernel = vae.encode(x).latent_dist.mode()
        ms_kernel = bench(lambda: vae.encode(x).latent_dist.mode(), iters)
    diff = (latent_kernel.float() - latent_ref.float()).norm() / latent_ref.float().norm()
    maxabs = (latent_kernel.float() - latent_ref.float()).abs().max()
    print(
        f"[end-to-end] encode wall {ms_ref:.1f} ms -> {ms_kernel:.1f} ms "
        f"({100.0 * (1 - ms_kernel / ms_ref):.1f}% saved)"
    )
    print(f"[end-to-end] latent relL2 {diff.item():.3e}  maxabs {maxabs.item():.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, help="Also run the end-to-end VAE comparison.")
    args = parser.parse_args()
    per_shape(args.iters)
    if args.checkpoint:
        end_to_end(args.checkpoint, max(5, args.iters // 4), args.batch)


if __name__ == "__main__":
    main()
