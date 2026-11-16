"""Does a larger VAE encode batch still produce the same latents, and is it faster?

The trainer encodes 32 samples per rank 8 at a time (``--vae-encode-batch 8``), so the VAE phase pays every per-call
cost -- the 2-3 TransData per convolution, the feat-cache cat, the kernel launches -- four times. Batching more
samples per call amortises those, but only pays if the latents stay identical: a conv's reduction order can depend on
its batch dimension through tiling.

    python scripts/probe_vae_encode_batch.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, iters=3):
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
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=656)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder
    from examples.worldmodel.cosmos3.vae_norm_kernel import enable_kernel_rmsnorm

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    vae.requires_grad_(False)
    enable_fast_encoder(vae)
    enable_kernel_rmsnorm(vae)

    torch.manual_seed(0)
    video = torch.randn(args.samples, 3, args.frames, args.height, args.width, device=device,
                        dtype=torch.bfloat16)

    def encode(batch: int):
        latents = []
        with torch.no_grad():
            for start in range(0, args.samples, batch):
                chunk = video[start:start + batch]
                latents.append(vae.encode(chunk).latent_dist.mode())
        return torch.cat(latents, dim=0)

    reference = encode(8)
    print(f"[vae] {args.samples} samples, encode batch sweep")
    print(f"{'batch':>6}{'ms':>10}{'s/sample':>11}{'bit_equal':>11}{'relL2':>10}{'peak GiB':>10}")
    for batch in (8, 12, 16, args.samples):
        if batch > args.samples:
            continue
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        out = encode(batch)
        peak = torch.npu.max_memory_allocated() / 2**30
        ms = bench(lambda b=batch: encode(b))
        equal = torch.equal(out, reference)
        rel = float((out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
        print(f"{batch:>6}{ms:>10.1f}{ms / args.samples / 1000:>11.3f}{str(equal):>11}{rel:>10.1e}{peak:>10.1f}")
        del out
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
