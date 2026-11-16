"""Does the fused norm kernel actually run inside the training VAE, and how fast is the encode there?

The three-arm A/B showed the norm-on arm dying with SIGSEGV on all ranks while the norm-off arm was fine, and
the norm-on+uint8 arm finishing barely faster than baseline. Both point at the training-side integration
rather than the kernel itself (the kernel is bit-identical in ``probe_norm_kernel.py``). This probe rebuilds
the VAE exactly the way ``train_sft.py`` does, counts kernel hits vs reference fallbacks per shape, times one
batch-8 encode, and prints where the time went.

    python scripts/probe_norm_kernel_training.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
import types
from collections import Counter

import torch


def bench(fn, iters=10):
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
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--preallocate-gib",
        type=float,
        default=0.0,
        help="Hold this much HBM before timing, to mimic the training process's resident model/optimizer state.",
    )
    args = parser.parse_args()

    from examples.worldmodel.cosmos3.train_sft import build_vae

    device = torch.device("npu:0")
    vae = build_vae(
        __import__("pathlib").Path(args.checkpoint),
        torch.bfloat16,
        0,
        False,
        True,  # fast encoder, as bench_online_step.sh passes
        None,
        norm_kernel=True,
    ).to(device)
    print(f"[vae] kernel norm modules {getattr(vae, 'kernel_rmsnorm_modules', None)}")

    hits = Counter()
    fallbacks = Counter()

    def instrument(module):
        kernel_forward = module.forward

        def forward(self, x, *rest, **kw):
            from examples.worldmodel.cosmos3.vae_norm_kernel import _plan

            b, c = int(x.shape[0]), int(x.shape[1])
            s = x.numel() // (b * c)
            ok = (
                x.device.type == "npu"
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and self.channel_first
                and not isinstance(self.bias, torch.Tensor)
                and x.dim() >= 3
                and _plan(c, s) is not None
            )
            if ok:
                hits[(c, s)] += 1
            else:
                fallbacks[(c, s, str(x.dtype), bool(x.is_contiguous()))] += 1
            return kernel_forward(x, *rest, **kw)

        module.forward = types.MethodType(forward, module)

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    for module in vae.modules():
        if isinstance(module, WanRMS_norm):
            instrument(module)

    ballast = []
    if args.preallocate_gib:
        ballast = [torch.empty(int(args.preallocate_gib * 2**30 // 4), device=device, dtype=torch.float32)]
        print(f"[ballast] holding {args.preallocate_gib:.0f} GiB")

    x = torch.randn(args.batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        vae.encode(x)
    print("[kernel hits]", dict(hits))
    print("[fallbacks]", dict(fallbacks))
    ms = bench(lambda: vae.encode(x), args.iters)
    print(f"[encode] {ms:.1f} ms per batch-{args.batch} call")
    # The tail micro-batch (32 samples encoded 8 at a time leaves 4) is its own compiled kernel; time it too.
    x4 = x[:4].contiguous()
    with torch.no_grad():
        vae.encode(x4)
    print(f"[encode] {bench(lambda: vae.encode(x4), args.iters):.1f} ms per batch-4 call (tail)")
    print("[kernel hits after tail]", dict(sorted(hits.items())))


if __name__ == "__main__":
    main()
