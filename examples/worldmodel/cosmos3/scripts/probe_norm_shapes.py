"""Record every ``WanRMS_norm`` call's shape/dtype and time the reference chain at those shapes.

The fused-norm kernel must be written against the real shapes and dtypes, and its payoff is bounded by what the
reference chain (``F.normalize(dim=1) * gamma * scale``) costs at exactly those shapes. This probe prints both.

    python scripts/probe_norm_shapes.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
import json
import types
from collections import Counter

import torch
import torch.nn.functional as F

RECIPE = (33, 544, 656)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    enable_fast_encoder(vae)
    print(
        "[config]",
        json.dumps({k: v for k, v in dict(vae.config).items() if isinstance(v, (int, float, str))}, default=str)[:600],
    )

    calls = Counter()

    def make_forward(module):
        def forward(self, x):
            calls[(tuple(x.shape), str(x.dtype), int(self.gamma.shape[0]))] += 1
            return F.normalize(x, dim=1) * self.gamma * self.scale

        return forward

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    for module in vae.modules():
        if isinstance(module, WanRMS_norm):
            module.forward = types.MethodType(make_forward(module), module)

    x = torch.randn(args.batch, 3, *RECIPE, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        vae.encode(x)
    print("[calls]")
    for (shape, dtype, channels), count in calls.most_common():
        print(f"  {shape} {dtype} C={channels} x{count}")

    print("[reference timing] F.normalize(dim=1) * gamma * scale, per shape")
    total = 0.0
    for (shape, dtype, channels), count in calls.most_common():
        dt = getattr(torch, dtype.split(".")[-1])
        t = torch.randn(*shape, device=device, dtype=dt)
        gamma = torch.ones(channels, 1, 1, 1, device=device, dtype=dt)
        scale = float(channels) ** 0.5
        for _ in range(5):
            y = F.normalize(t, dim=1) * gamma * scale
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            y = F.normalize(t, dim=1) * gamma * scale
        end.record()
        torch.npu.synchronize()
        ms = start.elapsed_time(end) / args.iters
        numel = t.numel()
        gbps = 2 * numel * t.element_size() / (ms * 1e6)
        # every call in the encoder runs at this shape; the per-encode cost is what the step pays
        print(
            f"  {shape} {dtype}: {ms:.3f} ms  ~{gbps:.0f} GB/s ideal-traffic  x{count}/encode  => {ms * count:.1f} ms/encode"
        )
        total += ms * count
    print(f"[reference] summed over one encode: {total:.1f} ms")


if __name__ == "__main__":
    main()
