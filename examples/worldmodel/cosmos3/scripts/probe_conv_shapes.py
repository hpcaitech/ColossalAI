"""What shapes do the encoder's convolutions actually have, and what MFU do they run at?

The production profile puts ``Conv3DV2`` at 42% of the encoder's device time (1.246 s of 2.95 s, 233 calls),
plus 0.275 s of convolution-internal ``TransData``. Before writing any conv kernel, this probe records every
``Conv3d`` call's input/output channels, kernel, stride and spatial shape, and computes the FLOPs each one
carries. Combined with the stage timings from ``profile_vae_deep.py`` it says whether the cube is starved
(a kernel worth writing) or the conv is already near peak (not worth it).

    python scripts/probe_conv_shapes.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--timed", action="store_true", help="Also time each unique shape (syncs per call).")
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    enable_fast_encoder(vae)

    calls = Counter()

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            weight = module.weight
            stride = module.stride
            groups = module.groups
            key = (tuple(x.shape), tuple(weight.shape), tuple(stride), groups)
            calls[key] += 1
        return hook

    handles = [module.register_forward_hook(make_hook(name))
               for name, module in vae.named_modules() if isinstance(module, nn.Conv3d)]

    x = torch.randn(args.batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        vae.encode(x)

    total_flops = 0.0
    print(f"{'input':26s} {'weight':22s} {'stride':10s} {'calls':>5s} {'GFLOP/call':>11s} {'GFLOP/encode':>12s}")
    for (shape, wshape, stride, groups), count in calls.most_common():
        out_c, in_c_per_group, kt, kh, kw = wshape
        spatial = shape[2] * shape[3] * shape[4]
        flops = 2.0 * shape[0] * spatial * out_c * (in_c_per_group * kt * kh * kw)
        total_flops += flops * count
        print(f"{str(shape):26s} {str(wshape):22s} {str(stride):10s} {count:5d} "
              f"{flops / 1e9:11.2f} {flops * count / 1e9:12.1f}")
    for handle in handles:
        handle.remove()
    print(f"[conv] total {total_flops / 1e12:.2f} TFLOP per encode at batch {args.batch}")

    if args.timed:
        print("[timed] per unique shape (sync per call; for MFU attribution only)")
        for (shape, wshape, stride, groups), count in calls.most_common(8):
            channels = wshape[0]
            sample = torch.randn(*shape, device=device, dtype=torch.bfloat16)
            conv = nn.Conv3d(
                shape[1], channels, wshape[2:], stride=stride, padding=tuple(k // 2 for k in wshape[2:]),
                groups=groups, bias=False,
            ).to(device, torch.bfloat16)
            with torch.no_grad():
                for _ in range(3):
                    conv(sample)
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                conv(sample)
                end.record()
                torch.npu.synchronize()
            ms = start.elapsed_time(end)
            flops = 2.0 * shape[0] * shape[2] * shape[3] * shape[4] * channels * (
                wshape[1] * wshape[2] * wshape[3] * wshape[4])
            print(f"  {str(shape):26s} {str(wshape):22s} {ms:8.3f} ms  {flops / ms / 1e9:7.1f} TFLOPS  x{count}")
            del sample, conv
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
