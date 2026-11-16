"""Does the Conv3D path re-convert its frozen weight to fractal format on every call?

The bf16 encode profile has ``aclnnConvolution_TransData`` 801 times for 278 convolutions -- just under three per
call, which is the signature of input + weight + output reformatting (NCDHW <-> NDC1HWC0). The weight never changes,
so one of those three is repeated work that ``torch_npu.npu_format_cast`` can remove by storing the weight in the
format the convolution consumes.

Gate: the cast only moves the same values, so the output must be bit-identical; then the question is kernels per
call and wall time.

    python scripts/probe_conv_transdata.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import tempfile
from collections import Counter

import torch
import torch_npu

# ACL format enum: 29 FRACTAL_NZ, 30 NCDHW (the default), 31 NDC1HWC0 (the 3D-conv activation layout),
# 33 FRACTAL_Z_3D (the 3D-conv *weight* layout). torch_npu's own Module.cast_weight pre-casts Linear weights to NZ
# for the same reason; for Conv3d the value to try is 33.
FORMATS = {
    "ND": 2,
    "NC1HWC0": 3,
    "FRACTAL_Z": 4,
    "NDHWC": 27,
    "FRACTAL_NZ": 29,
    "NDC1HWC0": 31,
    "FRACTAL_Z_3D": 33,
}


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


def kernel_counts(work, tag, iters=4):
    """Profile `iters` calls and return the kernel-name counts of the last one."""
    out = os.path.join(tempfile.gettempdir(), f"conv_probe_{tag}")
    os.makedirs(out, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=iters, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out),
    ) as prof:
        for _ in range(iters + 2):
            work()
            prof.step()
    torch.npu.synchronize()
    found = glob.glob(os.path.join(out, "**", "kernel_details.csv"), recursive=True)
    if not found:
        return Counter()
    rows = list(csv.DictReader(open(sorted(found)[-1], newline="")))
    steps = Counter(row.get("Step Id", "") for row in rows)
    keep = [step for step, count in steps.most_common(1)]
    if keep and keep[0]:
        rows = [row for row in rows if row.get("Step Id") == keep[0]]
    return Counter(row["Name"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--temporal", type=int, default=4)
    parser.add_argument("--spatial", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    vae.requires_grad_(False)

    convs = [m for m in vae.modules() if isinstance(m, WanCausalConv3d)]
    convs.sort(key=lambda m: -m.weight.numel())
    cases = convs[:2] + [convs[len(convs) // 2]]
    print(f"[conv] {len(convs)} causal convolutions; probe cases {[tuple(c.weight.shape) for c in cases]}")

    for index, conv in enumerate(cases):
        cout, cin, kt, kh, kw = conv.weight.shape
        shape = (args.batch, cin, args.temporal, args.spatial, args.spatial)
        x = torch.randn(shape, device=device, dtype=torch.bfloat16)

        def run(w=conv.weight):
            return torch.nn.functional.conv3d(
                x, w, conv.bias, conv.stride, (0, conv._padding[2], conv._padding[0]), conv.dilation, conv.groups
            )

        reference = run()
        ms_base = bench(run, args.iters)
        counts_base = Counter() if args.skip_profile else kernel_counts(run, f"{index}_base")
        transdata_base = sum(count for name, count in counts_base.items() if "TransData" in name)
        print(f"\n[case {index}] weight {tuple(conv.weight.shape)} input {shape}")
        print(f"  baseline {ms_base:.3f} ms  kernels {sum(counts_base.values())}  transdata {transdata_base}")
        for name, count in counts_base.most_common(6):
            print(f"    {count:>3}  {name}")

        for label, value in FORMATS.items():
            try:
                weight = torch_npu.npu_format_cast(conv.weight, value)
            except Exception as error:  # the format is not supported for this tensor/op combination
                print(f"  cast {label:<11} unsupported: {type(error).__name__}: {str(error)[:90]}")
                continue
            try:
                out = run(weight)
            except Exception as error:
                print(f"  cast {label:<11} conv rejected: {type(error).__name__}: {str(error)[:90]}")
                continue
            equal = torch.equal(out, reference)
            rel = float((out.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))
            ms = bench(lambda w=weight: run(w), args.iters)
            counts = Counter() if args.skip_profile else kernel_counts(lambda w=weight: run(w), f"{index}_{label}")
            transdata = sum(count for name, count in counts.items() if "TransData" in name)
            print(f"  cast {label:<11} {ms:7.3f} ms ({100 * (1 - ms / ms_base):+6.1f}%)  bit_equal={equal} "
                  f"relL2={rel:.1e}  kernels {sum(counts.values())}  transdata {transdata}")


if __name__ == "__main__":
    main()
