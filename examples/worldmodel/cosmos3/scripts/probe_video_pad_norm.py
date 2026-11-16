"""Correctness and cost of the uint8 -> normalised bf16 TileLang front end on real shapes.

    python scripts/probe_video_pad_norm.py --batch 8 --pad 16 --iters 5

Checks the kernel against ``torch_reference`` (bit equality, max abs, relL2) and times three paths for one
encode-batch: the host float32 pad+normalise + H2D (today), the uint8 H2D + kernel + device pad (new), and the
kernel alone. The sample shape is DROID's: frames=33, content=540x640, pad_h=4 (canvas 544), pad_w configurable.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=8, help="Encode-batch samples.")
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--pad-h", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main():
    import torch_npu  # noqa: F401

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.worldmodel.cosmos3.video_pad_norm import (
        normalize_uint8,
        pad_normalize,
        pad_normalize_torch,
        torch_reference,
    )

    args = parse_args()
    torch.npu.set_device(0)
    shape = (args.batch, 3, 33, 540, 640)
    host_uint8 = torch.randint(0, 256, shape, dtype=torch.uint8)
    device_uint8 = host_uint8.to(args.device)
    torch.npu.synchronize()

    actual = pad_normalize(device_uint8, args.pad_h, args.pad)
    reference = torch_reference(host_uint8, args.pad_h, args.pad)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    diff = (actual_cpu.float() - reference.float()).abs()
    rel = float(diff.norm() / reference.float().norm())
    print(f"[shape] {shape} -> {tuple(actual.shape)} ({actual.dtype})")
    actual_bf16 = actual_cpu.to(torch.bfloat16)
    reference_bf16 = reference.to(torch.bfloat16)
    print(f"[correct] fp32_max_abs={float(diff.max()):.3e} relL2={rel:.3e} "
          f"bf16_equal={torch.equal(actual_bf16, reference_bf16)}")

    def timed(fn, iters):
        fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(iters):
            torch.npu.synchronize()
            started = time.perf_counter()
            fn()
            torch.npu.synchronize()
            samples.append(time.perf_counter() - started)
        return statistics.median(samples)

    kernel_only = timed(lambda: normalize_uint8(device_uint8), args.iters)
    new_path = timed(lambda: pad_normalize(device_uint8, args.pad_h, args.pad), args.iters)
    torch_path = timed(lambda: pad_normalize_torch(device_uint8, args.pad_h, args.pad), args.iters)
    torch_out = pad_normalize_torch(device_uint8, args.pad_h, args.pad).cpu()
    # Today's path: per-sample (4-D) host float32 normalise + pad, then H2D of the padded float tensor and a bf16
    # cast on device. The 5-D CPU pad rejects the 4-element padding, which is why the dataset pads per sample.
    host_float = torch.stack(
        [torch.nn.functional.pad(sample.float() / 127.5 - 1.0, (0, args.pad, 0, args.pad_h), mode="reflect")
         for sample in host_uint8]
    )

    def old_path():
        return host_float.to(args.device, torch.bfloat16)

    old = timed(old_path, args.iters)
    print(f"[time] normalize_only(kernel-or-fallback) {kernel_only * 1e3:.2f} ms | "
          f"kernel+pad {new_path * 1e3:.2f} ms | torch_device_ops {torch_path * 1e3:.2f} ms | "
          f"host_float+H2D {old * 1e3:.2f} ms")
    print(f"[torch] bf16_equal_vs_host={torch.equal(torch_out, reference)}")
    print(f"[bytes] H2D uint8 {host_uint8.numel() / 2**20:.1f} MiB vs float32 {host_float.numel() * 4 / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
