"""One sample at a time, whole machine, decode + augment + VAE encode -> throughput.

Same shape as ``/mnt/disk0/jiangzs/share/bench_aug_dataset.py``: one process, no DataLoader workers,
``chunk_length=32``, ``fps=4.0``, ``resolution="480"``, ``max_episodes=4``, the same three-view stack and the same
Compose, and ``torch.set_num_threads()`` set to the whole machine's core count. What this adds is what that script
leaves out -- the VAE encode, so the reported number is a complete per-sample pipeline rate rather than the cost of
one transform.

Two consequences of the single-sample layout worth knowing before reading the number:

* one sample can only be encoded on one card, so the node's other seven NPUs are idle by construction;
* nothing overlaps: the decode, the transform and the encode run back to back on one sample.

    python bench_pipeline_single.py --checkpoint ... --data ... --augment --samples 12
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset
from examples.worldmodel.cosmos3.packing import normalize_latent

LATENT_MIB = 48 * 9 * 33 * 40 * 2 / 2**20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--threads", type=int, default=0, help="torch threads; 0 keeps the default (all cores).")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"host {torch.cuda.is_available()!s} threads {torch.get_num_threads()} augment {args.augment} "
          f"device {args.device}", flush=True)

    dataset = DroidActionPolicyDataset(
        args.data,
        chunk_length=32,
        fps=4.0,
        resolution="480",
        augment=args.augment,
        max_episodes=4,
    )
    print(f"windows {len(dataset)}", flush=True)

    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint / "vae", torch_dtype=torch.bfloat16)
    vae = vae.eval().requires_grad_(False).to(args.device)
    vae_dtype = next(vae.parameters()).dtype

    def encode(sample):
        video = sample["video"].unsqueeze(0).to(args.device, vae_dtype)
        with torch.no_grad():
            out = vae.encode(video)
            dist = getattr(out, "latent_dist", None)
            latent = dist.mode() if dist is not None else out
            return normalize_latent(latent, vae)

    # Warm both halves: the decoder opens an mp4, the VAE compiles its first kernels.
    encode(dataset[0])
    torch.npu.synchronize()

    data_times, vae_times, totals = [], [], []
    for index in range(args.samples):
        started = time.perf_counter()
        sample = dataset[index % len(dataset)]
        torch.npu.synchronize()
        after_data = time.perf_counter()
        encode(sample)
        torch.npu.synchronize()
        finished = time.perf_counter()
        data_times.append(after_data - started)
        vae_times.append(finished - after_data)
        totals.append(finished - started)
        print(f"sample {index:3d}  data {data_times[-1]:6.3f}s  vae {vae_times[-1]:6.3f}s  "
              f"total {totals[-1]:6.3f}s", flush=True)

    mean_total = statistics.mean(totals)
    median_total = statistics.median(totals)
    print()
    print(f"mean   data {statistics.mean(data_times):.3f}s  vae {statistics.mean(vae_times):.3f}s  "
          f"total {mean_total:.3f}s")
    print(f"median data {statistics.median(data_times):.3f}s  vae {statistics.median(vae_times):.3f}s  "
          f"total {median_total:.3f}s")
    print(f"throughput {1 / median_total:.3f} samples/s on one card  "
          f"({LATENT_MIB / median_total:.2f} MiB/s of latents)")
    print(f"data share {statistics.median(data_times) / median_total * 100:.1f}%  "
          f"vae share {statistics.median(vae_times) / median_total * 100:.1f}%")
    sys.stdout.flush()
    # Workers are absent here, but torch_npu's teardown can still wedge after device work; exit explicitly.
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
