"""Sweep the VAE encode configuration on one card with real DROID windows.

The training loop's VAE phase is the largest single block of the online step, so every configuration that could
change it has to be measured under the same shape the recipe uses: 33 frames, 544x736 canvas, bf16/fp16, the
fast encoder and the chunked call window. This script is the standard entry for that sweep:

    python scripts/bench_vae_sweep.py \
        --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
        --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success \
        --batch 8 --encode-batches 4,8 --chunks 0,4,8,12 --dtypes bfloat16,float16

Output is one JSON line per configuration, plus a human-readable table. ``--chunks 0`` keeps diffusers' native
loop (prime + 4 frames per call), which is the reference for both latency and the latent.

Timing discipline: one warmup encode per configuration (kernel selection is shape-dependent), then ``--iters``
timed encodes. Each timed unit encodes the whole ``--batch`` through the same micro-batching the trainer uses.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8, help="Samples per timed encode unit.")
    parser.add_argument("--encode-batches", default="4,8", help="Comma-separated micro-batch sizes.")
    parser.add_argument(
        "--chunks", default="0,4,8,12", help="Comma-separated pixel frames per encoder call; 0 = native."
    )
    parser.add_argument("--dtypes", default="bfloat16", help="Comma-separated torch dtypes.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--out", type=Path, help="JSON-lines result file.")
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def build_vae(checkpoint: Path, dtype: torch.dtype):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae.to("npu:0")


def real_batch(data: Path, batch: int, max_episodes: int):
    """One batch of real windows that share a canvas, so stacking is valid."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset, collate

    dataset = DroidActionPolicyDataset(
        data, chunk_length=32, resolution="480", augment=False, max_episodes=max_episodes
    )
    canvas = dataset[0]["canvas_hw"].tolist()
    chosen, index = [dataset[0]], 1
    while len(chosen) < batch:
        candidate = dataset[index % len(dataset)]
        if candidate["canvas_hw"].tolist() == canvas:
            chosen.append(candidate)
        index += 1
        if index > 8 * len(dataset):
            raise SystemExit("no single canvas covers the batch")
    video = collate(chosen)["video"]
    print(f"[data] windows={len(dataset)} shape={tuple(video.shape)} canvas={canvas}", flush=True)
    return video


def peak_gib():
    return torch.npu.max_memory_allocated() / 2**30


def main():
    import torch_npu

    args = parse_args()
    torch.npu.set_device(0)
    from examples.worldmodel.cosmos3.vae_chunked import enable_chunked_encode
    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    video = real_batch(args.data, args.batch, args.max_episodes)
    dtypes = [getattr(torch, name.strip()) for name in args.dtypes.split(",")]
    encode_batches = [int(value) for value in args.encode_batches.split(",")]
    chunks = [int(value) for value in args.chunks.split(",")]

    print(
        f"{'dtype':>9} {'chunk':>5} {'eb':>3} {'wall s':>8} {'s/sample':>9} {'samples/s':>10} " f"{'peak GiB':>9}",
        flush=True,
    )
    records = []
    for dtype in dtypes:
        vae = build_vae(args.checkpoint, dtype)
        enable_fast_encoder(vae)
        native_encode = vae._encode
        video_d = video.to(dtype).to(args.device)

        for chunk in chunks:
            vae._encode = native_encode
            if chunk:
                enable_chunked_encode(vae, chunk)

            for encode_batch in encode_batches:
                if encode_batch > args.batch:
                    continue
                torch.npu.reset_peak_memory_stats()

                def run():
                    parts = []
                    with torch.no_grad():
                        for start in range(0, args.batch, encode_batch):
                            out = vae.encode(video_d[start : start + encode_batch])
                            dist = getattr(out, "latent_dist", None)
                            parts.append(dist.mode() if dist is not None else out)
                    return parts

                run()  # warm the shape-specific kernels and the allocator
                torch.npu.synchronize()
                times = []
                for _ in range(args.iters):
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    run()
                    torch.npu.synchronize()
                    times.append(time.perf_counter() - started)
                wall = statistics.median(times)
                record = {
                    "dtype": str(dtype).replace("torch.", ""),
                    "chunk": chunk,
                    "encode_batch": encode_batch,
                    "batch": args.batch,
                    "wall_s": wall,
                    "s_per_sample": wall / args.batch,
                    "samples_per_s": args.batch / wall,
                    "peak_gib": peak_gib(),
                }
                records.append(record)
                print(
                    f"{record['dtype']:>9} {chunk:>5} {encode_batch:>3} {wall:8.3f} "
                    f"{record['s_per_sample']:9.3f} {record['samples_per_s']:10.2f} "
                    f"{record['peak_gib']:9.2f}",
                    flush=True,
                )

        del vae, video_d
        torch.npu.empty_cache()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"[out] {args.out}", flush=True)
    # torch_npu teardown can hang after device work; exit explicitly like the other bench scripts.
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
