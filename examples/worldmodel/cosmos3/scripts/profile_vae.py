"""Profile the frozen VAE encoder on the device the training loop runs it on.

The step is VAE-dominated at the recipe's window, so the launch configuration has to be chosen from these numbers:
how one ``encode`` scales with batch, frames and canvas, how much device memory each shape costs, and how much of
the cost is the host-to-device copy rather than the network itself.

Run it under the same launcher the training uses, so all eight cards are profiled at once::

    python -m torch.distributed.run --nproc_per_node=8 scripts/profile_vae.py \
        --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
        --out /home/niuzifan/logs/vae-profile

Each rank writes ``rank<N>.json`` and prints its own lines; the numbers are per device, so the spread across ranks
is what says whether eight concurrent encoders contend for anything.

``--compile`` is off by default: every new shape is a fresh graph, and a graph on this device is minutes, not
seconds. Turn it on for one shape at a time when that is the question.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

# frames, height, width of the real training input. DROID's three cameras are all 640x360; the recipe composes the
# wrist view on top and the two shoulder views side by side below at half width (2 x 320 wide, 180 tall), which
# gives a 640x540 image with aspect 0.844. `closest_canvas` maps that to the "4,3" entry of the 480p table, i.e.
# 736x544. Getting this wrong is easy and it changes the numbers by ~10%: 832x480 is the 16:9 entry, not this one.
RECIPE = (33, 544, 736)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", default="1,2,4,8", help="Batch sizes for the batch sweep.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also time torch.compile(vae._encode, dynamic=False) on the recipe shape. One graph per shape.",
    )
    parser.add_argument("--decode-batch", type=int, default=1, help="Also time one decode; 0 skips it.")
    return parser.parse_args()


def build_vae(checkpoint: Path, device):
    """The same VAE settings the training loop uses; HF32 has to stay off (see ``train_sft.build_vae``)."""
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=torch.float32).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae.to(device)


def measure(encode, video, warmup: int, reps: int):
    """Seconds per call and peak GiB, synchronised on both sides so the numbers are the device's."""
    with torch.no_grad():
        for _ in range(warmup):
            out = encode(video)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(reps):
            out = encode(video)
        torch.npu.synchronize()
        seconds = (time.perf_counter() - start) / reps
    return seconds, torch.npu.max_memory_allocated() / 2**30, tuple(out.latent_dist.mode().shape)


def one_row(vae, mode: str, batch: int, frames: int, height: int, width: int, args):
    """Encode a random video of this shape and return one result row; an out-of-memory shape is a row too."""
    row = {
        "mode": mode,
        "batch": batch,
        "frames": frames,
        "height": height,
        "width": width,
        "latent": [],
        "copy_s": 0.0,
        "encode_s": None,
        "per_sample_s": None,
        "peak_gib": 0.0,
    }
    try:
        video = torch.randn(batch, 3, frames, height, width, dtype=torch.float32)
        start = time.perf_counter()
        video = video.to("npu")
        torch.npu.synchronize()
        row["copy_s"] = round(time.perf_counter() - start, 4)
        seconds, peak, latent = measure(vae.encode, video, args.warmup, args.reps)
        row.update(
            latent=list(latent),
            encode_s=round(seconds, 4),
            per_sample_s=round(seconds / batch, 4),
            peak_gib=round(peak, 2),
        )
        del video
    except Exception as error:  # a shape that does not fit is a result, not a reason to abandon the sweep
        row["error"] = f"{type(error).__name__}: {error}".splitlines()[0][:120]
    finally:
        torch.npu.empty_cache()
    return row


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    vae = build_vae(args.checkpoint, f"npu:{local_rank}")
    recipe_frames, recipe_h, recipe_w = RECIPE

    rows = []
    for batch in [int(x) for x in args.batches.split(",")]:
        rows.append(one_row(vae, "batch", batch, recipe_frames, recipe_h, recipe_w, args))
    for frames in (9, 17, 33, 65):
        rows.append(one_row(vae, "frames", 2, frames, recipe_h, recipe_w, args))
    for height, width in ((192, 320), (480, 832), (544, 736), (640, 640)):
        rows.append(one_row(vae, "canvas", 2, recipe_frames, height, width, args))
    if args.compile:
        vae._encode = torch.compile(vae._encode, dynamic=False)
        rows.append(one_row(vae, "compiled", 2, recipe_frames, recipe_h, recipe_w, args))

    if args.decode_batch:
        shape = (args.decode_batch, 48, 9, recipe_h // 16, recipe_w // 16)
        latent = torch.randn(*shape, dtype=torch.float32).to("npu")
        with torch.no_grad():
            vae.decode(latent)
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            vae.decode(latent)
            torch.npu.synchronize()
            decode_seconds = time.perf_counter() - start
        rows.append(
            {
                "mode": "decode",
                "batch": args.decode_batch,
                "frames": recipe_frames,
                "height": recipe_h,
                "width": recipe_w,
                "latent": list(shape),
                "copy_s": 0.0,
                "encode_s": round(decode_seconds, 4),
                "per_sample_s": round(decode_seconds / args.decode_batch, 4),
                "peak_gib": round(torch.npu.max_memory_allocated() / 2**30, 2),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"rank{rank}.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        head = f"[r{rank}] {row['mode']:<8} B={row['batch']} T={row['frames']} {row['height']}x{row['width']}"
        if row["encode_s"] is None:
            print(f"{head} FAILED {row.get('error', '')}", flush=True)
            continue
        print(
            f"{head} latent={tuple(row['latent'])} copy={row['copy_s']:.3f}s encode={row['encode_s']:.3f}s "
            f"per_sample={row['per_sample_s']:.3f}s peak={row['peak_gib']:.2f}GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
