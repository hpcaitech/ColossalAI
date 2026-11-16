"""How much of the canvas padding does the VAE need? Scan right-padding widths and check the latent.

The dataset fits the three-view composite into a 544x736 canvas (4:3) and reflection-pads 96 columns on the right;
the latent is then cropped back to the 33x40 content region, so the encode of those 96 columns is thrown away
(~13% of the VAE work). The padding is not inert: the mid-block spatial attention sees the full canvas, so
removing it changes the content latents too.

This script encodes the same real batch at several input widths -- content only (640) plus a right pad of
0/16/32/64/96 columns, always a multiple of 16 -- and reports, against the full 736 recipe canvas:

* the wall time per sample (the padding cost),
* the cropped-latent relL2 (the semantic cost).

    python scripts/bench_vae_canvas.py --checkpoint ... --data .../success --batch 8 --pads 0,16,32,64,96

Gate: pick the largest saving whose relL2 stays within the bf16 cast noise (about 4e-3). If none does, the canvas
width is part of the recipe and must not be changed.
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pads", default="0,16,32,64,96", help="Right padding columns beyond the content width.")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def real_batch(data: Path, batch: int, max_episodes: int):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset, collate

    dataset = DroidActionPolicyDataset(
        data, chunk_length=32, resolution="480", augment=False, max_episodes=max_episodes
    )
    canvas = dataset[0]["canvas_hw"].tolist()
    content = dataset[0]["content_hw"].tolist()
    chosen, index = [dataset[0]], 1
    while len(chosen) < batch:
        candidate = dataset[index % len(dataset)]
        if candidate["canvas_hw"].tolist() == canvas and candidate["content_hw"].tolist() == content:
            chosen.append(candidate)
        index += 1
        if index > 8 * len(dataset):
            raise SystemExit("no single canvas/content pair covers the batch")
    video = collate(chosen)["video"]
    print(f"[data] shape={tuple(video.shape)} canvas={canvas} content={content}", flush=True)
    return video, canvas, content


def encode(vae, video, dtype):
    with torch.no_grad():
        out = vae.encode(video.to("npu:0", dtype))
        dist = getattr(out, "latent_dist", None)
        return dist.mode() if dist is not None else out


def timed(vae, video, dtype, warmup, iters):
    encode(vae, video, dtype)
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        started = time.perf_counter()
        encode(vae, video, dtype)
        torch.npu.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float(((a - b).norm() / b.norm()).item())


def main():
    import torch_npu
    from diffusers import AutoencoderKLWan

    args = parse_args()
    torch.npu.set_device(0)
    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    vae = AutoencoderKLWan.from_pretrained(args.checkpoint / "vae", torch_dtype=torch.bfloat16).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae = enable_fast_encoder(vae).to("npu:0")
    dtype = next(vae.parameters()).dtype

    video, canvas, content = real_batch(args.data, args.batch, args.max_episodes)
    content_h, content_w = int(content[0]), int(content[1])
    pads = [int(value) for value in args.pads.split(",")]
    if max(pads) + content_w > video.shape[-1]:
        raise SystemExit(f"pad exceeds the canvas: content {content_w} + pad {max(pads)} > {video.shape[-1]}")

    results = []
    reference = None
    for pad in pads:
        width = content_w + pad
        item = video[:, :, :, :, :width]
        torch.npu.reset_peak_memory_stats()
        latent = encode(vae, item, dtype)
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() / 2**30
        spatial = item.shape[-2] // latent.shape[-2]
        keep_h, keep_w = content_h // spatial, content_w // spatial
        cropped = latent[:, :, :, :keep_h, :keep_w]
        if pad == max(pads):
            reference = cropped
        wall = timed(vae, item, dtype, args.warmup, args.iters)
        results.append(
            {
                "pad": pad,
                "width": width,
                "wall_s": wall,
                "s_per_sample": wall / args.batch,
                "peak_gib": peak,
                "rel_l2": None,  # filled below against the widest reference
                "cropped": cropped,
            }
        )
        print(
            f"pad {pad:3d} width {width:4d}  {wall:.3f} s  {wall / args.batch:.3f} s/sample  " f"peak {peak:6.2f} GiB",
            flush=True,
        )

    for row in results:
        row["rel_l2"] = rel_l2(row.pop("cropped"), reference)
    base = next(row for row in results if row["pad"] == max(pads))
    print(f"\nreference: pad {base['pad']} ({base['width']} columns)")
    print(f"{'pad':>4} {'width':>6} {'s/sample':>9} {'speedup':>8} {'relL2':>10}")
    for row in results:
        print(
            f"{row['pad']:>4} {row['width']:>6} {row['s_per_sample']:9.3f} "
            f"{base['wall_s'] / row['wall_s']:8.3f} {row['rel_l2']:10.2e}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([{k: v for k, v in row.items()} for row in results], indent=2))
    # os._exit skips the interpreter's stdout flush; without this the summary table above never reaches the log.
    sys.stdout.flush()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
