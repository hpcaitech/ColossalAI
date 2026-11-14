"""Pre-extract VAE latents for the DROID roots, with decoding in dataloader workers.

The first version decoded inside the process that also ran the VAE, which made it decode-bound (1.86 s/sample
against 0.47 s of actual encoding). This version puts the decode in ``num_workers`` worker processes and keeps
only the encode on the device, which is how the training loop feeds the VAE.

Each rank takes a contiguous block of the window list rather than a stride slice: the windows are ordered
episode by episode with the start offset increasing inside an episode, so a block keeps each rank reading its
mp4 files forward instead of seeking to a different file for every sample. It encodes ``--encode-batch`` samples
per VAE call and writes ``shard-<rank>-<n>.pt`` files. Latents are stored already normalised and cropped --
exactly what ``pack_one`` consumes -- in bf16. Windows are enumerated the same way the trainer enumerates them,
so the cache covers every window a training run of the same subset would visit.

    python precompute_latents.py --checkpoint ... --data .../Cosmos3-DROID --out .../latents --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset, collate
from examples.worldmodel.cosmos3.packing import normalize_latent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="Parent of the success/ and failure/ LeRobot roots."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, help="Per root; omit for every episode.")
    parser.add_argument("--limit", type=int, help="Stop after this many windows across all ranks.")
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--rank", type=int, help="Global rank; defaults to $RANK.")
    parser.add_argument("--world", type=int, help="Global world size; defaults to $WORLD_SIZE.")
    parser.add_argument("--device", type=int, help="Device index in this process; use 0 with one visible card.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Run the recipe's crop/resize/color-jitter before the VAE. Each cached latent then corresponds to "
        "one draw, so the cache is single-use: it cannot be revisited across epochs the way an unaugmented one "
        "can, and the number of latents needed equals the number of training visits.",
    )
    parser.add_argument(
        "--shard-mode",
        default="block",
        choices=("block", "stride"),
        help="How ranks slice the window list. 'block' keeps each rank on a run of consecutive windows so the "
        "mp4 files are read forward; 'stride' is the old interleaved split, kept for comparison.",
    )
    return parser.parse_args()


def build_vae(checkpoint: Path, dtype: torch.dtype):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval().requires_grad_(False)
    return vae.to(f"npu:{torch.npu.current_device()}")


def main():
    args = parse_args()
    rank = int(args.rank) if args.rank is not None else int(os.environ.get("RANK", "0"))
    world = int(args.world) if args.world is not None else int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = args.device if args.device is not None else int(os.environ.get("LOCAL_RANK", str(rank % 8)))
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"
    torch.multiprocessing.set_sharing_strategy(os.environ.get("COSMOS3_SHM_STRATEGY", "file_system"))

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = DroidActionPolicyDataset(
        args.data,
        chunk_length=32,
        fps=15.0,
        resolution="480",
        # With --augment each latent is tied to one draw, so the cache cannot be replayed across epochs; without
        # it the same window maps to the same latent every time and the cache is reusable. The flag is what
        # makes the extraction rate comparable to the training rate, so it has to be explicit rather than assumed.
        augment=args.augment,
        max_episodes=args.max_episodes,
    )
    total = len(dataset)
    if args.limit is not None:
        total = min(total, args.limit)
    if args.shard_mode == "block":
        # The window list is ordered episode by episode with the start offset increasing inside an episode, so a
        # contiguous block lets each rank read its mp4 files forward after a single seek. A stride slice instead
        # makes every single sample jump to another file, which is what made the first run decode-bound.
        per_rank = (total + world - 1) // world
        first = rank * per_rank
        mine = list(range(first, min(first + per_rank, total)))
    else:
        mine = list(range(rank, total, world))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, mine),
        batch_size=args.encode_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=False,
        multiprocessing_context="spawn" if args.num_workers else None,
    )

    vae = build_vae(args.checkpoint, getattr(torch, args.dtype))
    vae_dtype = next(vae.parameters()).dtype
    spatial_factor = None

    index, buffer, buffer_shapes = {}, {}, {}
    shard_id = 0
    started = time.perf_counter()
    done = 0

    def flush():
        nonlocal shard_id, buffer, buffer_shapes
        if not buffer:
            return
        name = f"shard-{rank:03d}-{shard_id:04d}.pt"
        torch.save(buffer, args.out / name)
        for key, (shape, nh, nw) in buffer_shapes.items():
            index[key] = {
                "shard": name,
                "shape": list(shape),
                "keep_hw": [nh, nw],
                "canvas_hw": buffer[key]["canvas_hw"],
                "aspect": buffer[key]["aspect"],
            }
        shard_id += 1
        buffer, buffer_shapes = {}, {}

    # Iterate a known number of batches and never ask for one past the end: on exhaustion the iterator shuts its
    # workers down, and a spawned worker that only ever touched host memory hangs there inside torch_npu's
    # teardown. The count is known up front, so the shutdown is simply not triggered.
    batches = (len(mine) + args.encode_batch - 1) // args.encode_batch
    iterator = iter(loader)
    for batch_index in range(batches):
        batch = next(iterator)
        video = batch["video"].to(device, vae_dtype)
        with torch.no_grad():
            latent = vae.encode(video)
            dist = getattr(latent, "latent_dist", None)
            latent = dist.mode() if dist is not None else latent
            latent = normalize_latent(latent, vae).float()
        if spatial_factor is None:
            spatial_factor = int(video.shape[-2] // latent.shape[-2])
        for offset in range(latent.shape[0]):
            content_h, content_w = (int(v) for v in batch["content_hw"][offset].tolist())
            keep_h = max(content_h // spatial_factor, 1)
            keep_w = max(content_w // spatial_factor, 1)
            piece = latent[offset : offset + 1, :, :, :keep_h, :keep_w].contiguous().to(getattr(torch, args.dtype))
            sample_id = batch["sample_id"][offset]
            # The geometry travels with the latent: a reader that skips decoding has no other way to reproduce
            # the prompt's canvas and the rectified-flow shift key.
            buffer[sample_id] = {
                "latent": piece.cpu(),
                "canvas_hw": [int(v) for v in batch["canvas_hw"][offset].tolist()],
                "content_hw": [content_h, content_w],
                "aspect": batch["aspect_key"][offset],
            }
            buffer_shapes[sample_id] = (tuple(piece.shape), keep_h, keep_w)
        done += latent.shape[0]
        if len(buffer) >= args.shard_size:
            flush()
        if batch_index % 4 == 0 or batch_index == batches - 1:
            print(f"[rank {rank}] batches={batch_index + 1}/{batches} samples={done} at={time.perf_counter() - started:.1f}s", flush=True)
    flush()

    # One index file per rank. A rank only knows the samples it encoded, so a single index.json written by rank 0
    # would silently describe 1/world of the cache -- with more than one node the ranks writing into this
    # directory are neither contiguous nor owned by one host, and a reader has no way to notice the gap.
    (args.out / f"index-rank-{rank:03d}.json").write_text(json.dumps(index, indent=0))
    elapsed = time.perf_counter() - started
    print(
        f"[rank {rank}] samples={done} seconds={elapsed:.1f} per_sample={elapsed / max(done, 1):.3f}s "
        f"samples_per_s={done / elapsed:.3f}",
        flush=True,
    )
    # Skip normal interpreter teardown: the spawned workers only touched host memory and can wedge inside
    # torch_npu's shutdown, which would leave the wrapper script waiting on a finished job.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
