"""Throughput of the CPU half of the data pipeline, in the 8-card layout of one node.

This is the number that decides whether the VAE can ever be fed: everything in ``__getitem__`` runs in DataLoader
worker processes on the host CPU, so the ceiling is set by cores and by how the work is split, not by the NPU.
No device is touched here -- the measurement is deliberately VAE-free -- but the process layout matches training:
one process per card, ``--workers`` decode workers behind each, the same chunk length and canvas.

Configuration follows ``/mnt/disk0/jiangzs/share/bench_aug_dataset.py`` where it affects the transforms
(``chunk_length=32``, ``resolution="480"``, the three-view stack, the same Compose) and differs where that script's
settings only make sense for a per-call latency probe: the recipe's ``fps=15.0`` is used instead of 4.0, and enough
episodes/windows are indexed to sustain a slope rather than six calls.

    python bench_pipeline_cpu.py --data .../success --rank 0 --world 8 --workers 4 --augment
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    # Repository root, i.e. the parent of `examples/`: this file lives four levels down at
    # examples/worldmodel/cosmos3/scripts/, so parents[4] is the directory that makes `examples.*` importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset, collate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--limit", type=int, default=640, help="Windows across all ranks.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy(os.environ.get("COSMOS3_SHM_STRATEGY", "file_system"))
    dataset = DroidActionPolicyDataset(
        args.data,
        chunk_length=32,
        fps=args.fps,
        resolution="480",
        augment=args.augment,
        max_episodes=args.max_episodes,
    )
    total = min(len(dataset), args.limit)
    # Same contiguous-block split the extractor uses, so both cover comparable windows.
    per_rank = (total + args.world - 1) // args.world
    first = args.rank * per_rank
    mine = list(range(first, min(first + per_rank, total)))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, mine),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=False,
        multiprocessing_context="spawn" if args.workers else None,
    )

    batches = (len(mine) + args.batch - 1) // args.batch
    started = time.perf_counter()
    done = 0
    iterator = iter(loader)
    for index in range(batches):
        batch = next(iterator)
        # Touch one element so the batch cannot be optimised away, without paying for a real reduction: the cost
        # being measured is decode+transform+IPC, not arithmetic on the result. Five indices because the tensor is
        # [B, C, T, H, W] -- stopping at four leaves a row and float() refuses it.
        float(batch["video"][0, 0, 0, 0, 0])
        done += batch["video"].shape[0]
        if index % 4 == 0 or index == batches - 1:
            print(f"[rank {args.rank}] batches={index + 1}/{batches} samples={done} "
                  f"at={time.perf_counter() - started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    print(f"[rank {args.rank}] samples={done} seconds={elapsed:.1f} per_sample={elapsed / max(done, 1):.3f}s "
          f"samples_per_s={done / elapsed:.3f}", flush=True)
    sys.stdout.flush()
    # Workers only touched host memory; skip teardown rather than risk a wedge in torch_npu's shutdown path.
    os._exit(0)


if __name__ == "__main__":
    main()
