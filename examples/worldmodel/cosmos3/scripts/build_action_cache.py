"""Pre-extract each data parquet's action/state columns into a compact per-file float32 array.

Why: ``DroidActionPolicyDataset._read_actions`` reads a whole parquet column set the first time a worker touches a
file. With 32 DataLoader workers, a sampler step that crosses a file boundary makes every worker do that cold read
at once; measured on node6, one such step waited 127 s on the data loader (82% of the step). The columns are tiny
(16 floats + episode index per row), so a per-file ``.npy`` turns the read into an O(1) slice that the page cache
shares across workers, and removes the parquet decode from the hot path entirely.

Layout: ``<out>/<split>/data/chunk-XXX/file-YYY.npy``, float32 ``[rows, 17]``:
columns 0:7 joint action, 7 gripper action, 8:15 joint state, 15 gripper state, 16 episode index.
The dataset mirrors this path from ``Window.root``, so a split's cache is self-describing.

    python scripts/build_action_cache.py \
        --roots /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/failure \
        --out /mnt/disk0/jiangzs/nzf/action_cache
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINT_ACTION = "action.joint_position"
GRIPPER_ACTION = "action.gripper_position"
STATE_JOINT = "observation.state.joint_positions"
STATE_GRIPPER = "observation.state.gripper_position"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, required=True, help="LeRobot roots (success, failure).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-files", type=int, help="Only convert the first N files per root (smoke tests).")
    parser.add_argument("--force", action="store_true", help="Rewrite files that already exist.")
    return parser.parse_args()


def column_2d(table, name, width: int) -> np.ndarray:
    """One column as ``[rows, width]`` float32, accepting scalar or length-1 encodings."""
    values = table.column(name).to_pylist()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[1] != width:
        raise ValueError(f"{name}: expected width {width}, got {array.shape}")
    return array


def convert(path: Path, destination: Path) -> tuple[int, float]:
    started = time.perf_counter()
    table = pq.read_table(path, columns=[JOINT_ACTION, GRIPPER_ACTION, STATE_JOINT, STATE_GRIPPER, "episode_index"])
    rows = table.num_rows
    out = np.empty((rows, 17), dtype=np.float32)
    out[:, 0:7] = column_2d(table, JOINT_ACTION, 7)
    out[:, 7:8] = column_2d(table, GRIPPER_ACTION, 1)
    out[:, 8:15] = column_2d(table, STATE_JOINT, 7)
    out[:, 15:16] = column_2d(table, STATE_GRIPPER, 1)
    out[:, 16] = np.asarray(table.column("episode_index").to_pylist(), dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, out)
    return rows, time.perf_counter() - started


def main():
    args = parse_args()
    for root in args.roots:
        root = root.resolve()
        split = root.name
        files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        if args.limit_files:
            files = files[: args.limit_files]
        print(f"[{split}] {len(files)} parquet file(s) -> {args.out / split}", flush=True)
        total_rows = 0
        total_seconds = 0.0
        for index, path in enumerate(files):
            chunk = path.parent.name
            destination = args.out / split / "data" / chunk / f"{path.stem}.npy"
            if destination.exists() and not args.force:
                print(f"  [{index + 1}/{len(files)}] {path.name}: exists, skipped", flush=True)
                continue
            rows, seconds = convert(path, destination)
            total_rows += rows
            total_seconds += seconds
            if index % 10 == 0 or index + 1 == len(files):
                print(
                    f"  [{index + 1}/{len(files)}] {path.name}: {rows} rows in {seconds:.2f} s "
                    f"({rows / seconds:.0f} rows/s)",
                    flush=True,
                )
        print(f"[{split}] rows={total_rows} seconds={total_seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
