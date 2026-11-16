"""Delete old checkpoint generations, one node at a time.  Dry run unless ``--apply`` is given.

Every node now publishes its own generation (format 3), so a 32-rank run writes one ~17 GiB generation *per node*
per save.  At ``--save-every 100`` over 10k steps that is 1.7 TB per node, which a 7 TB disk survives but should not
be asked to.  Nothing in the training loop deletes generations: a resume needs the newest one, an audit may want an
older one, and an automatic delete is the wrong failure mode.  This script is the manual step.

    python scripts/prune_checkpoints.py /mnt/disk0/.../runs/<tag> --keep 3            # show what would go
    python scripts/prune_checkpoints.py /mnt/disk0/.../runs/<tag> --keep 3 --apply    # delete

It refuses to delete the generation ``latest.json`` points at, the newest generation, anything that is not a
completed ``checkpoint-*`` directory (staging ``.incomplete-*`` directories are reported, never removed), or a
directory whose ``complete.json`` is missing.  Run it on *each* node: the generations are node-local.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def generations(root: Path):
    rows = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir() or not (path / "complete.json").is_file():
            continue
        try:
            manifest = json.loads((path / "complete.json").read_text())
        except (OSError, ValueError):
            manifest = {}
        total = manifest.get("total_bytes") or 0
        rows.append((path.stat().st_mtime, path, manifest.get("step"), total))
    rows.sort(key=lambda row: (row[0], row[1].name))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the run directory that holds checkpoints/")
    parser.add_argument("--keep", type=int, default=3, help="how many of the newest generations to keep")
    parser.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = parser.parse_args()
    if args.keep < 1:
        raise SystemExit("--keep must be at least 1")

    root = args.run_dir / "checkpoints"
    if not root.is_dir():
        raise SystemExit(f"no checkpoints directory under {args.run_dir}")
    rows = generations(root)
    if not rows:
        print(f"[prune] {root}: no completed generations")
        return 0
    pointer = None
    latest = root / "latest.json"
    if latest.is_file():
        try:
            pointer = json.loads(latest.read_text()).get("generation")
        except (OSError, ValueError):
            pointer = None
    protected = {rows[-1][1].name} | ({pointer} if pointer else set())
    keep = {path.name for _, path, _, _ in rows[-args.keep :]}
    doomed = [row for row in rows if row[1].name not in keep and row[1].name not in protected]

    freed = sum(row[3] for row in doomed)
    print(f"[prune] {root}: {len(rows)} generations, keeping {len(keep)} (+{len(protected - keep)} protected)")
    for mtime, path, step, total in rows:
        tag = "keep" if path.name in keep else ("latest" if path.name in protected else "delete")
        print(f"  {tag:6s} step {str(step):>7s}  {total / 2**30:6.1f} GiB  {path.name}")
    staging = sorted(root.glob(".incomplete-*"))
    for path in staging:
        print(f"  staging (untouched) {path.name}")
    if not doomed:
        print("[prune] nothing to delete")
        return 0
    print(f"[prune] would free {freed / 2**30:.1f} GiB by deleting {len(doomed)} generations")
    if not args.apply:
        print("[prune] dry run; pass --apply to delete")
        return 0
    for _, path, _, _ in doomed:
        shutil.rmtree(path)
        print(f"[prune] deleted {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
