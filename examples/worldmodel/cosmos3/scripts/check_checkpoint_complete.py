"""Is this node's copy of a run's checkpoint resumable?  Reads one node's run directory and says why not.

``save_checkpoint`` writes the model, optimizer and public training state on rank zero's node, and each
rank's ``rng-<rank>.pt`` / ``master-<rank>.pt`` on the node that rank runs on.  With ``--out`` on a shared
filesystem every node sees one complete generation.  With ``--out`` on each node's local disk, rank zero's
node publishes ``checkpoint-<id>`` (with ``complete.json`` and ``latest.json``) while the other nodes keep
only their own rank files in ``.incomplete-<id>``.

``--resume`` handles both: it reads the generation name and manifest from rank zero, every rank reports
whether its node is complete, and a node that lacks the shared files receives them over the training
interconnect before validation.  This script applies the same rules from one node's point of view:

    python scripts/check_checkpoint_complete.py /mnt/disk0/jiangzs/nzf/bench_online/runs/<tag> --world 48

Exit status is 0 when this node can take part in a resume (either it already holds the generation, or it
holds the rank files it wrote and rank zero's node can serve the rest).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RANK_FILE_PREFIXES = ("rng-", "master-")


def is_rank_file(name: str) -> bool:
    return name.startswith(RANK_FILE_PREFIXES)


def rank_of(name: str) -> int | None:
    stem = Path(name).name
    _, _, suffix = stem.partition("-")
    try:
        return int(suffix.split(".")[0])
    except (TypeError, ValueError):
        return None


def published_generation(root: Path) -> Path | None:
    latest = root / "latest.json"
    if latest.is_file():
        return root / json.loads(latest.read_text())["generation"]
    generations = sorted(p for p in root.glob("checkpoint-*") if (p / "complete.json").is_file())
    return generations[-1] if generations else None


def newest_staging(root: Path) -> Path | None:
    staging = sorted(p for p in root.glob(".incomplete-*") if p.is_dir())
    return staging[-1] if staging else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="the run directory that holds metrics/ and checkpoints/")
    parser.add_argument("--world", type=int, default=48, help="world size the checkpoint must cover")
    parser.add_argument("--no-hash", action="store_true", help="skip the SHA256 pass over the shared files")
    args = parser.parse_args()

    root = args.run_dir / "checkpoints"
    generation = published_generation(root)
    if generation is None:
        staging = newest_staging(root)
        if staging is None:
            print(f"[ckpt] FAIL no checkpoint generation or staging directory under {root}")
            return 1
        held = sorted(p.name for p in staging.iterdir() if p.is_file())
        ranks = sorted({rank_of(name) for name in held if is_rank_file(name)} - {None})
        print(f"[ckpt] this node never published; newest staging is {staging.name}")
        print(f"[ckpt] it holds {len(ranks)} rank file(s): {ranks[:8]}{' ...' if len(ranks) > 8 else ''}")
        print(
            "[ckpt] RESUME-SAFE from this node: --resume fetches model/optimizer/training_state from "
            "rank 0's node and keeps these rank files"
        )
        return 0

    print(f"[ckpt] generation {generation.name}")
    complete = generation / "complete.json"
    if not complete.is_file():
        print("[ckpt] FAIL no complete.json next to this generation")
        return 1
    manifest = json.loads(complete.read_text())
    files = manifest.get("files") or {}
    manifest.get("sizes") or {}
    print(f"[ckpt] manifest lists {len(files)} files, {manifest.get('total_bytes', 0) / 2**30:.1f} GiB")

    problems: list[str] = []
    for prefix in ("model/", "optimizer/"):
        if not any(name.startswith(prefix) for name in files):
            problems.append(f"manifest has no {prefix} files")
    if "training_state.pt" not in files:
        problems.append("manifest has no training_state.pt")

    missing_rank_entries = [rank for rank in range(args.world) if f"rng-{rank:05d}.pt" not in files]
    if missing_rank_entries:
        shown = ", ".join(str(rank) for rank in missing_rank_entries[:6])
        print(
            f"[ckpt] WARN the manifest records no RNG for {len(missing_rank_entries)} of {args.world} ranks "
            f"({shown}{' ...' if len(missing_rank_entries) > 6 else ''}): this generation predates the "
            "per-rank digest gather, so only a shared filesystem can resume it"
        )

    on_disk = {path.name for path in generation.iterdir()} if generation.is_dir() else set()
    held = sorted(name for name in on_disk if is_rank_file(name))
    print(f"[ckpt] this node holds {len(held)} rank file(s): {', '.join(held[:6])}{' ...' if len(held) > 6 else ''}")

    shared = [name for name in files if not is_rank_file(name)]
    absent = [name for name in shared if not (generation / name).is_file()]
    if absent:
        print(f"[ckpt] this node is missing {len(absent)} of {len(shared)} shared files, e.g. {absent[:3]}")
    if not absent and not args.no_hash:
        import hashlib

        bad = []
        for name in shared:
            path = generation / name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != files[name]:
                bad.append(name)
        if bad:
            problems.append(f"{len(bad)} shared file(s) fail their checksum, e.g. {bad[:3]}")

    if problems:
        for problem in problems:
            print(f"[ckpt] FAIL {problem}")
        return 1
    if absent:
        print("[ckpt] RESUME-SAFE from this node: --resume fetches the shared files from rank 0's node")
    else:
        print("[ckpt] RESUME-SAFE on this node: the generation is complete here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
