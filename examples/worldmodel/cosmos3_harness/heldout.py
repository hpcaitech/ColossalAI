"""Held-out splitting shared by the GPU and Ascend Cosmos 3 runs.

The split is made before packing, cropping and augmentation, and it is grouped by source trajectory. A window-level
split would put adjacent frames of the same episode on both sides and make the held-out loss look better than the
model is; grouping first is the difference between measuring generalisation and measuring memorisation.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = ["split_by_group", "write_split_manifest", "load_split_manifest", "split_fingerprint"]


def split_by_group(
    records: Sequence[Mapping],
    *,
    group_key: str = "trajectory_id",
    sample_key: str = "sample_id",
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[str], List[str], Dict]:
    """Split sample ids by group so one source trajectory never lands on both sides.

    Args:
        records (Sequence[Mapping]): One mapping per sample, carrying a sample id and a group id.
        group_key (str, optional): Field that identifies the source trajectory/video. Defaults to ``"trajectory_id"``.
        sample_key (str, optional): Field that identifies the sample. Defaults to ``"sample_id"``.
        validation_fraction (float, optional): Fraction of groups held out. Defaults to 0.1.
        seed (int, optional): Seed of the deterministic group shuffle. Defaults to 0.

    Returns:
        Tuple[List[str], List[str], Dict]: Training ids, validation ids and a manifest describing the split.

    Raises:
        ValueError: If a record is missing a key, a duplicate sample id exists, or the fraction is not in (0, 1).
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}.")

    groups: Dict[str, List[str]] = {}
    seen_samples = set()
    for record in records:
        if sample_key not in record or group_key not in record:
            raise ValueError(f"Every record needs {sample_key!r} and {group_key!r}, got {sorted(record)}.")
        sample_id = str(record[sample_key])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample id {sample_id!r}; the split would silently drop one of them.")
        seen_samples.add(sample_id)
        groups.setdefault(str(record[group_key]), []).append(sample_id)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target = validation_fraction * len(seen_samples)
    validation_groups = []
    validation_count = 0
    for group_id in group_ids:
        if validation_count >= target:
            break
        validation_groups.append(group_id)
        validation_count += len(groups[group_id])

    validation_set = set(validation_groups)
    train_ids = [
        sample_id for group_id in group_ids if group_id not in validation_set for sample_id in groups[group_id]
    ]
    validation_ids = [
        sample_id for group_id in group_ids if group_id in validation_set for sample_id in groups[group_id]
    ]
    if not train_ids or not validation_ids:
        raise ValueError(
            f"The split put {len(train_ids)} samples in train and {len(validation_ids)} in validation. Use more "
            f"groups or a larger validation_fraction."
        )

    manifest = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "group_key": group_key,
        "sample_key": sample_key,
        "train_samples": train_ids,
        "validation_samples": validation_ids,
        "train_groups": [group_id for group_id in group_ids if group_id not in validation_set],
        "validation_groups": validation_groups,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "overlap": sorted(set(train_ids) & set(validation_ids)),
    }
    return train_ids, validation_ids, manifest


def split_fingerprint(manifest: Mapping) -> str:
    """Return a stable digest of the sample assignment, so both runs can prove they read the same split."""
    payload = json.dumps(
        {
            "train": manifest["train_samples"],
            "validation": manifest["validation_samples"],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_manifest(manifest: Mapping, path: str | Path) -> Path:
    """Write the split manifest as JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["fingerprint"] = split_fingerprint(manifest)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_split_manifest(path: str | Path) -> Dict:
    """Load a split manifest and verify its stored fingerprint."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = split_fingerprint(manifest)
    if manifest.get("fingerprint") != expected:
        raise ValueError(
            f"{path} carries fingerprint {manifest.get('fingerprint')!r}, but its sample lists hash to {expected}. "
            f"The manifest was edited after it was written; regenerate it instead of changing one side."
        )
    return manifest


def iter_grouped_records(path: str | Path, *, sample_key: str = "sample_id", group_key: str = "trajectory_id"):
    """Yield records from a JSONL file, requiring the two keys the grouped split needs."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if sample_key not in record or group_key not in record:
                raise ValueError(f"{path}:{line_number} is missing {sample_key!r} or {group_key!r}.")
            yield record
