"""Turn a training generation into a warm-start checkpoint directory, and prove the weights survive.

The 48-rank run's checkpoint only ever published on rank 0 (F23): its ``model/`` holds the full DiT state
dict, which is all a warm start needs.  ``build_model`` reads ``<checkpoint>/transformer`` through
``Cosmos3OmniTransformer.from_pretrained``, so the trained shards have to be renamed into the diffusers
convention and given a matching index next to the base checkpoint's ``config.json``.  Everything else (VAE,
tokenizer, configs) is reused from the base checkpoint through hard links, so nothing is copied twice.

    # what would change
    python scripts/prepare_warm_start.py --generation <run>/checkpoints/checkpoint-<id> \
        --base /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID --check

    # build it
    python scripts/prepare_warm_start.py ... --assemble /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-SFT100

``--check`` fails (exit 1) when the trained shards do not cover every key of the base transformer, or when a
shape differs: a warm start that silently drops parameters is worse than restarting from the base.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open

INDEX = "diffusion_pytorch_model.safetensors.index.json"


def read_index(directory: Path) -> tuple[dict[str, str], Path]:
    index = directory / INDEX
    if not index.is_file():
        candidates = sorted(directory.glob("*.index.json"))
        if not candidates:
            raise SystemExit(f"no *.index.json under {directory}")
        index = candidates[0]
    payload = json.loads(index.read_text())
    return payload["weight_map"], index


def read_headers(directory: Path, weight_map: dict[str, str]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """key -> (dtype, shape), read from the safetensors headers only."""
    by_file: dict[str, list[str]] = {}
    for key, name in weight_map.items():
        by_file.setdefault(name, []).append(key)
    headers: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, keys in by_file.items():
        with safe_open(directory / name, framework="pt", device="cpu") as handle:
            for key in keys:
                slice_ = handle.get_slice(key)
                headers[key] = (slice_.get_dtype(), tuple(slice_.get_shape()))
    return headers


def check(generation: Path, base: Path) -> bool:
    trained_dir = generation / "model"
    trained_map, _ = read_index(trained_dir)
    base_map, _ = read_index(base / "transformer")
    trained = read_headers(trained_dir, trained_map)
    reference = read_headers(base / "transformer", base_map)

    missing = sorted(set(reference) - set(trained))
    extra = sorted(set(trained) - set(reference))
    shape_mismatch = sorted(key for key in set(reference) & set(trained) if reference[key][1] != trained[key][1])
    dtype_mismatch = sorted(key for key in set(reference) & set(trained) if reference[key][0] != trained[key][0])
    print(f"[warm] base transformer   {len(reference):5d} tensors from {len(set(base_map.values()))} shard(s)")
    print(f"[warm] trained state dict {len(trained):5d} tensors from {len(set(trained_map.values()))} shard(s)")
    for label, keys in (
        ("missing in trained", missing),
        ("extra in trained", extra),
        ("shape mismatch", shape_mismatch),
        ("dtype mismatch", dtype_mismatch),
    ):
        print(f"[warm] {label:<20} {len(keys)}" + (f": {', '.join(keys[:6])}" if keys else ""))
    dtype_counts: dict[str, int] = {}
    for dtype, _shape in trained.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    print(f"[warm] trained dtypes {dtype_counts}")
    # A dtype difference is not a defect on its own: the trainer builds the whole model in bf16, so a tensor the
    # base stores in fp32 is saved back as the bf16 the model actually ran with.  What matters is whether the
    # *values* agree, which --compare-values checks.
    ok = not missing and not shape_mismatch
    print("[warm] COVERS THE BASE TRANSFORMER" if ok else "[warm] DOES NOT COVER THE BASE TRANSFORMER")
    return ok


def compare_values(generation: Path, base: Path) -> bool:
    """Every tensor must equal the base cast to bf16, except the ones the run actually trained."""
    import torch

    from examples.worldmodel.cosmos3.train_sft import TRAINABLE_KEYS

    trained_dir = generation / "model"
    trained_map, _ = read_index(trained_dir)
    base_map, _ = read_index(base / "transformer")
    if set(trained_map) != set(base_map):
        print("[warm] cannot compare values: key sets differ")
        return False

    def load(directory: Path, weight_map: dict[str, str], key: str):
        with safe_open(directory / weight_map[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    differing_frozen, differing_trainable, compared = [], [], 0
    for key in sorted(trained_map):
        trained = load(trained_dir, trained_map, key)
        reference = load(base / "transformer", base_map, key).to(torch.bfloat16)
        compared += 1
        if torch.equal(trained.to(torch.bfloat16), reference):
            continue
        (differing_trainable if any(tag in key for tag in TRAINABLE_KEYS) else differing_frozen).append(key)
        if len(differing_frozen) <= 5:
            delta = (trained.float() - reference.float()).abs().max().item() if differing_frozen else 0.0
            if differing_frozen and differing_frozen[-1] == key:
                print(f"[warm] frozen tensor changed: {key}  max|Δ| {delta:.3e}")
    print(
        f"[warm] compared {compared} tensors: {len(differing_trainable)} trained tensors changed (expected), "
        f"{len(differing_frozen)} frozen tensors changed"
    )
    if differing_trainable[:6]:
        print(f"[warm]   e.g. trained: {', '.join(differing_trainable[:6])}")
    if differing_frozen[:6]:
        print(f"[warm]   e.g. FROZEN:  {', '.join(differing_frozen[:6])}")
    ok = not differing_frozen
    print("[warm] FROZEN WEIGHTS PRESERVED" if ok else "[warm] FROZEN WEIGHTS CHANGED")
    return ok


def hardlink_tree(source: Path, target: Path) -> None:
    """Recreate ``source`` under ``target`` with hard links (same filesystem) or copies (otherwise)."""
    for root, _dirs, files in os.walk(source):
        relative = Path(root).relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(root) / name
            dst = target / relative / name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def assemble(generation: Path, base: Path, out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it or pick another name")
    hardlink_tree(base, out)
    transformer = out / "transformer"
    # The base transformer is replaced wholesale: drop its links, keep config.json.
    for stale in transformer.glob("diffusion_pytorch_model-*"):
        stale.unlink()
    for stale in transformer.glob("diffusion_pytorch_model.safetensors.index.json"):
        stale.unlink()

    trained_dir = generation / "model"
    trained_map, _ = read_index(trained_dir)
    shards = sorted(set(trained_map.values()))
    renamed = {}
    for ordinal, name in enumerate(shards, start=1):
        target_name = f"diffusion_pytorch_model-{ordinal:05d}-of-{len(shards):05d}.safetensors"
        shutil.copyfile(trained_dir / name, transformer / target_name)
        renamed[name] = target_name
    weight_map = {key: renamed[name] for key, name in trained_map.items()}
    (transformer / INDEX).write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum((trained_dir / n).stat().st_size for n in shards)},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"[warm] wrote {out}: {len(shards)} shard(s), {len(weight_map)} tensors")
    print(f"[warm] transformer dir: {sorted(p.name for p in transformer.iterdir())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation", type=Path, required=True, help="checkpoint-<id> directory")
    parser.add_argument("--base", type=Path, required=True, help="base checkpoint root (holds transformer/, vae/)")
    parser.add_argument("--check", action="store_true", help="compare key sets, shapes and dtypes")
    parser.add_argument(
        "--compare-values",
        action="store_true",
        help="bitwise audit: every frozen tensor must equal the base cast to bf16",
    )
    parser.add_argument("--assemble", type=Path, help="build the warm-start directory here")
    args = parser.parse_args()
    ok = check(args.generation, args.base) if args.check or args.assemble else True
    if args.compare_values:
        ok = compare_values(args.generation, args.base) and ok
    if args.assemble:
        if not ok:
            raise SystemExit("refusing to assemble: the trained state dict does not cover the base transformer")
        assemble(args.generation, args.base, args.assemble)
    if not args.check and not args.assemble:
        parser.error("pass --check and/or --assemble")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
