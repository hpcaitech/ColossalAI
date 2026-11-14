"""Collective, immutable checkpoint generations with rank-local random state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _accelerator():
    from colossalai.accelerator import get_accelerator

    return get_accelerator()


def save_checkpoint(directory, model, optimizer, state, *, booster=None):
    """All ranks participate; only rank zero publishes a complete generation.

    Model/optimizer writers are owned by Booster's checkpoint IO. The completion
    manifest covers every shard and RNG file. A failed generation never replaces latest.
    """
    rank, world = _rank_world()
    root = Path(directory)
    identifier = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(identifier, src=0)
    staging = root / (".incomplete-" + identifier[0])
    staging.mkdir(parents=True, exist_ok=True)
    booster = booster or model._colossal_booster
    booster.save_model(model, str(staging / "model"), shard=True, use_safetensors=True)
    booster.save_optimizer(optimizer, str(staging / "optimizer"), shard=True)
    accelerator = _accelerator()
    rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "accelerator": accelerator.name,
    }
    if accelerator.name != "cpu":
        rng["device"] = accelerator.get_rng_state_all()
    torch.save(rng, staging / f"rng-{rank:05d}.pt")
    if rank == 0:
        torch.save({**state, "world_size": world, "format_version": 1}, staging / "training_state.pt")
    _barrier()
    error = [None]
    generation = "checkpoint-" + identifier[0]
    if rank == 0:
        try:
            files = {}
            sizes = {}
            total_bytes = 0
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
                name = str(path.relative_to(staging))
                files[name] = file_digest(path)
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                sizes[name] = size
                total_bytes += size
            (staging / "complete.json").write_text(
                json.dumps(
                    {
                        "files": files,
                        "sizes": sizes,
                        "total_bytes": total_bytes,
                        "step": state.get("step"),
                        "epoch": state.get("epoch"),
                        "world_size": world,
                        "format_version": 1,
                        "created_at": time.time(),
                    },
                    sort_keys=True,
                )
            )
            os.replace(staging, root / generation)
            pointer = root / (".latest-" + identifier[0])
            pointer.write_text(json.dumps({"generation": generation}))
            os.replace(pointer, root / "latest.json")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Checkpoint publication failed: {error[0]}")
    return root / generation


def _validate_rng(rng, source):
    """Check saved CPU streams using isolated generators, without changing global RNG."""
    required = {"torch", "python", "numpy", "accelerator"}
    if not isinstance(rng, dict) or not required.issubset(rng):
        raise ValueError(f"{source}: incomplete rank RNG state")
    if rng["accelerator"] not in ("cpu", "cuda", "npu"):
        raise ValueError(f"{source}: unsupported RNG accelerator")
    value = rng["torch"]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError(f"{source}: torch RNG must be a one-dimensional uint8 tensor")
    try:
        torch.Generator(device="cpu").set_state(value)
        python_state = rng["python"]
        if (
            not isinstance(python_state, tuple)
            or len(python_state) != 3
            or type(python_state[0]) is not int
            or (
                python_state[2] is not None
                and (not isinstance(python_state[2], (int, float)) or not np.isfinite(python_state[2]))
            )
        ):
            raise ValueError("Invalid Python RNG state or Gaussian cache")
        random.Random().setstate(python_state)
        numpy_state = rng["numpy"]
        if (
            not isinstance(numpy_state, tuple)
            or len(numpy_state) != 5
            or not isinstance(numpy_state[1], np.ndarray)
            or numpy_state[1].dtype != np.uint32
            or numpy_state[1].shape != (624,)
        ):
            raise ValueError("NumPy RNG must contain a full MT19937 uint32 state")
        if (
            type(numpy_state[2]) is not int
            or not 0 <= numpy_state[2] <= 624
            or type(numpy_state[3]) is not int
            or numpy_state[3] not in (0, 1)
            or not isinstance(numpy_state[4], (int, float, np.floating))
            or not np.isfinite(numpy_state[4])
        ):
            raise ValueError("Invalid NumPy RNG position or Gaussian cache")
        np.random.RandomState().set_state(numpy_state)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(f"{source}: RNG state cannot be restored: {exc}") from exc
    if rng["accelerator"] == "cpu":
        if "device" in rng:
            raise ValueError(f"{source}: CPU RNG state cannot contain device streams")
    else:
        streams = rng.get("device")
        if not isinstance(streams, (list, tuple)) or not streams:
            raise ValueError(f"{source}: missing device RNG streams")
        if any(
            not isinstance(s, torch.Tensor) or s.dtype != torch.uint8 or s.ndim != 1 or not s.numel() for s in streams
        ):
            raise ValueError(f"{source}: device RNG streams must be nonempty one-dimensional uint8 tensors")


def _collective_preflight(check):
    """Every participant observes the same failure before any model/optimizer writes."""
    rank, world = _rank_world()
    result, error = None, None
    try:
        result = check()
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * world
        dist.all_gather_object(errors, error)
    failures = [entry for entry in errors if entry is not None]
    if failures:
        raise ValueError("Checkpoint preflight failed; " + "; ".join(failures))
    return result


def _validate_checkpoint_local(directory, *, expected_config=None):
    root = Path(directory)
    if (root / "latest.json").is_file():
        name = json.loads((root / "latest.json").read_text())["generation"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint generation")
        root = root / name
    if not (root / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {root}")
    files = json.loads((root / "complete.json").read_text())["files"]
    _, world = _rank_world()
    required = {"training_state.pt", *(f"rng-{r:05d}.pt" for r in range(world))}
    if (
        not required.issubset(files)
        or not any(k.startswith("model/") for k in files)
        or not any(k.startswith("optimizer/") for k in files)
    ):
        raise ValueError("Checkpoint is missing training state, rank RNG, model or optimizer files")
    for name, digest in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Checkpoint checksum mismatch or missing file: {name}")
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=False)
    required_state = {"step", "epoch", "sampler_cursor", "world_size", "format_version", "config"}
    if not isinstance(state, dict) or not required_state.issubset(state):
        raise ValueError("Incomplete public training state")
    if any(type(state[k]) is not int or state[k] < 0 for k in ("step", "epoch", "sampler_cursor")):
        raise ValueError("Invalid optimizer step, epoch or sampler cursor")
    if (
        type(state["format_version"]) is not int
        or type(state["world_size"]) is not int
        or state["format_version"] != 1
        or state["world_size"] != world
    ):
        raise ValueError("Resume requires the same world size and supported checkpoint format")
    if state["config"] is not None and not isinstance(state["config"], dict):
        raise ValueError("Checkpoint configuration must be a mapping or None")
    if expected_config is not None and state.get("config") != expected_config:
        raise ValueError("Resume configuration/replay fingerprint differs from the checkpoint")
    for rank in range(world):
        name = f"rng-{rank:05d}.pt"
        _validate_rng(torch.load(root / name, map_location="cpu", weights_only=False), name)
    return root, state


def validate_checkpoint(directory, *, expected_config=None):
    """Collectively verify all files and every rank's RNG without modifying live state."""
    return _collective_preflight(lambda: _validate_checkpoint_local(directory, expected_config=expected_config))


def restore_checkpoint(directory, model, optimizer, *, expected_config=None, booster=None):
    root, state = validate_checkpoint(directory, expected_config=expected_config)
    rank, _ = _rank_world()

    def check_local_device():
        rng = torch.load(root / f"rng-{rank:05d}.pt", map_location="cpu", weights_only=False)
        _validate_rng(rng, f"rng-{rank:05d}.pt")
        accelerator = _accelerator()
        if rng["accelerator"] != accelerator.name:
            raise ValueError("Resume requires the same accelerator type for rank RNG restoration")
        if accelerator.name != "cpu":
            reference = accelerator.get_rng_state_all()
            if len(rng["device"]) != accelerator.device_count() or len(reference) != len(rng["device"]):
                raise ValueError("Device RNG stream count differs from the current accelerator")
            if any(
                saved.dtype != live.dtype or saved.shape != live.shape for saved, live in zip(rng["device"], reference)
            ):
                raise ValueError("Device RNG tensor type/shape differs from the current accelerator")
        return rng, accelerator

    rng, accelerator = _collective_preflight(check_local_device)
    booster = booster or model._colossal_booster
    booster.load_model(model, str(root / "model"))
    booster.load_optimizer(optimizer, str(root / "optimizer"))
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if "device" in rng:
        accelerator.set_rng_state_all(rng["device"])
    return state
