"""Collective, immutable checkpoint generations with rank-local random state.

A save stages shards and per-rank files on every rank and is published by rank zero alone:
it hashes the staged tree into ``complete.json``, renames the staging directory into a
generation and swaps ``latest.json``.  A failed generation never replaces ``latest.json``.

With ``async_=True`` the shard writers are background threads and rank zero's publisher thread
joins them itself, then publishes as soon as the bytes are on disk.  The training step
therefore pays only the device-to-host staging copy -- the same exposed cost as the upstream
Cosmos ``DistributedCheckpointer`` -- and never waits for serialization, hashing or the rename.
Non-saving ranks have no asynchronous writer at all, and every rank's small synchronous files
(``rng-<rank>.pt``, ``training_state.pt``) are on disk before the save-time barrier, so the
publisher needs no further collective.  ``flush_checkpoint`` is where all ranks confirm the
outcome together.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

FORMAT_VERSION = 3
# Version 1 wrote the optimizer in the legacy .bin layout and carried no per-rank extra state.
# Version 2 added the immutable generation layout.  Version 3 adds the fp32 master weights that
# mixed-precision training actually updates; the bf16 model shards alone round those values away.
READABLE_FORMAT_VERSIONS = (1, 2, FORMAT_VERSION)

MASTER_WEIGHTS_FORMAT = 1


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


def _master_parameter_targets(model, optimizer):
    """Return ``(keys, master_tensors)`` for the optimizer's fp32 master parameters.

    Mixed-precision plugins keep a working bf16/fp16 model parameter and a separate fp32 master
    parameter that the optimizer actually updates.  Booster's model save writes the working
    parameter; its optimizer save writes the moments, not the master values.  Loading therefore
    rounds the masters back to the working dtype unless we snapshot them here.
    """
    getter = getattr(optimizer, "get_master_to_working_map", None)
    master_to_working = getter() if callable(getter) else None
    if not master_to_working:
        return [], []
    sources = [optimizer]
    inner = getattr(optimizer, "optim", None)
    if inner is not None:
        sources.append(inner)
    masters = []
    for source in sources:
        params = [p for group in getattr(source, "param_groups", ()) for p in group["params"]]
        masters = [p for p in params if id(p) in master_to_working]
        if masters:
            break
    if not masters:
        raise RuntimeError("mixed-precision optimizer exposes master weights but no master tensors were found")
    names = {id(param): name for name, param in model.named_parameters()}
    keys = []
    seen = {}
    for index, master in enumerate(masters):
        working = master_to_working.get(id(master))
        name = names.get(id(working), f"param-{index}")
        count = seen.get(name, 0)
        seen[name] = count + 1
        keys.append(f"{name}#{count}")
    return keys, masters


def master_weights_payload(model, optimizer):
    """Snapshot the fp32 master parameters for the checkpoint, or ``None`` for a pure fp32 run."""
    keys, masters = _master_parameter_targets(model, optimizer)
    if not masters:
        return None
    tensors = [master.detach().float().cpu().contiguous() for master in masters]
    return {"format": MASTER_WEIGHTS_FORMAT, "keys": keys, "tensors": tensors}


def load_master_weights(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != MASTER_WEIGHTS_FORMAT:
        raise ValueError(f"{path}: unsupported master-weight payload")
    keys, tensors = payload.get("keys"), payload.get("tensors")
    if not isinstance(keys, list) or not isinstance(tensors, list) or len(keys) != len(tensors):
        raise ValueError(f"{path}: malformed master-weight payload")
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"{path}: master weight {index} is not a float32 tensor")
    return keys, tensors


def validate_master_weights(keys, tensors, model, optimizer):
    current_keys, masters = _master_parameter_targets(model, optimizer)
    if keys != current_keys:
        raise ValueError("master-weight keys differ from the current model/optimizer layout")
    if len(tensors) != len(masters):
        raise ValueError("master-weight count differs from the current optimizer")
    for index, (tensor, master) in enumerate(zip(tensors, masters)):
        if tuple(tensor.shape) != tuple(master.shape):
            raise ValueError(f"master weight {index} shape {tuple(tensor.shape)} != {tuple(master.shape)}")


def apply_master_weights(tensors, model, optimizer):
    _, masters = _master_parameter_targets(model, optimizer)
    with torch.no_grad():
        for tensor, master in zip(tensors, masters):
            master.data.copy_(tensor.to(device=master.device, dtype=master.dtype))


def apply_checkpoint_modes(root) -> None:
    """Make a finished generation group-readable. Called before publication, never during a write."""
    for path in Path(root).rglob("*"):
        os.chmod(path, 0o664 if path.is_file() else 0o775)


def _digest_tree(root: Path):
    files = {}
    sizes = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(root))
        files[name] = file_digest(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        sizes[name] = size
        total_bytes += size
    return files, sizes, total_bytes


def _collective_errors(local_error: Optional[str]) -> None:
    """Gate every rank on the same outcome. All ranks MUST call this, even after a local failure.

    A single ``all_gather_object`` is both the barrier and the error aggregation: a rank that
    raises before it would leave its peers waiting inside the next collective.
    """
    rank, _ = _rank_world()
    if local_error is not None:
        local_error = f"rank {rank}: {local_error}"
    if not dist.is_initialized():
        if local_error is not None:
            raise RuntimeError(f"Checkpoint flush failed: {local_error}")
        return
    errors = [None] * dist.get_world_size()
    try:
        dist.all_gather_object(errors, local_error)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint error aggregation failed; a peer may have died: {exc}") from exc
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError("Checkpoint flush failed; " + "; ".join(failures))


@dataclass
class PendingCheckpoint:
    """A generation whose files are queued or written but which is not yet published.

    ``publisher`` exists on rank zero only; the other ranks hold the same deterministic paths
    and learn the outcome from :func:`flush_checkpoint`.
    """

    generation: Path
    staging: Path
    identifier: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    world: int = 1
    checkpoint_io: Any = None
    publisher: Optional[threading.Thread] = None
    error: Optional[str] = None

    def start_publisher(self, writers=(), chmod: bool = False):
        self.publisher = threading.Thread(
            target=self._publish,
            args=(list(writers), chmod),
            name="checkpoint-publisher",
            daemon=True,
        )
        self.publisher.start()
        return self

    def _publish(self, writers, chmod: bool) -> None:
        try:
            _publish_generation(self, writers, chmod)
        except BaseException as exc:  # reported by flush_checkpoint on every rank
            self.error = f"{type(exc).__name__}: {exc}"


def _publish_generation(pending: PendingCheckpoint, writers, chmod: bool) -> None:
    """Rank zero only. Joins the shard writers, then makes the generation visible.

    No collectives: this runs on the publisher thread while training continues.
    """
    for writer in writers:
        writer.synchronize()
    if chmod:
        apply_checkpoint_modes(pending.staging)
    files, sizes, total_bytes = _digest_tree(pending.staging)
    (pending.staging / "complete.json").write_text(
        json.dumps(
            {
                "files": files,
                "sizes": sizes,
                "total_bytes": total_bytes,
                "step": pending.step,
                "epoch": pending.epoch,
                "world_size": pending.world,
                "format_version": FORMAT_VERSION,
                "created_at": time.time(),
            },
            sort_keys=True,
        )
    )
    root = pending.generation.parent
    os.replace(pending.staging, pending.generation)
    pointer = root / (".latest-" + pending.identifier)
    pointer.write_text(json.dumps({"generation": pending.generation.name}))
    os.replace(pointer, root / "latest.json")


def save_checkpoint(
    directory,
    model,
    optimizer,
    state,
    *,
    booster=None,
    async_: bool = False,
    on_rank_state: Optional[Callable[[], dict]] = None,
    chmod: bool = False,
) -> PendingCheckpoint:
    """All ranks participate; only rank zero publishes a complete generation.

    Model/optimizer writers are owned by Booster's checkpoint IO.  The completion manifest
    covers every shard and RNG file.  A failed generation never replaces latest.

    ``on_rank_state`` snapshots rank-private randomness that the global RNG does not cover
    (a separate ``torch.Generator``, for example); it is stored under ``rng-<rank>.pt["extra"]``
    and handed back by :func:`apply_checkpoint`.

    With ``async_=True`` the returned generation is not published yet.  The caller must pass it
    to :func:`flush_checkpoint` at a point every rank reaches.
    """
    rank, world = _rank_world()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identifier = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(identifier, src=0)
    staging = root / (".incomplete-" + identifier[0])
    staging.mkdir(parents=True, exist_ok=True)
    booster = booster or model._colossal_booster
    checkpoint_io = getattr(booster, "checkpoint_io", None)
    booster.save_model(model, str(staging / "model"), shard=True, use_safetensors=True, use_async=async_)
    booster.save_optimizer(optimizer, str(staging / "optimizer"), shard=True, use_async=async_)
    # The working parameters above are bf16/fp16 under mixed precision.  The optimizer updates the
    # separate fp32 master parameters, and Booster's optimizer save does not persist their values.
    # Snapshot them into the same staging generation; the host copy is the exposed cost of an exact
    # resume, while serialization joins the generation's normal publication path.
    master_payload = master_weights_payload(model, optimizer)
    if master_payload is not None:
        torch.save(master_payload, staging / f"master-{rank:05d}.pt")
    accelerator = _accelerator()
    rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "accelerator": accelerator.name,
    }
    if accelerator.name != "cpu":
        rng["device"] = accelerator.get_rng_state_all()
    extra = on_rank_state() if on_rank_state is not None else None
    if extra is not None:
        rng["extra"] = extra
    torch.save(rng, staging / f"rng-{rank:05d}.pt")
    if rank == 0:
        torch.save(
            {
                **state,
                "world_size": world,
                "format_version": FORMAT_VERSION,
                "master_weights": master_payload is not None,
            },
            staging / "training_state.pt",
        )
    _barrier()
    pending = PendingCheckpoint(
        generation=root / ("checkpoint-" + identifier[0]),
        staging=staging,
        identifier=identifier[0],
        step=state.get("step"),
        epoch=state.get("epoch"),
        world=world,
        checkpoint_io=checkpoint_io,
    )
    # Copy the writer list: the publisher must not race the training thread's next registration,
    # and clearing it here would break CheckpointIO._sync_io as the back-pressure point.
    writers = list(getattr(checkpoint_io, "async_writers", ()) or ()) if async_ else []
    if async_:
        if rank == 0:
            pending.start_publisher(writers, chmod=chmod)
        return pending
    error = [None]
    if rank == 0:
        try:
            _publish_generation(pending, writers=(), chmod=chmod)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Checkpoint publication failed: {error[0]}")
    # All ranks know the generation id; the path is deterministic even on ranks that did not publish it.
    return pending


def flush_checkpoint(pending: PendingCheckpoint, *, ledger=None, checkpoint_id=None, stats=None) -> Path:
    """Join a pending generation on every rank and publish one shared outcome.

    Collective. Every rank reaches this at the same point in the loop: on the next save (before
    staging into the reused pinned buffers) and once on the way out.
    """
    rank, _ = _rank_world()
    local_error = None
    if rank == 0 and pending.publisher is not None:
        pending.publisher.join()
        if pending.error is not None:
            local_error = pending.error
    # Drain this rank's own writers. Rank zero's were already joined by its publisher; the list
    # is still registered, so this second synchronous join is what frees the staging buffers.
    if pending.checkpoint_io is not None:
        try:
            pending.checkpoint_io._sync_io()
        except BaseException as exc:
            local_error = local_error or f"{type(exc).__name__}: {exc}"
    _collective_errors(local_error)
    if ledger is not None and checkpoint_id is not None:
        finished = ledger.end(
            checkpoint_id,
            status="complete",
            path=pending.generation,
            generation=pending.generation.name,
        )
        if stats is not None:
            stats["count"] += 1
            stats["bytes"] += int(finished.get("bytes") or 0)
    return pending.generation


def _validate_rng(rng, source):
    """Check saved CPU streams using isolated generators, without changing global RNG."""
    required = {"torch", "python", "numpy", "accelerator"}
    if not isinstance(rng, dict) or not required.issubset(rng):
        raise ValueError(f"{source}: incomplete rank RNG state")
    if rng["accelerator"] not in ("cpu", "cuda", "npu"):
        raise ValueError(f"{source}: unsupported RNG accelerator")
    if "extra" in rng and not isinstance(rng["extra"], dict):
        raise ValueError(f"{source}: rank extra state must be a mapping")
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
        or state["format_version"] not in READABLE_FORMAT_VERSIONS
        or state["world_size"] != world
    ):
        raise ValueError("Resume requires the same world size and supported checkpoint format")
    if state["config"] is not None and not isinstance(state["config"], dict):
        raise ValueError("Checkpoint configuration must be a mapping or None")
    if expected_config is not None and state.get("config") != expected_config:
        raise ValueError("Resume configuration/replay fingerprint differs from the checkpoint")
    master_weights = state.get("master_weights", False)
    if not isinstance(master_weights, bool):
        raise ValueError("Checkpoint master-weight flag must be boolean")
    if master_weights:
        for rank in range(world):
            if not (root / f"master-{rank:05d}.pt").is_file():
                raise ValueError(f"missing master-{rank:05d}.pt")
    for rank in range(world):
        name = f"rng-{rank:05d}.pt"
        _validate_rng(torch.load(root / name, map_location="cpu", weights_only=False), name)
    return root, state


def validate_checkpoint(directory, *, expected_config=None):
    """Collectively verify all files and every rank's RNG without modifying live state."""
    return _collective_preflight(lambda: _validate_checkpoint_local(directory, expected_config=expected_config))


def apply_checkpoint(root, state, model, optimizer, *, booster=None, on_rank_state=None):
    """Load a validated generation into live training state. Collective."""

    def check_local_device():
        rank, _ = _rank_world()
        rng = torch.load(Path(root) / f"rng-{rank:05d}.pt", map_location="cpu", weights_only=False)
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
        master_payload = None
        if state.get("master_weights"):
            master_payload = load_master_weights(Path(root) / f"master-{rank:05d}.pt")
            validate_master_weights(master_payload[0], master_payload[1], model, optimizer)
        return rng, accelerator, master_payload

    rng, accelerator, master_payload = _collective_preflight(check_local_device)
    booster = booster or model._colossal_booster
    booster.load_model(model, str(Path(root) / "model"))
    booster.load_optimizer(optimizer, str(Path(root) / "optimizer"))
    if master_payload is not None:
        # load_model seeds the master parameters from the bf16 working shards; restore the exact fp32 values
        # that were present when the generation was written.
        apply_master_weights(master_payload[1], model, optimizer)
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if "device" in rng:
        accelerator.set_rng_state_all(rng["device"])
    if on_rank_state is not None:
        on_rank_state(rng.get("extra"))
    return state


def restore_checkpoint(directory, model, optimizer, *, expected_config=None, booster=None, on_rank_state=None):
    root, state = validate_checkpoint(directory, expected_config=expected_config)
    return apply_checkpoint(root, state, model, optimizer, booster=booster, on_rank_state=on_rank_state)
