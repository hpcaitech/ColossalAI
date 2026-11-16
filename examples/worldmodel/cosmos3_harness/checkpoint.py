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

When ``--out`` is each node's local disk, rank zero's node holds the model, optimizer and training
state while every other node holds only the rank files it wrote.  The manifest still covers all of
them (each rank gathers its own file digests before the barrier), and resume runs on every node at
once: :func:`materialize_local_generation` reads the generation name and manifest from rank zero,
each rank reports whether its node is complete, and only a node that is missing the shared files
receives them, streamed over the interconnect the job already has open.  A shared filesystem makes
that step a no-op.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

FORMAT_VERSION = 4
# Version 1 wrote the optimizer in the legacy .bin layout and carried no per-rank extra state.
# Version 2 added the immutable generation layout.  Version 3 adds the fp32 master weights that
# mixed-precision training actually updates; the bf16 model shards alone round those values away.
READABLE_FORMAT_VERSIONS = (1, 2, 3, FORMAT_VERSION)

MASTER_WEIGHTS_FORMAT = 1

#: Shared by the writer and the validator.  ``extra`` is the caller's rank-private state (the noise generator);
#: ``device`` exists only for a non-CPU accelerator.  A future field belongs in this schema before the writer
#: emits it, otherwise the next resume rejects a generation its own writer just produced.
RNG_REQUIRED_FIELDS = ("torch", "python", "numpy", "accelerator")
RNG_OPTIONAL_FIELDS = ("device", "extra")


def _rng_fields(accelerator_name: str) -> Tuple[str, ...]:
    fields = RNG_REQUIRED_FIELDS
    if accelerator_name != "cpu":
        fields = fields + ("device",)
    return fields


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


#: Files that exactly one rank writes and that only that rank reads back.  They live on the node that
#: wrote them, so a run whose ``--out`` is each node's local disk legitimately lacks the other nodes'
#: copies; ``complete.json`` still records every one of them, digest included.
_RANK_FILE_PREFIXES = ("rng-", "master-", "ema-")


class _DigestingFile:
    """A file object that hashes the bytes it writes, so a saved file's digest costs no second read."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def write(self, data) -> int:
        self._digest.update(data)
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


#: Payload tag for :func:`rank_tensors_payload`.  Bumped only if the layout below changes.
RANK_TENSORS_FORMAT = "cosmos3-rank-tensors-1"


def rank_tensors_payload(keys, tensors) -> Dict[str, Any]:
    """Package one rank's named fp32 tensors for its own file in the generation.

    This is the shape the master weights already use, exposed for state that belongs to the caller
    rather than to the optimizer -- an EMA's shadow, for instance.  It goes in a file of its own rather
    than in ``rng-<rank>.pt`` because every rank reads its own RNG file and nothing else reads this: a
    payload the size of a model copy has no business on a path other ranks walk.

    Args:
        keys (Sequence[str]): A stable name per tensor, so a restore can refuse a changed layout
            instead of matching by position.
        tensors (Sequence[torch.Tensor]): The values, in the same order as ``keys``.

    Returns:
        dict: The payload to hand to ``save_checkpoint(rank_tensors=...)``.

    Raises:
        ValueError: ``keys`` and ``tensors`` disagree in length.
    """
    keys = [str(key) for key in keys]
    if len(keys) != len(tensors):
        raise ValueError("rank tensor keys and tensors differ in length")
    return {
        "format": RANK_TENSORS_FORMAT,
        "keys": keys,
        "tensors": [tensor.detach().float().cpu().contiguous() for tensor in tensors],
    }


def load_rank_tensors(path):
    """Read a :func:`rank_tensors_payload` file back, checking the layout before the caller uses it.

    Args:
        path (str | Path): The ``<label>-<rank>.pt`` file inside a generation.

    Returns:
        tuple[list[str], list[torch.Tensor]]: The keys and the fp32 tensors.

    Raises:
        ValueError: The file is not such a payload, or its entries are not fp32 tensors.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != RANK_TENSORS_FORMAT:
        raise ValueError(f"{path}: unsupported rank tensor payload")
    keys, tensors = payload.get("keys"), payload.get("tensors")
    if not isinstance(keys, list) or not isinstance(tensors, list) or len(keys) != len(tensors):
        raise ValueError(f"{path}: malformed rank tensor payload")
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"{path}: rank tensor {index} is not a float32 tensor")
    return keys, tensors


def _save_with_digest(payload, path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("wb") as stream:
        writer = _DigestingFile(stream)
        torch.save(payload, writer)
    return {"sha256": writer.hexdigest(), "size": path.stat().st_size}


def _is_rank_local(name) -> bool:
    return str(name).startswith(_RANK_FILE_PREFIXES)


def _rank_of_file(name) -> Optional[int]:
    stem = Path(str(name)).name
    _, _, suffix = stem.partition("-")
    try:
        return int(suffix.split(".")[0])
    except (TypeError, ValueError):
        return None


def _local_rank_index() -> int:
    """This process's rank inside its node.  torchrun sets ``LOCAL_RANK``; a single-process run is 0."""
    raw = os.environ.get("LOCAL_RANK")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _checkpoint_root_of(directory) -> Path:
    """The directory that holds the generations, whichever of the two a caller passed."""
    path = Path(directory)
    return path.parent if path.name.startswith("checkpoint-") else path


def _describe_generation(directory) -> Dict[str, Any]:
    """Rank zero's view of a generation: its name, the manifest, and the size of every shared file.

    Only rank zero's node holds ``complete.json`` when ``--out`` is node-local, so this is the payload
    every other node receives before it knows what to fetch.
    """
    root = Path(directory)
    if (root / "latest.json").is_file():
        name = json.loads((root / "latest.json").read_text())["generation"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint generation")
        root = root / name
    if not (root / "complete.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {root}")
    manifest_text = (root / "complete.json").read_text()
    manifest = json.loads(manifest_text)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{root}: complete.json has no file table")
    sizes = dict(manifest.get("sizes") or {})
    for name in files:
        if _is_rank_local(name) or name in sizes:
            continue
        path = root / name
        if not path.is_file():
            raise ValueError(f"{root}: manifest lists {name}, which is not on rank 0's node")
        sizes[name] = path.stat().st_size
    return {"generation": root.name, "files": files, "sizes": sizes, "manifest": manifest_text}


def _generation_ready(directory: Path, files: Dict[str, Any], rank: int) -> bool:
    """Whether this node already holds every file this rank will read."""
    if not (directory / "complete.json").is_file():
        return False
    for name in files:
        if _is_rank_local(name) and _rank_of_file(name) != rank:
            continue
        if not (directory / name).is_file():
            return False
    return True


def _transfer_device():
    accelerator = _accelerator()
    if accelerator.name == "cpu":
        return torch.device("cpu")
    return accelerator.get_current_device()


def _transfer_chunk_bytes() -> int:
    raw = os.environ.get("COSMOS3_CHECKPOINT_TRANSFER_CHUNK_MIB", "64")
    try:
        mebibytes = max(1, int(raw))
    except (TypeError, ValueError):
        mebibytes = 64
    return mebibytes * 1024 * 1024


def _transfer_file(source, destination, size, *, device, chunk_bytes) -> None:
    """Stream one file from rank zero to every node; only each node's head rank writes it."""
    if not dist.is_initialized():
        raise RuntimeError("checkpoint transfer requires an initialized process group")
    rank = dist.get_rank()
    reader = Path(source).open("rb") if rank == 0 and source is not None else None
    writer = Path(destination).open("wb") if destination is not None else None
    try:
        remaining = int(size)
        while remaining > 0:
            count = min(chunk_bytes, remaining)
            buffer = torch.empty(count, dtype=torch.uint8, device=device)
            if reader is not None:
                data = reader.read(count)
                if len(data) != count:
                    raise RuntimeError(f"{source}: short read while transferring the checkpoint")
                buffer.copy_(torch.frombuffer(data, dtype=torch.uint8))
            dist.broadcast(buffer, src=0)
            if writer is not None:
                writer.write(buffer.cpu().numpy().tobytes())
            remaining -= count
    finally:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()


def _local_rank_file(local_root: Path, generation: str, name: str) -> Optional[Path]:
    """Where this node kept a rank file at save time: the published generation, else its staging dir."""
    candidates = [local_root / generation / name]
    if generation.startswith("checkpoint-"):
        candidates.append(local_root / (".incomplete-" + generation[len("checkpoint-") :]) / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _node_needs_write(flags, rank: int, local: int) -> bool:
    """Whether this node is missing something.  Ranks are laid out node by node, as torchrun starts them."""
    raw = os.environ.get("LOCAL_WORLD_SIZE")
    try:
        local_world = max(1, int(raw)) if raw is not None else 1
    except (TypeError, ValueError):
        local_world = 1
    start = rank - local
    stop = min(len(flags), start + local_world)
    return not all(bool(flags[index]) for index in range(max(0, start), max(0, stop)))


def _fetch_generation(payload, local_root: Path, generation_dir: Path, *, rank: int, world: int, flags) -> None:
    """Rebuild this node's copy of the generation: rank zero's shared files plus the rank files it wrote.

    Every rank joins the broadcast, but only a node that is actually incomplete writes the bytes out;
    a node whose disks are already complete participates and keeps its files.
    """
    generation = payload["generation"]
    files = payload["files"]
    sizes = payload.get("sizes") or {}
    local = _local_rank_index()
    needs_write = _node_needs_write(flags, rank, local)
    missing = [
        name
        for name in files
        if _is_rank_local(name)
        and _rank_of_file(name) == rank
        and _local_rank_file(local_root, generation, name) is None
    ]
    local_error = None
    if missing:
        local_error = (
            f"rank {rank}: this node no longer has {', '.join(sorted(missing))}; the save wrote them here and "
            "only this node ever held them"
        )
    errors = [local_error]
    if dist.is_initialized():
        errors = [None] * world
        dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error]
    if failures:
        raise ValueError("Checkpoint resume cannot rebuild the generation: " + "; ".join(failures))

    writer_node = local == 0 and needs_write
    # Every rank on an incomplete node copies its own rank file in here, but only the node head creates
    # the directory, so its name has to be the same for all of them: a per-process ``uuid4`` gave each
    # rank a directory of its own and only the head's existed, so every other rank on the node died on
    # the copy below while the rest of the job waited for it inside the broadcast.  One rank per node --
    # the only layout that was tested -- never reaches that, because there the head is the only rank.
    # The generation's own identifier is the same on every rank and distinct per generation.
    suffix = generation[len("checkpoint-") :] if generation.startswith("checkpoint-") else generation
    staging = local_root / (".resume-" + suffix)
    if writer_node:
        # An interrupted resume leaves one behind; publishing it would mix that attempt's files with these.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
    _barrier()
    device = _transfer_device()
    chunk_bytes = _transfer_chunk_bytes()
    transferred = 0
    for name in sorted(files):
        if _is_rank_local(name):
            continue
        target = staging / name if writer_node else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(
                f"[checkpoint] sending {name} ({int(sizes.get(name) or 0) / 2**20:.0f} MiB) to every node", flush=True
            )
        _transfer_file(
            generation_dir / name if rank == 0 else None,
            target,
            int(sizes.get(name) or 0),
            device=device,
            chunk_bytes=chunk_bytes,
        )
        transferred += 1
    if needs_write:
        # This rank's own files never leave their node: copy them out of the staging directory the save wrote.
        for name in files:
            if not _is_rank_local(name) or _rank_of_file(name) != rank:
                continue
            source = _local_rank_file(local_root, generation, name)
            shutil.copyfile(source, staging / name)
        if local == 0:
            (staging / "complete.json").write_text(payload["manifest"])
    _barrier()
    if writer_node:
        if generation_dir.exists():
            shutil.rmtree(generation_dir)
        os.replace(staging, generation_dir)
        pointer = local_root / ".latest-resume"
        pointer.write_text(json.dumps({"generation": generation}))
        os.replace(pointer, local_root / "latest.json")
    _barrier()
    if rank == 0:
        print(f"[checkpoint] materialised {generation} from {transferred} shared files", flush=True)


def materialize_local_generation(directory) -> Path:
    """Make the generation complete on this node, fetching rank zero's files when the disks are local.

    A run whose ``--out`` is each node's local disk publishes the model, optimizer and training state on
    rank zero's node only; every other node keeps just the rank files it wrote.  Resume runs on all nodes
    at once, so rank zero publishes the generation's name and manifest, each rank reports whether its node
    is complete, and a node that is missing the shared files receives them -- once per node, streamed over
    the interconnect the job already has open.  A shared filesystem makes this a no-op.
    """
    rank, world = _rank_world()
    payload, error = None, None
    if rank == 0:
        try:
            payload = _describe_generation(directory)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        holder = [{"payload": payload, "error": error}]
        dist.broadcast_object_list(holder, src=0)
        payload, error = holder[0]["payload"], holder[0]["error"]
    if error is not None:
        raise ValueError(f"Checkpoint generation on rank 0's node is not readable: {error}")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint generation is not readable")
    generation_dir = _checkpoint_root_of(directory) / payload["generation"]
    ready = _generation_ready(generation_dir, payload["files"], rank)
    flags = [ready]
    if dist.is_initialized():
        flags = [None] * world
        dist.all_gather_object(flags, ready)
    if all(flags):
        return generation_dir
    if not dist.is_initialized():
        raise ValueError(
            f"Incomplete checkpoint on this node: {generation_dir}; fetching the shared files needs the "
            "distributed job that resume runs in"
        )
    _fetch_generation(payload, _checkpoint_root_of(directory), generation_dir, rank=rank, world=world, flags=flags)
    return generation_dir


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
    #: Every rank's own files (``rng-*``, ``master-*``) with their digests.  They are written on that rank's
    #: node, so rank zero only knows them because they were gathered before the barrier.
    rank_files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
    # Rank files written on other nodes are in no local walk; the gather is the only record of them.
    local_names = set(sizes)
    for name, meta in (pending.rank_files or {}).items():
        digest = meta.get("sha256")
        if not isinstance(digest, str):
            continue
        files[name] = digest
        if name in local_names:
            continue
        size = int(meta.get("size") or 0)
        sizes[name] = size
        total_bytes += size
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
    rank_tensors: Optional[Mapping[str, Dict[str, Any]]] = None,
    chmod: bool = False,
) -> PendingCheckpoint:
    """All ranks participate; only rank zero publishes a complete generation.

    Model/optimizer writers are owned by Booster's checkpoint IO.  The completion manifest
    covers every shard and RNG file.  A failed generation never replaces latest.

    ``on_rank_state`` snapshots rank-private randomness that the global RNG does not cover
    (a separate ``torch.Generator``, for example); it is stored under ``rng-<rank>.pt["extra"]``
    and handed back by :func:`apply_checkpoint`.

    ``rank_tensors`` maps a label to a :func:`rank_tensors_payload`, written as ``<label>-<rank>.pt``
    beside the master weights.  The public state lists the labels present, so a run that keeps such
    state can refuse a generation that lacks it instead of silently starting that state over.

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
    rank_files: Dict[str, Dict[str, Any]] = {}
    master_payload = master_weights_payload(model, optimizer)
    if master_payload is not None:
        rank_files[f"master-{rank:05d}.pt"] = _save_with_digest(master_payload, staging / f"master-{rank:05d}.pt")
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
    rank_files[f"rng-{rank:05d}.pt"] = _save_with_digest(rng, staging / f"rng-{rank:05d}.pt")
    for label in sorted(rank_tensors or {}):
        name = f"{label}-{rank:05d}.pt"
        if not _is_rank_local(name):
            raise ValueError(f"rank tensor label {label!r} is not a known rank-local file prefix")
        rank_files[name] = _save_with_digest(rank_tensors[label], staging / name)
    if dist.is_initialized():
        # One small object exchange: every rank's own files stay on its node, so this is how the manifest
        # rank zero writes can still vouch for them when the run resumes from node-local disks.
        gathered = [None] * world
        dist.all_gather_object(gathered, {"rank": rank, "files": dict(rank_files)})
        for entry in gathered:
            if isinstance(entry, dict):
                rank_files.update(entry.get("files") or {})
    if rank == 0:
        torch.save(
            {
                **state,
                "world_size": world,
                "format_version": FORMAT_VERSION,
                "master_weights": master_payload is not None,
                "rank_tensors": sorted(rank_tensors or {}),
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
        rank_files=rank_files,
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
    required = set(RNG_REQUIRED_FIELDS)
    if not isinstance(rng, dict) or not required.issubset(rng):
        raise ValueError(f"{source}: incomplete rank RNG state")
    allowed = set(_rng_fields(str(rng.get("accelerator") or ""))) | {"extra"}
    unexpected = set(rng) - allowed
    if unexpected:
        # The file is the complete rank state this format version wrote; an unknown field means the writer and
        # this reader disagree, and silently ignoring it would resume from a state that is only partly restored.
        raise ValueError(f"{source}: unexpected RNG fields {sorted(unexpected)}")
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
    rank, world = _rank_world()
    required = {"training_state.pt", f"rng-{rank:05d}.pt"}
    if (
        not required.issubset(files)
        or not any(k.startswith("model/") for k in files)
        or not any(k.startswith("optimizer/") for k in files)
    ):
        raise ValueError("Checkpoint is missing training state, rank RNG, model or optimizer files")
    # Verify each file once per machine that holds it: a rank file by the rank that owns it, the shared
    # shards by the node's head rank.  ``_collective_preflight`` aggregates the failures, so one bad digest
    # still stops every rank before anything loads; what it no longer does is read the whole generation once
    # per rank -- 48 ranks times ~24 GiB on the six-node run, before the first step of every resume.
    head = _local_rank_index() == 0
    for name, digest in files.items():
        owner = _rank_of_file(name) if _is_rank_local(name) else None
        if owner is not None and owner != rank:
            # Another rank's RNG or master weights: on this node only when the filesystem is shared, and
            # that rank verifies them there.
            continue
        if owner is None and not head:
            continue
        path = root / name
        if not path.exists():
            raise ValueError(f"Checkpoint checksum mismatch or missing file: {name}")
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
        name = f"master-{rank:05d}.pt"
        if name not in files or not (root / name).is_file():
            raise ValueError(f"missing {name}")
    # Each rank restores only its own stream, so each validates only its own; every rank still does one, and
    # the preflight aggregates the failures.
    name = f"rng-{rank:05d}.pt"
    _validate_rng(torch.load(root / name, map_location="cpu", weights_only=False), name)
    return root, state


def validate_checkpoint(directory, *, expected_config=None):
    """Collectively verify the generation, materialising it on every node first when disks are node-local."""
    root = materialize_local_generation(directory)
    return _collective_preflight(lambda: _validate_checkpoint_local(root, expected_config=expected_config))


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
