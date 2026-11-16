# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
import threading
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import _TYPES, safe_open, save_file

from ._optimizer_safetensors import flatten_optimizer, unflatten_optimizer

try:
    from tensornvme.async_file_io import AsyncFileWriter

    HAS_TENSORNVME = True
except Exception:
    AsyncFileWriter = None
    HAS_TENSORNVME = False
_TYPES_INV = {v: k for k, v in _TYPES.items()}
ASYNC_WRITE_ENTRIES = 32

# TensorNVMe is a CUDA-only native extension, so Ascend NPU and CPU runs have no async backend
# there.  One warning is enough; a warning per save would drown the training log.
_threaded_fallback_warned = False


def _require_async_file_writer():
    if AsyncFileWriter is None:
        raise RuntimeError("Async checkpoint I/O requires TensorNVMe")
    return AsyncFileWriter


def _normalise_metadata(metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Safetensors metadata values are strings; optimizer schemas are JSON encoded once."""
    normalised = {k: v if isinstance(v, str) else json.dumps(v) for k, v in (metadata or {}).items()}
    if any(type(k) is not str for k in normalised):
        raise TypeError("Safetensors metadata keys must be strings")
    return normalised


def _to_host(tensor):
    """Detach a tensor and make sure a background thread may read it without a device copy."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    tensor = tensor.data.cpu()
    return tensor if tensor.is_contiguous() else tensor.contiguous()


class ThreadedSafetensorsWriter:
    """Background safetensors writer used when TensorNVMe is unavailable.

    Upstream Cosmos pays exactly one device-to-host staging copy before a save returns and
    leaves serialization, disk I/O and the ``latest`` pointer to a background worker.  The
    checkpoint IO's pinned buffers already provide that staging copy, so this writer keeps the
    same contract as :class:`tensornvme.async_file_io.AsyncFileWriter`
    (``sync_before_step`` / ``synchronize``) and only moves the write off the caller's thread.
    Every tensor is host-resident before the thread starts: a device tensor here would be read
    while the optimizer is already updating it.
    """

    def __init__(self, path: str, state_dict: Dict[str, torch.Tensor], metadata=None):
        global _threaded_fallback_warned
        if not _threaded_fallback_warned:
            _threaded_fallback_warned = True
            warnings.warn(
                "TensorNVMe is not installed; async checkpoint writes fall back to a background thread. "
                "The device-to-host copy is still staged before the save returns, so the training step "
                "is not blocked on serialization or disk I/O.",
                RuntimeWarning,
            )
        self.path = path
        self._error = None
        self._metadata = _normalise_metadata(metadata)
        host = {name: _to_host(tensor) for name, tensor in state_dict.items()}
        self._thread = threading.Thread(target=self._write, args=(host,), name="safetensors-writer", daemon=True)
        self._thread.start()

    def _write(self, state_dict: Dict[str, torch.Tensor]) -> None:
        try:
            save_file(state_dict, self.path, metadata=self._metadata)
        except BaseException as exc:  # re-raised on the training thread by synchronize()
            self._error = exc

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def sync_before_step(self) -> None:
        """The host copy happened in the caller; nothing device-side is still in flight."""

    def synchronize(self) -> None:
        self._thread.join()
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f"Background safetensors write failed for {self.path}") from error


@dataclass
class TensorInfo:
    dtype: str
    shape: List[int]
    data_offsets: Tuple[int, int]


@dataclass
class PreparedData:
    n: int
    header_bytes: bytes
    offset: int


def _flatten_optim_state_dict(state_dict: dict, seperator: str = ".") -> Tuple[dict, dict]:
    """Encode optimizer data with the versioned, non-executable schema."""
    return flatten_optimizer(state_dict)


def _unflatten_optim_state_dict(
    flat_dict: dict, metadata: Optional[dict] = None, seperator: str = ".", *, expected_kind=None
):
    return unflatten_optimizer(flat_dict, metadata, expected_kind, seperator)


def prepare(
    data: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> Tuple[PreparedData, List[torch.Tensor], List[str]]:
    # Safetensors metadata values are strings. Do not mutate cached metadata or
    # JSON-encode already encoded optimizer schemas a second time.
    metadata = _normalise_metadata(metadata)

    tensors = []
    tensor_keys = []
    header = {}
    offset = 0

    header_metadata = {"format": "pt"}
    if metadata is not None:
        header_metadata.update(metadata)
    header["__metadata__"] = header_metadata

    for name, tensor in data.items():
        n = tensor.numel() * tensor.element_size()
        tensor_info = TensorInfo(
            dtype=_TYPES_INV[tensor.dtype], shape=list(tensor.shape), data_offsets=(offset, offset + n)
        )
        offset += n
        header[name] = asdict(tensor_info)
        tensors.append(tensor)
        tensor_keys.append(name)

    header_buf = json.dumps(header).encode("utf-8")

    extra = (8 - len(header_buf) % 8) % 8
    header_buf += b" " * extra

    n = len(header_buf)

    return PreparedData(n=n, header_bytes=header_buf, offset=offset), tensors, tensor_keys


def save(path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> None:
    if AsyncFileWriter is None:
        return ThreadedSafetensorsWriter(path, state_dict, metadata)
    prepared_data, tensors, _ = prepare(state_dict, metadata)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset
    writer_cls = _require_async_file_writer()
    f_writer = writer_cls(path, n_entries=ASYNC_WRITE_ENTRIES, backend="pthread", n_tasks=2 + len(tensors))
    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    for tensor in tensors:
        f_writer.write_raw(tensor, tensor.data_ptr(), tensor.numel() * tensor.element_size(), f_writer.offset)
    return f_writer


def save_nested(path: str, state_dict: Dict[str, torch.Tensor]) -> None:
    flatten_data, metadata = _flatten_optim_state_dict(state_dict)
    return save(path, flatten_data, metadata)


def move_and_save(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    state_dict_pinned: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    if AsyncFileWriter is None:
        # Same staging as the TensorNVMe path, done synchronously: the caller must not hand a
        # live device tensor to a thread that runs while the optimizer keeps updating it.
        host = {}
        for name, tensor in state_dict.items():
            pinned = state_dict_pinned.get(name) if state_dict_pinned is not None else None
            if isinstance(pinned, torch.Tensor) and pinned.shape == tensor.shape and pinned.dtype == tensor.dtype:
                pinned.copy_(tensor)
                host[name] = pinned
            else:
                host[name] = tensor
        return ThreadedSafetensorsWriter(path, host, metadata)
    prepared_data, _, tensor_keys = prepare(state_dict, metadata)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset
    writer_cls = _require_async_file_writer()
    f_writer = writer_cls(path, n_entries=ASYNC_WRITE_ENTRIES, backend="pthread", n_tasks=2 + len(tensor_keys))
    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    if tensor_keys:
        f_writer.register_h2d(len(tensor_keys))
    for name in tensor_keys:
        if state_dict_pinned:
            f_writer.write_tensor(state_dict[name], state_dict_pinned[name])
        else:
            f_writer.write_tensor(state_dict[name])
    return f_writer


class _FileTensors(Mapping):
    def __init__(self, reader):
        self.reader = reader
        self.names = set(reader.keys())

    def __getitem__(self, key):
        if key not in self.names:
            raise KeyError(key)
        return self.reader.get_tensor(key)

    def __contains__(self, key):
        return key in self.names

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def load_flat(checkpoint_path, seperator: str = ".", *, expected_kind=None):
    """Load data-only optimizer state; legacy embedded pickle is rejected."""
    with safe_open(checkpoint_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        # The lazy mapping validates metadata before fetching tensor values;
        # both metadata and values come from this one open file.
        return _unflatten_optim_state_dict(_FileTensors(reader), metadata, seperator, expected_kind=expected_kind)


def save_optimizer_file(path, state_dict):
    """Synchronously write the same data-only schema used by async optimizers."""
    import os
    import tempfile
    from pathlib import Path

    tensors, metadata = flatten_optimizer(state_dict)
    path = Path(path)
    # Private storage also supports aliased tensors with safetensors.save_file.
    tensors = {key: tensor.cpu().clone().contiguous() for key, tensor in tensors.items()}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".optimizer-", delete=False) as temporary:
        temporary_path = temporary.name
    try:
        save_file(tensors, temporary_path, metadata=metadata)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
