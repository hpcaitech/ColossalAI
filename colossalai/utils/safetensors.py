# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
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
    warnings.warn(
        "Please install the latest tensornvme to use async save. pip install git+https://github.com/hpcaitech/TensorNVMe.git"
    )
_TYPES_INV = {v: k for k, v in _TYPES.items()}
ASYNC_WRITE_ENTRIES = 32


def _require_async_file_writer():
    if AsyncFileWriter is None:
        raise RuntimeError("Async checkpoint I/O requires TensorNVMe")
    return AsyncFileWriter


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
    metadata = {k: v if isinstance(v, str) else json.dumps(v) for k, v in (metadata or {}).items()}
    if any(type(k) is not str for k in metadata):
        raise TypeError("Safetensors metadata keys must be strings")

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
