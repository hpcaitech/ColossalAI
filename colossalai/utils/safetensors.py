# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import _TYPES, load_file, safe_open

try:
    from tensornvme.async_file_io import AsyncFileWriter
except Exception:
    warnings.warn(
        "Please install the latest tensornvme to use async save. pip install git+https://github.com/hpcaitech/TensorNVMe.git"
    )
_TYPES_INV = {v: k for k, v in _TYPES.items()}
import io

from torch.distributed.distributed_c10d import _pickler, _unpickler

ASYNC_WRITE_ENTRIES = 32


def _object_to_tensor(obj, device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    return byte_tensor


def _tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


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


def _cast_to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    return _object_to_tensor(obj, "cpu")


def _cast_to_object(tensor: torch.Tensor):
    return _tensor_to_object(tensor, tensor.numel() * tensor.element_size())


def _flatten_optim_state_dict(state_dict: dict, seperator: str = ".") -> Tuple[dict, Optional[dict]]:
    flat_dict = {}
    non_tensor_keys = []
    if "state" in state_dict:
        # 3-level dict
        states = state_dict["state"]
    else:
        # 2-level dict, usually for optimizer state dict shard
        states = state_dict

    for idx, d in states.items():
        for k, v in d.items():
            if v is None:
                continue
            nested_key = f"state{seperator}{idx}{seperator}{k}"
            if not isinstance(v, torch.Tensor):
                non_tensor_keys.append(nested_key)
            flat_dict[nested_key] = _cast_to_tensor(v)
    if "param_groups" in state_dict:
        flat_dict["param_groups"] = _cast_to_tensor(state_dict["param_groups"])
        non_tensor_keys.append("param_groups")
    if len(non_tensor_keys) > 0:
        metadata = {"non_tensor_keys": non_tensor_keys}
    else:
        metadata = None
    return flat_dict, metadata


def _unflatten_optim_state_dict(flat_dict: dict, metadata: Optional[dict] = None, seperator: str = "."):
    state_dict = {}

    if metadata is not None and "non_tensor_keys" in metadata:
        non_tensor_keys = json.loads(metadata["non_tensor_keys"])
    else:
        non_tensor_keys = []
    flat_dict = {k: _cast_to_object(v) if k in non_tensor_keys else v for k, v in flat_dict.items()}
    if "param_groups" in flat_dict:
        # 3-level dict
        state_dict["param_groups"] = flat_dict.pop("param_groups")
        state_dict["state"] = {}
        states = state_dict["state"]
    else:
        # 2-level dict, usually for optimizer state dict shard
        states = state_dict

    for k, v in flat_dict.items():
        parts = k.split(seperator)
        assert len(parts) == 3 and parts[0] == "state"
        idx = int(parts[1])
        key = parts[2]
        if idx not in states:
            states[idx] = {}
        states[idx][key] = v

    return state_dict


def prepare(
    data: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> Tuple[PreparedData, List[torch.Tensor], List[str]]:
    if metadata is not None:
        assert isinstance(metadata, dict)
        for k, v in metadata.items():
            metadata[k] = json.dumps(v)
            assert isinstance(k, str)
            assert isinstance(metadata[k], str)

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
    f_writer = AsyncFileWriter(path, n_entries=ASYNC_WRITE_ENTRIES, backend="pthread", n_tasks=2 + len(tensors))
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
    f_writer = AsyncFileWriter(path, n_entries=ASYNC_WRITE_ENTRIES, backend="pthread", n_tasks=2 + len(tensor_keys))
    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    f_writer.register_h2d(len(tensor_keys))
    for name in tensor_keys:
        if state_dict_pinned:
            f_writer.write_tensor(state_dict[name], state_dict_pinned[name])
        else:
            f_writer.write_tensor(state_dict[name])
    return f_writer


def load_flat(checkpoint_path, seperator: str = "."):
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    state_dict_load = load_file(checkpoint_path)
    state_dict = _unflatten_optim_state_dict(state_dict_load, metadata, seperator)
    return state_dict
