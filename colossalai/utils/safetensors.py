# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import _TYPES, load_file, safe_open

try:
    from tensornvme.async_file_io import AsyncFileWriter
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")
_TYPES_INV = {v: k for k, v in _TYPES.items()}


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


# class TupleEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, tuple):
#             return {"__tuple__": True, "items": list(obj)}
#         return super().default(obj)


# def tuple_decoder(d):
#     if "__tuple__" in d:
#         return tuple(d["items"])
#     return d
# 自定义 JSON 编码器，处理 tuple
class NestedTupleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return {"__tuple__": True, "items": list(obj)}
        return super().default(obj)


# 自定义解码器，处理 tuple
def nested_tuple_decoder(d):
    if "__tuple__" in d:
        return tuple(d["items"])
    return d


def flatten_dict(nested_dict, parent_key="", separator="^"):
    """
    Flatten a nested dictionary, generating a flattened dictionary where the keys are joined by the specified separator.

    nested_dict: The input nested dictionary.
    parent_key: The parent key currently being processed.
    separator: The separator used to join keys, default is '_', but can be customized to another symbol. :return: A flattened dictionary."
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            v = torch.tensor(v, dtype=torch.float16) if not isinstance(v, torch.Tensor) else v
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(flattened_dict, separator="^"):
    """
    Restore a flattened dictionary back to a multi-level nested dictionary.

    flattened_dict: The flattened dictionary.
    separator: The separator used during flattening, default is '_', but can be customized to another symbol. :return: The restored nested dictionary.
    """
    nested_dict = {}
    for key, value in flattened_dict.items():
        keys = key.split(separator)
        try:
            keys[0] = int(keys[0])
        except ValueError:
            warnings.warn(f"{key[0]} can't convert to integer")
        d = nested_dict
        for part in keys[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        assert isinstance(value, torch.Tensor)
        d[keys[-1]] = value

    return nested_dict


def prepare(
    data: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> Tuple[PreparedData, List[torch.Tensor], List[str]]:
    if metadata is not None:
        assert isinstance(metadata, dict)
        for k, v in metadata.items():
            metadata[k] = json.dumps(v, cls=NestedTupleEncoder)
            print("metadata[k]", type(metadata[k]))
            assert isinstance(k, str)
            assert isinstance(metadata[k], str)

    tensors = []
    tensor_keys = []
    header = {}
    offset = 0

    if metadata is not None:
        header["__metadata__"] = metadata

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


def save(
    f_writer: AsyncFileWriter, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> None:
    prepared_data, tensors, _ = prepare(state_dict, metadata)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset

    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    for tensor in tensors:
        f_writer.write_raw(tensor, tensor.data_ptr(), tensor.numel() * tensor.element_size(), f_writer.offset)


def save_nested(
    f_writer: AsyncFileWriter, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> None:
    flatten_data = flatten_dict(state_dict)
    save(f_writer, flatten_data, metadata)


def move_and_save(
    f_writer: AsyncFileWriter,
    state_dict: Dict[str, torch.Tensor],
    state_dict_pinned: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    prepared_data, _, tensor_keys = prepare(state_dict)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset

    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    f_writer.register_h2d(len(tensor_keys))
    for name in tensor_keys:
        if state_dict_pinned:
            f_writer.write_tensor(state_dict[name], state_dict_pinned[name])
        else:
            f_writer.write_tensor(state_dict[name])


def load_flat(checkpoint_path):
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    state_dict_load = load_file(checkpoint_path)
    state_dict = unflatten_dict(state_dict_load)
    if metadata is None:
        return state_dict
    print("metadata", metadata)
    metadata = dict(
        map(lambda item: (item[0], json.loads(item[1], object_hook=nested_tuple_decoder)), metadata.items())
    )
    # metadata = json.loads(metadata, object_hook=tuple_decoder)
    combined_state_dict = {"state": state_dict}
    combined_state_dict.update(metadata)
    return combined_state_dict
