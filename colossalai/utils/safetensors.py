# a python safetensors serializer modified from https://github.com/huggingface/safetensors/blob/41bd1acf38ad28ac559522d40596c6c802f79453/safetensors/src/tensor.rs#L214
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from safetensors.torch import _TYPES

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


def prepare(data: Dict[str, torch.Tensor]) -> Tuple[PreparedData, List[torch.Tensor]]:
    sorted_data = sorted(data.items(), key=lambda x: (x[1].dtype, x[0]))

    tensors = []
    metadata = {}
    offset = 0

    for name, tensor in sorted_data:
        n = tensor.numel() * tensor.element_size()
        tensor_info = TensorInfo(
            dtype=_TYPES_INV[tensor.dtype], shape=list(tensor.shape), data_offsets=(offset, offset + n)
        )
        offset += n
        metadata[name] = asdict(tensor_info)
        tensors.append(tensor)

    metadata_buf = json.dumps(metadata).encode("utf-8")

    extra = (8 - len(metadata_buf) % 8) % 8
    metadata_buf += b" " * extra

    n = len(metadata_buf)

    return PreparedData(n=n, header_bytes=metadata_buf, offset=offset), tensors


def save(f_writer: AsyncFileWriter, state_dict: Dict[str, torch.Tensor]) -> None:
    prepared_data, tensors = prepare(state_dict)
    n, header_bytes, _ = prepared_data.n, prepared_data.header_bytes, prepared_data.offset

    f_writer.write(n.to_bytes(8, byteorder="little"))
    f_writer.write(header_bytes)

    for tensor in tensors:
        f_writer.write_raw(tensor, tensor.data_ptr(), tensor.numel() * tensor.element_size(), f_writer.offset)
