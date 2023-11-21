import io
import json
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def read_string_by_schema(data: Dict[str, Any], schema: str) -> str:
    """
    Read a feild of the dataset be schema
    Args:
        data: Dict[str, Any]
        schema: cascaded feild names seperated by '.'. e.g. person.name.first will access data['person']['name']['first']
    """
    keys = schema.split(".")
    result = data
    for key in keys:
        result = result.get(key, None)
        if result is None:
            return ""
    assert isinstance(result, str), f"dataset element is not a string: {result}"
    return result


def pad_to_max_len(
    sequence: List[torch.Tensor], max_length: int, padding_value: int, batch_first: bool = True, padding_side="left"
):
    """
    Args:
        sequence: a batch of tensor of shape [batch_size, seq_len] if batch_first==True
    """
    if padding_side == "left":
        reversed_sequence = [seq.flip(dims=(0,)) for seq in sequence]
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences=reversed_sequence, batch_first=batch_first, padding_value=padding_value
        )
        to_pad = max_length - padded.size(1)
        padded = F.pad(padded, (0, to_pad), value=padding_value)
        return torch.flip(padded, dims=(1,))
    elif padding_side == "right":
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences=sequence, batch_first=batch_first, padding_value=padding_value
        )
        to_pad = max_length - padded.size(1)
        return F.pad(padded, (0, to_pad), value=padding_value)
    else:
        raise RuntimeError(f"`padding_side` can only be `left` or `right`, " f"but now `{padding_side}`")


def chuncate_sequence(sequence: List[torch.Tensor], max_length: int, dtype: Any):
    """
    Args:
        sequence: a batch of tensor of shape [batch_size, seq_len] if batch_first==True
    """
    return [
        torch.Tensor(seq[:max_length]).to(dtype) if len(seq) > max_length else torch.Tensor(seq).to(dtype)
        for seq in sequence
    ]
