import io
import json

import torch.distributed as dist


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


def read_data_by_schema(data, schema: str):
    keys = schema.split(".")
    result = data
    for key in keys:
        result = result.get(key)
        if result is None:
            return None
    return result


def read_string_by_schema(data, schema: str):
    ret = read_data_by_schema(data, schema)
    if ret:
        return ret
    else:
        return ""
