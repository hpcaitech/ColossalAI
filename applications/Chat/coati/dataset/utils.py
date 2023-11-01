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


def read_string_by_schema(data, schema: str):
    """
    Read a feild of the dataset be schema
    Args:
        data: List[Any]
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
