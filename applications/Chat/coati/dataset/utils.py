import io
import json
from typing import Dict, List

import numpy as np
import torch
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


def to_tensor(d: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.array(v)) for k, v in d.items()}
