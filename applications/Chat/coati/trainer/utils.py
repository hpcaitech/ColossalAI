from typing import Any

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def to_device(x: Any, device: torch.device) -> Any:

    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)
