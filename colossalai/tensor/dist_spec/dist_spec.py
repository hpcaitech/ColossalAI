from enum import Enum
from torch.distributed import ProcessGroup
from typing import Optional


class DistPlacementPattern(Enum):
    REPLICATE = 'r'
    SHARD = 's'


class _DistSpec:

    def __init__(self,
                 dist_placement_pattern: DistPlacementPattern,
                 process_group: Optional[ProcessGroup] = None,
                 **meta_info):
        self.placement = dist_placement_pattern
        self.process_group = process_group
        for k, v in meta_info.items():
            setattr(self, k, v)

    def __eq__(self, other: "_DistSpec") -> bool:
        if dir(self) != dir(other):
            return False
        for attr in dir(self):
            if not attr.startswith('__') and getattr(self, attr) != getattr(other, attr):
                return False
        return True
