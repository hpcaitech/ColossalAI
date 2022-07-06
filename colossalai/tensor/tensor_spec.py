from typing import Optional
from colossalai.tensor.distspec import _DistSpec, DistPlacementPattern
from .compute_spec import ComputeSpec
from colossalai.tensor import ProcessGroup
from dataclasses import dataclass


@dataclass
class ColoTensorSpec:
    pg: ProcessGroup
    dist_attr: Optional[_DistSpec] = _DistSpec(DistPlacementPattern.REPLICATE)
    compute_attr: Optional[ComputeSpec] = None
