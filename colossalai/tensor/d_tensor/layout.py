from dataclasses import dataclass

import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec


@dataclass
class Layout:
    """Layout of a tensor.

    Attributes:
        device_mesh: the device mesh to store the tensor distributedly.
        device_type: the type of the device mesh, e.g. 'cpu' or 'cuda'.
        sharding_spec: the sharding specification to describe how the tensor is sharded.
        entire_shape: the entire shape of the global tensor.
    """
    device_mesh: DeviceMesh
    device_type: torch.device
    sharding_spec: ShardingSpec
    entire_shape: torch.Size = None
