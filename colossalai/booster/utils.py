from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from colossalai.device.alpha_beta_profiler import AlphaBetaProfiler
from colossalai.device.device_mesh import DeviceMesh


def initialize_device_mesh(world_size: int = -1,
                           physical_devices: List[int] = None,
                           logical_mesh_shape: Tuple[int] = None,
                           logical_mesh_id: torch.Tensor = None):
    '''
    This method is used to initialize the device mesh.

    Args:
        world_size: the size of device mesh. If the world_size is -1,
            the world size will be set to the number of GPUs in the current machine.
        physical_devices: the physical devices used to initialize the device mesh.
        logical_mesh_shape(optional): the logical_mesh_shape is used to specify the logical
            mesh shape.
        logical_mesh_id(optional): the logical_mesh_id is used to specify the logical mesh id.
    '''
    # if world_size is not set, use the world size from torch.distributed
    if world_size == -1:
        world_size = dist.get_world_size()

    if physical_devices is None:
        physical_devices = [i for i in range(world_size)]
    physical_mesh = torch.tensor(physical_devices)

    if logical_mesh_shape is None and logical_mesh_id is None:
        ab_profiler = AlphaBetaProfiler(physical_devices)
        # search for the best logical mesh shape
        logical_mesh_id = ab_profiler.search_best_logical_mesh()
        logical_mesh_id = torch.Tensor(logical_mesh_id).to(torch.int)
        logical_mesh_shape = logical_mesh_id.shape

    elif logical_mesh_shape is not None and logical_mesh_id is None:
        logical_mesh_id = physical_mesh.reshape(logical_mesh_shape)

    device_mesh = DeviceMesh(physical_mesh_id=physical_mesh, logical_mesh_id=logical_mesh_id, init_process_group=True)
    return device_mesh
