from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist

from colossalai.device.alpha_beta_profiler import AlphaBetaProfiler
from colossalai.device.device_mesh import DeviceMesh


@dataclass
class DeviceMeshInfo:
    """
    This class is used to store the information used to initialize the device mesh.

    Args:
        physical_ids (List[int]): The physical ids of the current booster. For example, if we have the last 4 GPUs on a 8-devices cluster, then the physical ids should be [4, 5, 6, 7].
        mesh_shapes (List[Union[torch.Size, List[int], Tuple[int]]]): The shape of the mesh. For example, if we have 4 GPUs and we want to use 2D mesh with mesh shape [2, 2], then the mesh shape should be [2, 2].
    """

    physical_ids: List[int]
    mesh_shape: Union[torch.Size, List[int], Tuple[int]] = None

    def __post_init__(self):
        if self.mesh_shape is not None:
            world_size = len(self.physical_ids)
            mesh_shape_numel = torch.Size(self.mesh_shape).numel()
            assert (
                world_size == mesh_shape_numel
            ), f"the numel of mesh_shape should be equal to world size, but got {world_size} != {mesh_shape_numel}"


def initialize_device_mesh(device_mesh_info: DeviceMeshInfo):
    """
    This method is used to initialize the device mesh.

    Args:
        device_mesh_info (DeviceMeshInfo): The information used to initialize device mesh.
    """
    # parse the device mesh info
    physical_devices = device_mesh_info.physical_ids
    physical_mesh = torch.tensor(physical_devices)
    logical_mesh_shape = device_mesh_info.mesh_shape

    if logical_mesh_shape is None:
        ab_profiler = AlphaBetaProfiler(physical_devices)
        # search for the best logical mesh shape
        logical_mesh_id = ab_profiler.search_best_logical_mesh()
        logical_mesh_id = torch.Tensor(logical_mesh_id).to(torch.int)

    else:
        logical_mesh_id = physical_mesh.reshape(logical_mesh_shape)

    device_mesh = DeviceMesh(physical_mesh_id=physical_mesh, logical_mesh_id=logical_mesh_id, init_process_group=True)
    return device_mesh


class DeviceMeshManager:
    """
    Device mesh manager is responsible for creating and managing device meshes.
    """

    def __init__(self):
        self.device_mesh_store: Dict[str, DeviceMesh] = dict()

    def create_device_mesh(self, name, device_mesh_info: DeviceMeshInfo) -> DeviceMesh:
        """
        Create a device mesh and store it in the manager.

        Args:
            name (str): name of the device mesh
            device_mesh_info (DeviceMeshInfo): the information used to initialize the device mesh
        """
        if name not in self.device_mesh_store:
            device_mesh = initialize_device_mesh(device_mesh_info)
            self.device_mesh_store[name] = device_mesh
            return device_mesh
        else:
            raise ValueError(f"Device mesh {name} already exists.")

    def get(self, name: str) -> DeviceMesh:
        """
        Get a device mesh by name.

        Args:
            name (str): name of the device mesh

        Returns:
            DeviceMesh: the device mesh
        """
        if name in self.device_mesh_store:
            return self.device_mesh_store[name]
        else:
            raise ValueError(f"Device mesh {name} does not exist.")

    def destroy(self, name: str) -> None:
        """
        Destroy a device mesh by name.

        Args:
            name (str): name of the device mesh
        """
        if name in self.device_mesh_store:
            for pgs in self.device_mesh_store[name].process_groups_dict.values():
                for pg in pgs:
                    dist.destroy_process_group(pg)
            del self.device_mesh_store[name]
        else:
            raise ValueError(f"Device mesh {name} does not exist.")

    def destroy_all(self):
        """
        Destroy all device meshes.
        """
        for name in self.device_mesh_store:
            for pgs in self.device_mesh_store[name].process_groups_dict.values():
                for pg in pgs:
                    dist.destroy_process_group(pg)

        self.device_mesh_store.clear()
