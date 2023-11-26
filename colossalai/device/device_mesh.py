"""This code is adapted from Alpa
    https://github.com/alpa-projects/alpa/
   with some changes. """

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


@dataclass
class ProcessGroupContainer:
    process_group: ProcessGroup
    ranks: List[int]


# modified from alpa LogicalDeviceMesh(https://github.com/alpa-projects/alpa/blob/main/alpa/shard_parallel/auto_sharding.py)
class DeviceMesh:
    """A logical view of a physical cluster. For example, we could view a physical cluster
    with 16 devices as a device mesh with shape (2, 2, 4) or (4, 4).

    Arguments:
        physical_mesh_id (torch.Tensor): physical view of the devices in global rank.
        logical_mesh_id (torch.Tensor): logical view of the devices in global rank.
        mesh_shape (torch.Size, optional): shape of logical view.
        mesh_alpha (List[float], optional): coefficients used for computing
            communication cost (default: None)
        mesh_beta (List[float], optional): coefficients used for computing
            communication cost (default: None)
        init_process_group (bool, optional): initialize logical process group
            during initializing the DeviceMesh instance if the init_process_group set to True.
            Otherwise, users need to call create_process_groups_for_logical_mesh manually to init logical process group.
            (default: False)
        device (str): the device for the process groups used by the DeviceMesh instance. (default: 'cuda')
    """

    _DIST_BACKEND = {"cuda": "nccl", "cpu": "gloo", "npu": "hccl"}

    def __init__(
        self,
        physical_mesh_id: torch.Tensor,
        mesh_shape: torch.Size = None,
        logical_mesh_id: torch.Tensor = None,
        mesh_alpha: List[float] = None,
        mesh_beta: List[float] = None,
        init_process_group: bool = False,
        device: str = "cuda",
    ):
        # ============================
        # Physical & Logical Mesh IDs
        # ============================
        self._physical_mesh_id = physical_mesh_id
        assert physical_mesh_id.dim() == 1, "physical_mesh_id should be a 1D tensor."

        # logical mesh ids can be obtained via two ways
        # 1. provide physical mesh id and provide mesh shape
        # 2. directly supply the logical mesh id
        assert mesh_shape is None or logical_mesh_id is None, (
            "Only one of mesh_shape and logical_mesh_id can be specified."
            "Logical mesh IDs are obtained from either mesh_shape + physical_mesh_id or directly from the user-supplied logical_mesh_id"
        )

        if logical_mesh_id is None:
            self._mesh_shape = mesh_shape
            self._logical_mesh_id = self._physical_mesh_id.reshape(self._mesh_shape)
        else:
            self._logical_mesh_id = logical_mesh_id
            self._mesh_shape = self._logical_mesh_id.shape

        # ensure two things:
        # 1. logical and physical mesh IDs should contain the same elements
        # 2. there is no duplicate IDs in each mesh, e.g. [2, 2] is not allowed
        assert torch.equal(
            torch.unique(self._physical_mesh_id), torch.unique(self.logical_mesh_id)
        ), "physical and logical mesh IDs should contain the same elements, please check if you have consistent physical_mesh_id and logical_mesh_id."
        assert (
            torch.unique(self._physical_mesh_id).numel() == self._physical_mesh_id.numel()
        ), "Found duplicate IDs in the physical_mesh_id and this is not allowed, please check your physical_mesh_id again."
        assert (
            torch.unique(self.logical_mesh_id).numel() == self.logical_mesh_id.numel()
        ), "Found duplicate IDs in the logical_mesh_id and this is not allowed, please check your logical_mesh_id again."

        # ===============================================
        # coefficient for alpha-beta communication model
        # alpha is latency and beta is bandwidth
        # ===============================================
        # if the values are not provided, we assume they are 1 for simplicity
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self._mesh_shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self._mesh_shape)

        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

        # ensure the alpha and beta have the same shape
        assert len(self.mesh_alpha) == len(
            self.mesh_beta
        ), "mesh_alpha and mesh_beta should have the same length, please check your mesh_alpha and mesh_beta again."

        # =========================
        # Device for Process Group
        # =========================
        self._device = device
        self._dist_backend = self._DIST_BACKEND[device]

        # =========================
        # Process Group Management
        # =========================
        # the _global_to_local_rank_mapping is structured as follows
        # {
        #    <global-rank>: [ <local-rank-on-axis-0>, <local-rank-on-axis-1>, <local-rank-on-axis-2>, ...]
        # }
        self._global_to_local_rank_mapping = dict()
        self._init_global_to_logical_rank_mapping(
            mapping=self._global_to_local_rank_mapping, tensor=self.logical_mesh_id
        )

        # create process group
        self._process_group_dict = {}
        self._ranks_in_the_process_group = {}
        self._global_rank_of_current_process = None
        self._is_initialized = False

        # attribute used to indicate whether this object
        # is created using DeviceMesh.from_process_group
        # this attribute can be used to do some check in methods
        # such get_process_group as no global rank information
        # is known if created with from_process_group
        self._is_init_from_process_group = False

        # initialize process group if specified
        self._init_ranks_in_the_same_group()
        self._init_process_group = init_process_group
        if init_process_group:
            self.init_logical_process_group()

    @property
    def shape(self) -> torch.Size:
        """
        Return the shape of the logical mesh.
        """
        return self._mesh_shape

    @property
    def num_devices(self) -> int:
        """
        Return the number of devices contained in the device mesh.
        """
        return reduce(operator.mul, self._physical_mesh_id.shape, 1)

    @property
    def logical_mesh_id(self) -> torch.Tensor:
        """
        Return the logical mesh id.
        """
        return self._logical_mesh_id

    @property
    def is_initialized(self) -> bool:
        """
        Return whether the process group is initialized.
        """
        return self._is_initialized

    @staticmethod
    def from_process_group(process_group: Union[ProcessGroup, List[ProcessGroup]]) -> "DeviceMesh":
        """
        Create a DeviceMesh instance from the current process group. Please note that the DeviceMesh object created with this method
        will not have information about the physical mesh id, and thus will not be able to query for other ranks and perform alpha-beta communication.

        Args:
            process_group (Union[ProcessGroup, List[ProcessGroup]]): the process group or a list of process groups for the device mesh.
                If the input is a ProcessGroup object, a 1D DeviceMesh object will be created. If the input is a list of ProcessGroup objects,
                the ProcessGroup at the ith index will correspond to the process group in the ith axis of the device mesh.

        Returns:
            DeviceMesh: the device mesh instance.
        """

        def _get_device_by_backend(process_group):
            """
            Get the device type given a process group's backend.
            """
            backend = dist.get_backend(process_group)
            for _device, _backend in DeviceMesh._DIST_BACKEND.items():
                if _backend == backend:
                    return _device
            return None

        if isinstance(process_group, ProcessGroup):
            process_group = [process_group]

        # get mesh shape
        mesh_shape = [dist.get_world_size(pg) for pg in process_group]

        # get device
        device_list = [_get_device_by_backend(pg) for pg in process_group]

        # make sure all devices are the same
        assert all(
            [device == device_list[0] for device in device_list]
        ), "All devices should be the same, please check your input process groups are created with the same distributed backend."

        # create a fake physical mesh id
        # as we only get the process group associated with the current process,
        # we cannot get the global ranks for all processes in the mesh
        # therefore, we only use this fake physical mesh id to create the device mesh
        # and will remove this fake physical mesh id later
        fake_physical_mesh_id = torch.arange(reduce(operator.mul, mesh_shape, 1))

        # create the device mesh
        device_mesh = DeviceMesh(physical_mesh_id=fake_physical_mesh_id, mesh_shape=mesh_shape, device=device_list[0])

        # hack the device attribute
        device_mesh._physical_mesh_id = None
        device_mesh._logical_mesh_id = None
        device_mesh._global_rank_of_current_process = dist.get_rank()
        device_mesh._is_initialized = False
        device_mesh._process_group_dict = {
            device_mesh._global_rank_of_current_process: {axis: pg for axis, pg in enumerate(process_group)}
        }

        return device_mesh

    def get_process_group(self, axis: int, global_rank: int = None) -> ProcessGroup:
        """
        Return the process group on the specified axis.

        Args:
            axis (int): the axis of the process group.
            global_rank (int, optional): the global rank of the process group. If not specified, the current process is used. (default: None)
        """
        if global_rank is None:
            global_rank = self._global_rank_of_current_process
        elif self._is_init_from_process_group:
            raise RuntimeError(
                "The logical device mesh is create with DeviceMesh.from_process_group, this method is not supported for this creation method as no global rank information is known."
            )
        return self._process_group_dict[global_rank][axis]

    def get_process_group_for_all_axes(self, global_rank: int = None) -> Dict[int, ProcessGroup]:
        """
        Return the process groups for all axes.

        Args:
            global_rank (int, optional): the global rank of the process
        """
        if global_rank is None:
            global_rank = self._global_rank_of_current_process
        elif self._is_init_from_process_group:
            raise RuntimeError(
                "The logical device mesh is create with DeviceMesh.from_process_group, this method is not supported for this creation method as no global rank information is known."
            )
        return self._process_group_dict[global_rank]

    def get_ranks_in_process_group(self, axis: int, global_rank: int = None) -> List[int]:
        """
        Return the ranks in the process group on the specified axis.

        Args:
            axis (int): the axis of the process group.
            global_rank (int, optional): the global rank of the process
        """
        if global_rank is None:
            global_rank = self._global_rank_of_current_process
        elif self._is_init_from_process_group:
            raise RuntimeError(
                "The logical device mesh is create with DeviceMesh.from_process_group, this method is not supported for this creation method as no global rank information is known."
            )
        return self._ranks_in_the_process_group[global_rank][axis]

    def __deepcopy__(self, memo) -> "DeviceMesh":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "_process_group_dict":
                setattr(result, k, __import__("copy").deepcopy(v, memo))
            else:
                # process group cannot be copied
                # thus, we share them directly
                setattr(result, k, v)
        return result

    def _init_global_to_logical_rank_mapping(
        self, mapping: Dict, tensor: torch.Tensor, index_list: List[int] = []
    ) -> Dict[int, List[int]]:
        """
        Build a global rank to local rank mapping for each process group in different axis in the logical device mesh.

        Args:
            mapping (Dict): a dictionary that maps the global rank to the local rank in the logical device mesh.
            tensor (torch.Tensor): the tensor that contains the logical mesh ids.
            index_list (List[int])

        Returns:
            mapping (Dict): a dictionary that maps the global rank to the local rank in the logical device mesh.
                The value is a list of integers and each integer represents the local rank in the indexed axis.
        """
        for index, inner_tensor in enumerate(tensor):
            # index means the local rank in the current axis
            # inner_tensor refers to the processes with the same local rank

            if inner_tensor.numel() == 1:
                # if the inner_tensor only has one element, it means that
                # it already reaches the last axis
                # we append its local_rank in the last axis to the index_list
                # and assign to the mapping
                # the value of the mapping is the the local rank at the indexed axis of the device mesh
                mapping[int(inner_tensor)] = index_list + [index]
            else:
                # we recursively go into the function until we reach the last axis
                # meanwhile, we should add the local rank in the current axis in the index_list
                self._init_global_to_logical_rank_mapping(mapping, inner_tensor, index_list + [index])

    def init_logical_process_group(self):
        """
        This method is used to initialize the logical process groups which will be used in communications
        among logical device mesh.
        Note: if init_process_group set to False, you have to call this method manually. Otherwise,
        the communication related function, such as ShapeConsistencyManager.apply will raise errors.
        """
        # sanity check
        assert (
            dist.is_initialized
        ), "The torch.distributed should be initialized before calling init_logical_process_group"
        assert (
            not self._is_initialized
        ), "The logical process group has been initialized, do not call init_logical_process_group twice"

        # update the global rank of the current process
        self._global_rank_of_current_process = dist.get_rank()
        duplicate_check_list = []

        # flatten the global ranks to 1D list
        global_rank_flatten_list = self._physical_mesh_id.view(-1).tolist()

        for global_rank in global_rank_flatten_list:
            # find the other ranks which are in the same process group as global_rank
            ranks_in_same_group_by_axis = self._collate_global_ranks_in_same_process_group(global_rank)

            for axis, ranks_in_same_group in ranks_in_same_group_by_axis.items():
                # skip duplicated process group creation
                if ranks_in_same_group in duplicate_check_list:
                    continue

                # create the process group
                pg_handler = dist.new_group(ranks=ranks_in_same_group, backend=self._dist_backend)

                # keep this process group in the process_groups_dict
                for rank in ranks_in_same_group:
                    if rank not in self._process_group_dict:
                        self._process_group_dict[rank] = dict()
                    self._process_group_dict[rank][axis] = pg_handler

        # update the init flag
        # we only allow init for once
        self._is_initialized = True

    def _init_ranks_in_the_same_group(self):
        """
        This method is used to initialize the ranks_in_the_same_group dictionary.
        """
        # flatten the global ranks to 1D list
        global_rank_flatten_list = self._physical_mesh_id.view(-1).tolist()

        for global_rank in global_rank_flatten_list:
            # find the other ranks which are in the same process group as global_rank
            ranks_in_same_group_by_axis = self._collate_global_ranks_in_same_process_group(global_rank)

            for axis, ranks_in_same_group in ranks_in_same_group_by_axis.items():
                # create dict for each rank
                if global_rank not in self._process_group_dict:
                    self._ranks_in_the_process_group[global_rank] = dict()

                # keep this process group in the process_groups_dict
                self._ranks_in_the_process_group[global_rank][axis] = ranks_in_same_group

    def global_rank_to_local_rank(self, rank: int, axis: int = None) -> Union[List[int], int]:
        """
        Return the local rank of the given global rank in the logical device mesh.

        Args:
            rank (int): the global rank in the logical device mesh.
            axis (int): the axis of the logical device mesh.
        """
        if self._is_init_from_process_group:
            raise RuntimeError(
                "The logical device mesh is create with DeviceMesh.from_process_group, this method is not supported for this creation method as no global rank information is known."
            )

        local_ranks = self._global_to_local_rank_mapping[rank]
        if axis:
            return local_ranks[axis]
        else:
            return local_ranks

    def _collate_global_ranks_in_same_process_group(self, global_rank):
        """
        Give a global rank and return all global ranks involved in its associated process group in each axis.

        Example:

        ```python
        physical_mesh_id = torch.arange(0, 16)
        mesh_shape = (4, 4)

        # logical mesh will look like
        # [[0, 1, 2, 3],
        #  [4, 5, 6, 7],
        #  [8, 9, 10,11],
        #  [12,13,14,15]]

        device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
        print(device_mesh.collate_global_ranks_in_same_process_group(0))

        # key is axis name
        # value is a list of global ranks in same axis with rank 0
        # output will look like
        # {
            0: [0, 4, 8, 12],
            1: [0, 1, 2, 3]
        #  }
        """
        # We have init the global rank to local rank by calling _init_global_to_logical_rank_mapping
        # for self._global_to_local_rank_mapping
        # the key is the global rank
        # the value is the list of local ranks corresponding to the global rank with respect of different axes
        # we can see the list of local ranks as the process coordinates for simplicity
        # the key and value are all unique, therefore,
        # we can also to use the coordinates to find the global rank

        # =========================================================================
        # Step 1
        # find all the process_coordinates for processes in the same process group
        # as the given global rank
        # =========================================================================

        # each
        processes_in_the_same_process_group = {}

        for dim in range(self.logical_mesh_id.dim()):
            # iterate over the dimension size so that we can include all processes
            # in the same process group in the given axis
            # the _local_rank refers to the local rank of the current process
            for _local_rank in range(self.logical_mesh_id.shape[dim]):
                # if this dimension is not initialized yet,
                # initialize it with an empty array
                if dim not in processes_in_the_same_process_group:
                    processes_in_the_same_process_group[dim] = []

                # get the local rank corresponding to the global rank
                process_coordinates = self._global_to_local_rank_mapping[global_rank].copy()

                # replace the local rank in the given dimension with the
                # local rank of the current process iterated
                process_coordinates[dim] = _local_rank
                processes_in_the_same_process_group[dim].append(process_coordinates)

        # =================================================================
        # Step 2
        # Use local rank combination to find its corresponding global rank
        # =================================================================
        # the key of the dict is the axis
        # the value is the list of global ranks which are in the same process group as the given global rank
        global_pg_ranks = {}
        for dim, coordinates_of_all_processes in processes_in_the_same_process_group.items():
            global_pg_ranks[dim] = []
            for process_coordinates in coordinates_of_all_processes:
                # find the global rank by local rank combination
                for _global_rank, _process_coordinates in self._global_to_local_rank_mapping.items():
                    if process_coordinates == _process_coordinates:
                        global_pg_ranks[dim].append(_global_rank)
        return global_pg_ranks

    def flatten(self):
        """
        Flatten the logical mesh into an effective 1d logical mesh,
        """
        if self._is_init_from_process_group:
            raise RuntimeError(
                "The logical device mesh is create with DeviceMesh.from_process_group, this method is not supported for this creation method as no global rank information is known."
            )

        flatten_mesh_shape_size = len(self._mesh_shape)
        flatten_mesh_shape = [self.num_devices]
        return DeviceMesh(
            self._physical_mesh_id,
            tuple(flatten_mesh_shape),
            mesh_alpha=[max(self.mesh_alpha)] * (flatten_mesh_shape_size - 1),
            mesh_beta=[max(self.mesh_beta)] * (flatten_mesh_shape_size - 1),
            init_process_group=self._init_process_group,
        )

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes + 0.1

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes
            + 0.01
        )

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return (
            self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes + 0.001
        )

    def all_to_all_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        penalty_factor = num_devices / 2.0
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices / num_devices * num_bytes * penalty_factor
            + 0.001
        )
