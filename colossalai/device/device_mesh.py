import operator
from functools import reduce
from typing import List, Tuple

import torch
import torch.distributed as dist


class DeviceMesh:
    """A logical view of a physical mesh. The logical view is used in the
    search process.
    A physical mesh can have multiple logical views. (e.g., a 2x8 physical mesh
    can be viewed as a 1x16 or a 4x4 logical mesh). Each mesh dimension has its
    own latency and bandwidth. We use alpha-beta model to model the
    communication cost.

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
        need_flatten(bool, optional): initialize flatten_device_mesh during initializing the DeviceMesh instance if the need_flatten set to True.
    """

    def __init__(self,
                 physical_mesh_id: torch.Tensor,
                 mesh_shape: torch.Size = None,
                 logical_mesh_id: torch.Tensor = None,
                 mesh_alpha: List[float] = None,
                 mesh_beta: List[float] = None,
                 init_process_group: bool = False,
                 need_flatten: bool = True):
        self.physical_mesh_id = physical_mesh_id
        if logical_mesh_id is None:
            self.mesh_shape = mesh_shape
            self._logical_mesh_id = self.physical_mesh_id.reshape(self.mesh_shape)
        else:
            self._logical_mesh_id = logical_mesh_id
            self.mesh_shape = self._logical_mesh_id.shape

        # map global rank into logical rank
        self.convert_map = {}
        self._global_rank_to_logical_rank_map(self._logical_mesh_id, [])
        # coefficient for alpha-beta communication model
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.mesh_shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.mesh_shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)
        self.init_process_group = init_process_group
        self.need_flatten = need_flatten
        if self.init_process_group:
            self.process_groups_dict = self.create_process_groups_for_logical_mesh()
        if self.need_flatten and self._logical_mesh_id.dim() > 1:
            self.flatten_device_mesh = self.flatten()
            # Create a new member `flatten_device_meshes` to distinguish from original flatten methods (Because I'm not sure if there are functions that rely on the self.flatten())
            # self.flatten_device_meshes = FlattenDeviceMesh(self.physical_mesh_id, self.mesh_shape, self.mesh_alpha,
            #                                                self.mesh_beta)

    @property
    def shape(self):
        return self.mesh_shape

    @property
    def num_devices(self):
        return reduce(operator.mul, self.physical_mesh_id.shape, 1)

    @property
    def logical_mesh_id(self):
        return self._logical_mesh_id

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'process_groups_dict':
                setattr(result, k, __import__("copy").deepcopy(v, memo))
            else:
                setattr(result, k, v)

        return result

    def flatten(self):
        """
        Flatten the logical mesh into an effective 1d logical mesh,
        """
        flatten_mesh_shape_size = len(self.mesh_shape)
        flatten_mesh_shape = [self.num_devices]
        return DeviceMesh(self.physical_mesh_id,
                          tuple(flatten_mesh_shape),
                          mesh_alpha=[max(self.mesh_alpha)] * (flatten_mesh_shape_size - 1),
                          mesh_beta=[min(self.mesh_beta)] * (flatten_mesh_shape_size - 1),
                          init_process_group=self.init_process_group,
                          need_flatten=False)

    def _global_rank_to_logical_rank_map(self, tensor, index_list):
        '''
        This method is a helper function to build convert_map recursively.
        '''
        for index, inner_tensor in enumerate(tensor):
            if inner_tensor.numel() == 1:
                self.convert_map[int(inner_tensor)] = index_list + [index]
            else:
                self._global_rank_to_logical_rank_map(inner_tensor, index_list + [index])

    def create_process_groups_for_logical_mesh(self):
        '''
        This method is used to initialize the logical process groups which will be used in communications
        among logical device mesh.
        Note: if init_process_group set to False, you have to call this method manually. Otherwise,
        the communication related function, such as ShapeConsistencyManager.apply will raise errors.
        '''
        process_groups_dict = {}
        check_duplicate_list = []
        global_rank_flatten_list = self.physical_mesh_id.view(-1).tolist()
        for global_rank in global_rank_flatten_list:
            process_groups = self.global_rank_to_process_groups_with_global_rank(global_rank)
            for axis, process_group in process_groups.items():
                if axis not in process_groups_dict:
                    process_groups_dict[axis] = []
                if process_group not in check_duplicate_list:
                    check_duplicate_list.append(process_group)
                    process_group_handler = dist.new_group(process_group)
                    process_groups_dict[axis].append((process_group, process_group_handler))

        return process_groups_dict

    def global_rank_to_logical_rank(self, rank):
        return self.convert_map[rank]

    def global_rank_to_process_groups_with_logical_rank(self, rank):
        '''
        Give a global rank and return all logical process groups of this rank.
        for example:
            physical_mesh_id = torch.arange(0, 16).reshape(2, 8)
            mesh_shape = (4, 4)
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7],
            #  [8, 9, 10,11],
            #  [12,13,14,15]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
            print(device_mesh.global_rank_to_process_groups_with_logical_rank(0))
        output:
            # key is axis name
            # value is a list of logical ranks in same axis with rank 0
            {0: [[0, 0], [1, 0], [2, 0], [3, 0]], 1: [[0, 0], [0, 1], [0, 2], [0, 3]]}
        '''
        process_groups = {}
        for d in range(self.logical_mesh_id.dim()):
            for replacer in range(self.logical_mesh_id.shape[d]):
                if d not in process_groups:
                    process_groups[d] = []
                process_group_member = self.convert_map[rank].copy()
                process_group_member[d] = replacer
                process_groups[d].append(process_group_member)
        return process_groups

    def global_rank_to_process_groups_with_global_rank(self, rank):
        '''
        Give a global rank and return all process groups of this rank.
        for example:
            physical_mesh_id = torch.arange(0, 16).reshape(2, 8)
            mesh_shape = (4, 4)
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7],
            #  [8, 9, 10,11],
            #  [12,13,14,15]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
            print(device_mesh.global_rank_to_process_groups_with_global_rank(0))
        output:
            # key is axis name
            # value is a list of global ranks in same axis with rank 0
            {0: [0, 4, 8, 12], 1: [0, 1, 2, 3]}
        '''
        logical_process_groups = self.global_rank_to_process_groups_with_logical_rank(rank)
        process_groups = {}
        for dim, logical_ranks in logical_process_groups.items():
            process_groups[dim] = []
            for logical_rank in logical_ranks:
                for g_rank, l_rank in self.convert_map.items():
                    if l_rank == logical_rank:
                        process_groups[dim].append(g_rank)
        return process_groups

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.1)

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes +
                0.01)

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
                0.001)

    def all_to_all_cost(self, num_bytes, mesh_dim):
        num_devices = self.logical_mesh_id.shape[mesh_dim]
        penalty_factor = num_devices / 2.0
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] *
                (num_devices - 1) / num_devices / num_devices * num_bytes * penalty_factor + 0.001)


class FlattenDeviceMesh(DeviceMesh):

    def __init__(self, physical_mesh_id, mesh_shape, mesh_alpha=None, mesh_beta=None):
        super().__init__(physical_mesh_id,
                         mesh_shape,
                         mesh_alpha,
                         mesh_beta,
                         init_process_group=False,
                         need_flatten=False)
        # Different from flatten(), mesh_shape leaves unchanged, mesh_alpha and mesh_beta are scalars
        self.mesh_alpha = max(self.mesh_alpha)
        self.mesh_beta = min(self.mesh_beta)
        # Different from original process_groups_dict, rank_list is not stored
        self.process_number_dict = self.create_process_numbers_for_logical_mesh()

    def create_process_numbers_for_logical_mesh(self):
        '''
        Build 1d DeviceMesh in column-major(0) and row-major(1)
        for example:
            mesh_shape = (2,4)
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7]]
            # return {0: [0, 4, 1, 5, 2, 6, 3, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7]}
        '''
        num_devices = reduce(operator.mul, self.mesh_shape, 1)
        process_numbers_dict = {}
        process_numbers_dict[0] = torch.arange(num_devices).reshape(self.mesh_shape).transpose(1, 0).flatten().tolist()
        process_numbers_dict[1] = torch.arange(num_devices).reshape(self.mesh_shape).flatten().tolist()
        return process_numbers_dict

    def mix_gather_cost(self, num_bytes):
        num_devices = reduce(operator.mul, self.mesh_shape, 1)
        return (self.mesh_alpha + self.mesh_beta * (num_devices - 1) / num_devices * num_bytes + 0.1)
