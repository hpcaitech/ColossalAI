from functools import reduce
import operator
import torch


class DeviceMesh:
    """A logical view of a physical mesh. The logical view is used in the
    search process.
    A physical mesh can have multiple logical views. (e.g., a 2x8 physical mesh
    can be viewed as a 1x16 or a 4x4 logical mesh). Each mesh dimension has its
    own latency and bandwidth. We use alpha-beta model to model the
    communication cost.
    
    Arguments:
        physical_mesh_id (torch.Tensor): physical view of the devices in global rank.
        mesh_shape (torch.Size): shape of logical view.
        mesh_alpha (List[float], optional): coefficients used for computing
            communication cost (default: None)
        mesh_beta (List[float], optional): coefficients used for computing
            communication cost (default: None)
    """

    def __init__(self, physical_mesh_id, mesh_shape, mesh_alpha=None, mesh_beta=None):
        self.physical_mesh_id = physical_mesh_id
        self.mesh_shape = mesh_shape
        self._logical_mesh_id = self.physical_mesh_id.reshape(self.mesh_shape)
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

    @property
    def shape(self):
        return self.mesh_shape

    @property
    def num_devices(self):
        return reduce(operator.mul, self.physical_mesh_id.shape, 1)

    @property
    def logical_mesh_id(self):
        return self._logical_mesh_id

    def _global_rank_to_logical_rank_map(self, tensor, index_list):
        '''
        This method is a helper function to build convert_map recursively.
        '''
        for index, inner_tensor in enumerate(tensor):
            if inner_tensor.numel() == 1:
                self.convert_map[int(inner_tensor)] = index_list + [index]
            else:
                self._global_rank_to_logical_rank_map(inner_tensor, index_list + [index])

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
