from tokenize import group
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from .._utils import unflatten


class ParameterStore:

    def __init__(self):
        self._partition_count = gpc.get_world_size(ParallelMode.DATA)
        self._partition_id = gpc.get_local_rank(ParallelMode.DATA)
        self._fp16_groups_flat = dict()
        self._params_not_in_partition = dict()
        self._params_in_partition = dict()
        self._partition_size = dict()
        self._first_offset = dict()
        self._reordered_fp16_groups = dict()
        self._reordered_fp16_indices = dict()
        self._data_parallel_partition_fp16_param = dict()
        self._group_paddings = dict()

    @property
    def partition_count(self):
        return self._partition_count

    @property
    def partition_id(self):
        return self.partition_id

    @property
    def num_param_groups(self):
        return len(self._fp16_groups_flat)

    def is_param_in_current_partition(self, param):
        pass

    def get_partition_size(self, partition_id):
        pass

    def get_first_offset(self, partition_id):
        pass

    def add_group_padding(self, group_id, padding):
        self._group_paddings[group_id] = padding

    def get_group_padding(self, group_id):
        return self._group_paddings[group_id]

    def add_reordered_fp16_params(self, group_id, tensor_list, indices):
        self._reordered_fp16_groups[group_id] = tensor_list
        self._reordered_fp16_indices[group_id] = indices

    def get_reordered_fp16_params(self, group_id):
        tensor_list = self._reordered_fp16_groups[group_id]
        indices = self._reordered_fp16_indices[group_id]
        return tensor_list, indices

    def add_flat_fp16_param(self, group_id, tensor):
        self._fp16_groups_flat[group_id] = tensor

    def get_flat_fp16_param(self, group_id):
        return self._fp16_groups_flat[group_id]

    def sync_fp16_param(self, group_id, fp16_params):
        fp16_flat_param = self.get_flat_fp16_param(group_id=group_id)
        reordered_fp16_params, index_mapping = self.get_reordered_fp16_params(
            group_id=group_id)

        updated_params = unflatten(fp16_flat_param, reordered_fp16_params)

        # update the tensor data
        # LSG: TODO: check if this is necessary
        for p, q in zip(reordered_fp16_params, updated_params):
            p.data = q.data

        # update the parameters in fp16 groups
        for idx, param in enumerate(fp16_params):
            new_idx = index_mapping[idx]
            param.data = reordered_fp16_params[new_idx].data

    def partition_flat_fp16_param(self, group_id):
        flat_param = self.get_flat_fp16_param(group_id)
        partitions = []
        num_elements = flat_param.numel()

        assert num_elements % self._partition_count == 0, 'the flat tensor has incorrect padding'
        partition_size = num_elements // self._partition_count

        for partition_id in range(self._partition_count):
            start = partition_id * partition_size
            end = (partition_id + 1) * partition_size
            slice_tensor = flat_param.narrow(0, start, end)
            partitions.append(slice_tensor)

        return partitions

    def add_data_parallel_partition(self, group_id, partition_id, tensor):
        if group_id not in self._data_parallel_partition_fp16_param:
            self._data_parallel_partition_fp16_param[group_id] = dict()

        param_group = self._data_parallel_partition_fp16_param[group_id]
        param_group[partition_id] = tensor

    def get_data_parallel_partition(self, group_id, partition_id):
        return self._data_parallel_partition_fp16_param[group_id][partition_id]

    def update_partition_size(self, group_id):
        fp16_flat_tensor = self.get_flat_fp16_param(group_id)
        num_elements = fp16_flat_tensor.numel()
        partition_size = num_elements // self._partition_count
        self._partition_size[group_id] = partition_size

    def get_partition_size(self, group_id):
        return self._partition_size[group_id]

    def update_partition_ownership(self, group_id, partition_id):
        # calculate partition size
        partition_size = self.get_partition_size(group_id)
        tensor_list = self.get_reordered_fp16_params(group_id)

        # init data structure
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        def _is_head_in_partition():
            return current_index >= start_index and current_index < end_index

        def _is_tail_in_partition(tensor_size):
            return start_index > current_index and \
                start_index < (current_index + tensor_size)

        for tensor in tensor_list:
            tensor_size = tensor.numel()

            if _is_head_in_partition():
                params_in_partition.append(tensor)
            elif _is_tail_in_partition(tensor_size):
                params_in_partition.append(tensor)
                assert (first_offset == 0), \
                    "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index
            else:
                params_not_in_partition.append(tensor)
            current_index = current_index + tensor_size

        self._params_in_partition[group_id] = params_in_partition
        self._params_not_in_partition[group_id] = params_not_in_partition
