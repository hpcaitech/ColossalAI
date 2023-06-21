from typing import List

from torch._utils import _flatten_dense_tensors

from .base_store import BaseStore


class GradientStore(BaseStore):

    def __init__(self, *args, partition_grad=False):
        super().__init__(*args)

        self._grads_of_params = dict()
        self._working_index = 0 if partition_grad else self._local_rank

    def get_partitioned_gradients_by_param_id(self, group_id, param_id):
        if group_id in self._grads_of_params:
            if param_id in self._grads_of_params[group_id]:
                return self._grads_of_params[group_id][param_id]
        return []

    def append_gradients_by_param_id(self, grad, group_id, param_id):
        if group_id not in self._grads_of_params:
            self._grads_of_params[group_id] = dict()
        if param_id not in self._grads_of_params[group_id]:
            self._grads_of_params[group_id][param_id] = [grad]
        else:
            self._grads_of_params[group_id][param_id].append(grad)

    def add_gradients_by_param_id(self, grad, grad_idx, group_id, param_id):
        self._grads_of_params[group_id][param_id][grad_idx].add_(grad)

    def get_working_grads_by_group_id(self, group_id):
        grad_list = []
        for param_grads in self._grads_of_params[group_id].values():
            grad_list.append(param_grads[self._working_index])

        return grad_list

    def reset_grads_by_group_id(self, group_id):
        self._grads_of_params[group_id] = dict()

    def reset_all_gradients(self):
        self._grads_of_params = dict()
