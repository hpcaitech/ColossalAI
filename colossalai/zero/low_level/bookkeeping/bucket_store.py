import torch
from torch._utils import _flatten_dense_tensors
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)

        # init and reset
        self.current_group_id = 0
        self.grad_to_param_mapping = dict()

        self.reset()

    def num_elements_in_bucket(self):
        return self._num_elements_in_bucket

    def add_param_grad(self, group_id, param, grad, padding_size):
        self.current_group_id = group_id
        with torch.no_grad():
            if padding_size > 0:
                grad = torch.nn.functional.pad(grad.view(-1), [0, padding_size])
            else:
                grad = grad.view(-1)
            self._num_elements_in_bucket += grad.numel()
            grad_list = grad.split(grad.numel() // self._world_size)
            for rank in range(self._world_size):
                self.grad_to_param_mapping[id(grad_list[rank])] = id(param)
                self._grad_in_bucket[rank].append(grad_list[rank])

    def get_grad(self):
        return self._grad_in_bucket

    def get_param_id_of_grad(self, grad):
        return self.grad_to_param_mapping[id(grad)]

    def get_flatten_grad(self):
        flat_grad = []
        for rank, grad_list in self._grad_in_bucket.items():
            flat_grad.append(_flatten_dense_tensors(grad_list))
        flat_grad = _flatten_dense_tensors(flat_grad)
        return flat_grad

    def unflatten_grad(self, flat_grad):
        grad_list = flat_grad.split(len(flat_grad) // self._world_size)

    def reset(self):
        self.grad_to_param_mapping = dict()
        self._num_elements_in_bucket = 0
        self._grad_in_bucket = dict()
        for rank in range(self._world_size):
            self._grad_in_bucket[rank] = []
