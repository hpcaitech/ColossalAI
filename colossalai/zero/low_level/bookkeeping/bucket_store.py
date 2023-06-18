import torch
from torch._utils import _flatten_dense_tensors
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)

        # init and reset
        self.current_group_id = 0

        self.reset()

    def num_elements_in_bucket(self):
        return self._num_elements_in_bucket

    def add_param_grad(self, group_id, grad, padding_size):
        self.current_group_id = group_id
        with torch.no_grad():
            if padding_size > 0:
                grad = torch.nn.functional.pad(grad.view(-1), [0, padding_size])
            else:
                grad = grad.view(-1)
            self._num_elements_in_bucket += grad.numel()
            grad_list = grad.split(grad.numel() // self._world_size)
            for rank in range(self._world_size):
                self._grad_in_bucket[rank].append(grad_list[rank])

    def get_grad(self):
        return self._grad_in_bucket

    def flatten_grad(self):
        for rank, grad_list in self._grad_in_bucket.items():
            self._grad_in_bucket[rank] = _flatten_dense_tensors(grad_list)

    def reset(self):
        self._num_elements_in_bucket = 0
        self._grad_in_bucket = dict()
        for rank in range(self._world_size):
            self._grad_in_bucket[rank] = []
