import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class BucketStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)

        # init and reset
        self.current_group_id = 0
        self.grad_to_param_mapping = dict()

        self._param_list = []
        self._padding_size = []

        self.reset()

    def num_elements_in_bucket(self):
        return self._num_elements_in_bucket

    def add_param_grad(self, group_id, param, padding_size):
        self._param_list.append(param)
        self._padding_size.append(padding_size)
        self._num_elements_in_bucket += (param.numel() + padding_size)
        self.current_group_id = group_id

    def build_grad_in_bucket(self):
        for param, padding_size in zip(self._param_list, self._padding_size):
            with torch.no_grad():
                grad = param.grad.view(-1)
                if padding_size > 0:
                    grad = torch.nn.function.pad(grad, [0, padding_size])
                grad_list = grad.split(grad.numel() // self._world_size)
                for rank in range(self._world_size):
                    grad_current_rank = grad_list[rank].detach()
                    self.grad_to_param_mapping[id(grad_current_rank)] = id(param)
                    self._grad_in_bucket[rank].append(grad_current_rank)
            param.grad = None

    def get_grad(self):
        return self._grad_in_bucket

    def get_flatten_grad(self):
        flat_grad = []
        for _, grad_list in self._grad_in_bucket.items():
            flat_grad.append(_flatten_dense_tensors(grad_list))
        flat_grad = _flatten_dense_tensors(flat_grad)
        return flat_grad

    def get_param_id_of_grad(self, grad):
        return self.grad_to_param_mapping[id(grad)]

    def reset(self):
        self.grad_to_param_mapping = dict()
        self._num_elements_in_bucket = 0
        self._param_list = []
        self._grad_in_bucket = dict()
        for rank in range(self._world_size):
            self._grad_in_bucket[rank] = []
