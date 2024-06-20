from typing import Dict

import torch
from torch import Tensor
from torch._utils import _flatten_dense_tensors
from torch.distributed import ProcessGroup

from colossalai.accelerator.api import get_accelerator

from .base_store import BaseStore


class BucketStore(BaseStore):
    def __init__(
        self,
        torch_pg: ProcessGroup,
        reduce_bucket_size: int,
    ):
        super().__init__(torch_pg)
        self.reduce_bucket_size = reduce_bucket_size
        self.reset_all()
        self.comm_stream = get_accelerator().Stream()

    def reset_all(self) -> None:
        # init
        self.current_group_id = 0
        self._num_elements_in_bucket = 0
        # mapping gradient slices and parameter
        self.grad_to_param_mapping = dict()

        self._grad_in_bucket = dict()
        self._param_list = []
        self._padding_size = []
        for rank in range(self._world_size):
            self._grad_in_bucket[rank] = []

        # offset_list records number of tensors in the bucket before each reduction
        self.offset_list = [0]

    def num_elements_in_bucket(self) -> int:
        """Return the total number of elements in bucket

        Returns:
            int: the total number of elements in bucket
        """

        return self._num_elements_in_bucket

    def reset_num_elements_in_bucket(self):
        """Set the number of elements in bucket to zero."""

        self._num_elements_in_bucket = 0

    def add_param_grad(self, group_id: int, param: Tensor, padding_size: int):
        """Add a param to bucket and record the padding size of a param for gradient padding

        Args:
            group_id (int): The index of a parameter group
            param (Tensor): The parameter
            padding_size (int): The padding size of the parameter
        """

        self._param_list.append(param)
        self._padding_size.append(padding_size)
        self._num_elements_in_bucket += param.numel() + padding_size
        self.current_group_id = group_id

        # number of tensors in current bucket
        self.offset_list[-1] += 1

    def build_grad_in_bucket(self):
        """Organize parameters' gradient(padding and split), follows the parameters' splitting method

        Data structure of self._grad_in_bucket:
        {
        rank0: [grad0_rank0, grad1_rank0, ...]
        rank1: [grad0_rank1, grad1_rank1, ...]
        }
        """
        for param, padding_size in zip(self._param_list, self._padding_size):
            grad = param.grad.clone().detach().flatten()
            if padding_size > 0:
                with torch.no_grad():
                    grad = torch.nn.functional.pad(grad.view(-1), [0, padding_size])
            grad_list = grad.split(grad.numel() // self._world_size)
            for rank in range(self._world_size):
                grad_current_rank = grad_list[rank].clone().detach()
                self.grad_to_param_mapping[id(grad_current_rank)] = id(param)
                self._grad_in_bucket[rank].append(grad_current_rank)
            param.grad = None

        self.offset_list.append(0)

    def get_grad(self) -> Dict:
        """Return the dictionary of gradients slices, of which the keys are ranks

        Returns:
            Dict: The dictionary of gradients slices
        """

        return self._grad_in_bucket

    def get_flatten_grad(self) -> Tensor:
        """Return the flattened gradients slices in the bucket, the data organization of the flattened tensor:
        [grad0_rank0, grad1_rank0, ..., grad_0_rank1, grad1_rank1, ....]

        Returns:
            Tensor: the flattened gradients slices in the bucket
        """

        flat_grad = []
        for grad_list in self._grad_in_bucket.values():
            flat_grad.append(_flatten_dense_tensors(grad_list))
        flat_grad = _flatten_dense_tensors(flat_grad)
        return flat_grad

    def get_param_id_of_grad(self, grad: Tensor) -> int:
        """Return the id of a parameter which the gradient slice belongs to

        Args:
            grad (Tensor): the gradient slice

        Returns:
            int: the id of a parameter which the gradient slice belongs to
        """

        return self.grad_to_param_mapping[id(grad)]

    def reset(self):
        """Reset the bucket storage after reduction, only release the tensors have been reduced"""
        cur_offset = self.offset_list.pop(0)
        self._param_list = self._param_list[cur_offset:]
        self._padding_size = self._padding_size[cur_offset:]
        for _ in range(cur_offset):
            del self.grad_to_param_mapping[next(iter(self.grad_to_param_mapping))]
        for rank in range(self._world_size):
            self._grad_in_bucket[rank] = self._grad_in_bucket[rank][cur_offset:]
