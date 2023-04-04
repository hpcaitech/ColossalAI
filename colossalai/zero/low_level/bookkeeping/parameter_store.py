from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class ParameterStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)
        # param partitioning data structures
        self._fp16_param_to_rank = dict()
        self._rank_groupid_to_fp16_param_list = dict()
        self._rank_group_id_to_flat_fp16_param = dict()

        # param reduction data structures
        self._is_param_reduced = dict()
        self._reduced_param = []

    def set_param_to_rank(self, tensor: Tensor, rank: int) -> None:
        """
        Set the mapping between parameter to rank, each parameter should be owned by a rank.

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        :param rank: The rank of which the process is responsible for updating the parameter
        :type rank: int
        """

        self._fp16_param_to_rank[tensor] = rank

    def get_param_rank(self, tensor: Tensor) -> int:
        """
        Gives the rank which the parameter belongs to

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor
        """
        return self._fp16_param_to_rank[tensor]

    def belongs_to_current_rank(self, tensor) -> bool:
        """
        Check whether a parameter is supposed to be updated by the process of the current rank

        :param tensor: A :class:`torch.Tensor` object
        :type tensor: torch.Tensor

        :return: True if the parameter should be updated by the current rank. Otherwise false.
        :rtype: bool
        """

        tensor_rank = self._fp16_param_to_rank[tensor]
        return tensor_rank == self._local_rank

    def add_fp16_param_list_by_rank_group(self, rank, group_id, tensor_list) -> None:
        if rank not in self._rank_groupid_to_fp16_param_list:
            self._rank_groupid_to_fp16_param_list[rank] = dict()

        if group_id not in self._rank_groupid_to_fp16_param_list[rank]:
            self._rank_groupid_to_fp16_param_list[rank][group_id] = []

        self._rank_groupid_to_fp16_param_list[rank][group_id].extend(tensor_list)

    def get_fp16_params_by_rank_group(self, rank, group_id) -> List[Tensor]:
        return self._rank_groupid_to_fp16_param_list[rank][group_id]

    def add_flat_fp16_param_by_rank_group(self, rank, group_id, tensor) -> None:
        if rank not in self._rank_group_id_to_flat_fp16_param:
            self._rank_group_id_to_flat_fp16_param[rank] = dict()

        self._rank_group_id_to_flat_fp16_param[rank][group_id] = tensor

    def get_flat_fp16_param_by_rank_group(self, rank, group_id) -> Tensor:
        return self._rank_group_id_to_flat_fp16_param[rank][group_id]

    def is_param_reduced(self, tensor):
        return self._is_param_reduced[tensor]

    def set_param_reduction_state(self, tensor, state):
        self._is_param_reduced[tensor] = state

    def get_param_reduction_states(self):
        return self._is_param_reduced

    def reset_previous_reduced_params(self):
        self._reduced_param = []

    def add_previous_reduced_param(self, tensor):
        self._reduced_param.append(tensor)

    def clear_grads_of_previous_reduced_params(self):
        if len(self._reduced_param) > 0:
            for param in self._reduced_param:
                param.grad = None
            self.reset_previous_reduced_params()
