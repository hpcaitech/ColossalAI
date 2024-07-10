from typing import List, Optional

from torch import Tensor

from .base_store import BaseStore


class GradientStore(BaseStore):
    def __init__(self, *args, partition_grad: bool = False):
        super().__init__(*args)
        """
        self._grads_of_params mapping the parameter and its gradient slices
        data structure:
        {
         group_id:{
            param_id: [grad_rank0, grad_rank1, ...]
          }
        }
        """
        self._grads_of_params = dict()
        # stage 2
        self._partition_grads = partition_grad
        self._working_index = 0 if partition_grad else self._local_rank
        # for zero2, it's `param_id: [grad_local_rank]`
        self.grad_to_param_mapping = dict()

    def get_partitioned_gradients_by_param_id(self, group_id: int, param_id: int) -> List:
        """Return list of gradient slices of a specific parameter

        Args:
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter

        Returns:
            List: the list of gradient slices of a parameter.
        """

        if group_id in self._grads_of_params:
            if param_id in self._grads_of_params[group_id]:
                return self._grads_of_params[group_id][param_id]
        # the param has no grad, for instance, in layer drop
        return []

    def append_gradients_by_param_id(self, grad: Tensor, group_id: int, param_id: int):
        """Append a gradient slice to the parameter's gradient slice list

        Args:
            grad (Tensor): The gradient slice to append to list
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter
        """

        if group_id not in self._grads_of_params:
            self._grads_of_params[group_id] = dict()
        if param_id not in self._grads_of_params[group_id]:
            self._grads_of_params[group_id][param_id] = [grad]
        else:
            self._grads_of_params[group_id][param_id].append(grad)

        self.grad_to_param_mapping[id(grad)] = param_id

    def add_gradients_by_param_id(self, grad: Tensor, grad_idx: int, group_id: int, param_id: int):
        """Add a gradient slice on an existing slice of the parameter's gradient
        Used when no_sync is not activated.

        Args:
            grad (Tensor): The split gradient to append to list
            grad_idx (int): The index of the existing slice
            group_id (int): The index of a parameter group
            param_id (int): The id of a parameter
        """

        self._grads_of_params[group_id][param_id][grad_idx].add_(grad)

    def get_working_grads_by_group_id(self, group_id: int) -> List:
        """Return list of working gradient slices in the group

        Args:
            group_id (int): The index of a parameter group

        Returns:
            List: the list working gradient slices in the group
        """

        grad_list = []
        # When using LoRa and the user sets multiple param_groups, it is possible that some param_groups have no parameters with gradients.
        if group_id not in self._grads_of_params.keys():
            return grad_list
        for param_grads in self._grads_of_params[group_id].values():
            grad_list.append(param_grads[self._working_index])

        return grad_list

    def get_working_grad_by_param_id(self, param_id) -> Tensor:
        """
        Return the working gradient for the specified parameter.

        Args:
            param_id (int): The index of the parameter.

        Returns:
            Tensor: The the working gradient slices for the specified param_id.
        """

        for group in self._grads_of_params.values():
            if param_id in group.keys():
                return group[param_id][self._working_index]
        return None

    def reset_grads_by_group_id(self, group_id: int):
        self._grads_of_params[group_id] = dict()

    def reset_all_gradients(self):
        self._grads_of_params = dict()

    def get_param_id_for_grad(self, grad: Tensor) -> Optional[int]:
        """Return the id of a parameter which the gradient slice belongs to

        Args:
            grad (Tensor): the gradient slice

        Returns:
            int: the id of a parameter which the gradient slice belongs to
        """

        return self.grad_to_param_mapping.get(id(grad), None)
