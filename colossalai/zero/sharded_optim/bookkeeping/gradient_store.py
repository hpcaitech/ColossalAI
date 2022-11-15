from typing import List

from torch import Tensor

from .base_store import BaseStore


class GradientStore(BaseStore):

    def __init__(self, *args):
        super().__init__(*args)
        # bookkeeping data structures
        self._averaged_gradients = dict()

        # for backward reduction hooks
        self._grad_acc_objs = []

    def add_accumulate_grad_object(self, obj):
        """
        Keep :class:`AccumulateGrad` objects. If these objects are not kept, reduction hooks may not
        be attached successfully.

        :param obj: An object of :class:`AccumulateGrad` class
        :type obj: :class:`AccumulateGrad`
        """

        self._grad_acc_objs.append(obj)

    def get_averaged_gradients_by_group(self, group_id: int) -> List[Tensor]:
        """
        Return average gradients of a parameter group

        :param group_id: The index of parameter group
        :type group_id: int

        :return: Return the list of averaged gradients of a parameter group. Each element is a gradient, not a parameter.
        :rtype: List[torch.Tensor]
        """

        return self._averaged_gradients[group_id]

    def add_average_gradient_by_group(self, group_id: int, tensor: Tensor) -> None:
        """
        Append an average gradient to the list of averaged gradients of a parameter group

        :param group_id: The index of a parameter group
        :param tensor: A :class:`torch.Tensor` object
        :type group_id: int
        :type tensor: torch.Tensor

        """

        if group_id in self._averaged_gradients:
            self._averaged_gradients[group_id].append(tensor)
        else:
            self._averaged_gradients[group_id] = [tensor]

    def reset_average_gradients_by_group(self, group_id: int) -> None:
        """
        Reset the bookkeeping data structure for averaged gradients to an empty list

        :param group_id: The index of a parameter group
        :type group_id: int
        """

        self._averaged_gradients[group_id] = []
