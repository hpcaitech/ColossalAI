from torch import Tensor
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class ParameterStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)

        # record the padding size of each param
        self._padding_map = dict()

        # mapping working param and master param
        self.master_to_working_param = dict()
        self.working_to_master_param = dict()

    def record_param_padding_size(self, param: Tensor, padding_size: int):
        """
        Record the padding size of a param
        :param param: The parameter
        :param paddind_size: The padding size of the parameter
        :type param: Tensor
        :type padding_size: int

        """
        self._padding_map[id(param)] = padding_size

    def get_param_padding_size(self, param: Tensor) -> int:
        """
        Return the padding size of the parameter
        :param param: The parameter
        :type param: Tensor

        :return: Return the padding size of the parameter
        :rtype: int
        """
        return self._padding_map[id(param)]

    def link_master_and_working_param(self, master_param: Tensor, working_param: Tensor):
        """
        Mapping master parameter and working parameter
        :param master_param: The parameter copy in optimizer
        :param working_param: The parameter of the model
        :type master_param: Tensor
        :type working_param: Tensor

        """
        self.master_to_working_param[id(master_param)] = working_param
        self.working_to_master_param[id(working_param)] = master_param
