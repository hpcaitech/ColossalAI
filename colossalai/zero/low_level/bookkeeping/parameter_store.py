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
        """Record the padding size of a param

        Args:
            param (Tensor): The parameter
            padding_size (int): The padding size of the parameter
        """

        self._padding_map[id(param)] = padding_size

    def get_param_padding_size(self, param: Tensor) -> int:
        """Return the padding size of the parameter

        Args:
            param (Tensor): The parameter

        Returns:
            int: the padding size of the parameter
        """

        return self._padding_map[id(param)]

    def link_master_and_working_param(self, master_param: Tensor, working_param: Tensor):
        """Mapping master parameter and working parameter

        Args:
            master_param (Tensor): The parameter copy in optimizer
            working_param (Tensor): The parameter of the model
        """

        self.master_to_working_param[id(master_param)] = working_param
        self.working_to_master_param[id(working_param)] = master_param
