from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

from .base_store import BaseStore


class ParameterStore(BaseStore):

    def __init__(self, torch_pg: ProcessGroup):
        super().__init__(torch_pg)

        self._padding_map = dict()
        self._position_map = dict()
        self._marked_params = dict()
        self._numel_per_split_param = dict()

        self.master_to_working_param = dict()
        self.working_to_master_param = dict()

    def record_param_padding_size(self, param, padding_size):
        self._padding_map[id(param)] = padding_size

    def get_param_padding_size(self, param):
        return self._padding_map[id(param)]

    def record_offset_in_flatten(self, param, position):
        self._position_map[id(param)] = position

    def get_offset_in_flatten(self, param):
        return self._position_map[id(param)]

    def record_numel_per_split_param(self, param, numel):
        self._numel_per_split_param[id(param)] = numel

    def get_numel_per_split_param(self, param):
        return self._numel_per_split_param[id(param)]

    def link_master_and_working_param(self, master_param, working_param):
        self.master_to_working_param[id(master_param)] = working_param
        self.working_to_master_param[id(working_param)] = master_param
