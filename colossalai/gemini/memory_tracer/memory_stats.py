from typing import Any, Dict, List, Optional

import torch

from colossalai.gemini.memory_tracer import OrderedParamGenerator


class MemStats(object):

    def __init__(self) -> None:
        """
        Store the non model data statistics used for Gemini and ZeroOptimizer.
        """
        # (preop_step, List[param])
        self._step_param_dict = dict()
        # (param, List[preop_step])
        self._param_step_dict = dict()
        # (preop_step, non_model_data) non model data used during preop_step ~ (preop_step+1)
        self._step_nmd_dict = dict()
        self._param_runtime_order = OrderedParamGenerator()

        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._prev_md_cuda = -1

        # old version
        self._model_data_cuda_list = []
        self._model_data_cpu_list = []

        self._overall_cuda_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []

    def calc_max_cuda_non_model_data(self):
        if self._prev_overall_cuda != -1 and self._prev_md_cuda != -1:
            max_cuda_non_model_data = self._prev_overall_cuda - self._prev_md_cuda
            self._step_nmd_dict[self._preop_step - 1] = max_cuda_non_model_data
            # compatibility of the old version.
            self._non_model_data_cuda_list.append(max_cuda_non_model_data)

    def record_max_cuda_model_data(self, val):
        self._prev_md_cuda = val

    def record_max_cuda_overall_data(self, val):
        self._prev_overall_cuda = val

    def increase_preop_step(self, param_list: List[torch.nn.Parameter]):
        """
        the time step is increased. param list is used between current and the next
        time step.

        Args:
            param_list (List[torch.nn.Parameter]): a list of torch paramters.
        """
        for p in param_list:
            if p not in self._param_step_dict:
                self._param_step_dict[p] = [self._preop_step]
            else:
                self._param_step_dict[p].append(self._preop_step)
            self._param_runtime_order.append(p)
        self._step_param_dict[self._preop_step] = param_list
        self._preop_step += 1

    def param_used_step(self, param: torch.nn.Parameter) -> Optional[List[int]]:
        """param_used_step
        get the timestep list using the param

        Args:
            param (torch.nn.Parameter): a torch param

        Returns:
            Optional[List[int]]: a list of int indicates the time step of preop hook.
        """
        if param not in self._param_step_dict:
            return None
        else:
            return self._param_step_dict[param]

    def param_order(self):
        if self._param_runtime_order.is_empty():
            raise RuntimeError
        else:
            return self._param_runtime_order

    ## APIs to be depracated
    def append_overall_data(self, device_type: str, val: float):
        if device_type == 'cuda':
            self._overall_cuda_list.append(val)
        elif device_type == 'cpu':
            self._overall_cpu_list.append(val)
        else:
            raise TypeError

    def append_model_data(self, device_type: str, val: float):
        if device_type == 'cuda':
            self._model_data_cuda_list.append(val)
        elif device_type == 'cpu':
            self._model_data_cpu_list.append(val)
        else:
            raise TypeError

    def last_model_data(self, device_type: str):
        if len(self._model_data_cuda_list) == 0:
            return None
        if device_type == 'cuda':
            return self._model_data_cuda_list[-1]
        elif device_type == 'cpu':
            return self._model_data_cpu_list[-1]
        else:
            raise TypeError

    def append_non_model_data(self, device_type: str, val=None):
        if device_type == 'cuda':
            if val is None:
                if len(self._overall_cuda_list) == 0 or len(self._model_data_cuda_list) == 0:
                    return
                self._non_model_data_cuda_list.append(self._overall_cuda_list[-1] - self._model_data_cuda_list[-1])
            else:
                self._non_model_data_cuda_list.append(val)
        elif device_type == 'cpu':
            if val is None:
                if len(self._overall_cuda_list) == 0 or len(self._model_data_cuda_list) == 0:
                    return
                self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
            else:
                self._non_model_data_cuda_list.append(val)
        else:
            raise TypeError

    def overall_mem_stats(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._overall_cuda_list
        elif device_type == 'cpu':
            return self._overall_cpu_list
        else:
            raise TypeError

    def model_data_list(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._model_data_cuda_list
        elif device_type == 'cpu':
            return self._model_data_cpu_list
        else:
            raise TypeError

    def non_model_data_list(self, device_type: str) -> List[int]:
        if device_type == 'cuda':
            return self._non_model_data_cuda_list
        elif device_type == 'cpu':
            return self._non_model_data_cpu_list
        else:
            raise TypeError

    def max_non_model_data(self, device_type: str) -> float:
        if device_type == 'cuda':
            return max(self._non_model_data_cuda_list)
        elif device_type == 'cpu':
            return max(self._non_model_data_cpu_list)
        else:
            raise TypeError

    def max_overall_cuda(self, device_type: str) -> float:
        if device_type == 'cuda':
            return max(self._overall_cuda_list)
        elif device_type == 'cpu':
            return max(self._overall_cpu_list)
        else:
            raise TypeError

    def clear(self):
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cpu_list = []
        self._non_model_data_cuda_list = []

        self._param_runtime_order.clear()
        self._step_param_dict.clear()
        self._param_step_dict.clear()
        self._step_nmd_dict.clear()
        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._prev_md_cuda = -1
