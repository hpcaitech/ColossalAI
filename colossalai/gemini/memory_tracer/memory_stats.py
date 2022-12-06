from typing import Any, Dict, List


class MemStats(object):

    def __init__(self) -> None:
        """
        Store the non model data statistics used for Gemini and ZeroOptimizer.
        """
        # p -> list of non_model data volumn visied in order.
        self.param_non_model_data_map: Dict(Any, List[int]) = {}

        self._model_data_cuda_list = []
        self._model_data_cpu_list = []

        self._overall_cuda_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []

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

    def append_non_model_data(self, device_type: str):
        if len(self._overall_cuda_list) == 0 or len(self._model_data_cuda_list) == 0:
            return
        if device_type == 'cuda':
            self._non_model_data_cuda_list.append(self._overall_cuda_list[-1] - self._model_data_cuda_list[-1])
        elif device_type == 'cpu':
            self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
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
