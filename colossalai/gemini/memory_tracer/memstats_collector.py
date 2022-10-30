from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.utils.memory import colo_device_memory_used, colo_device_memory_capacity
from colossalai.utils import get_current_device
from colossalai.gemini.stateful_tensor import StatefulTensor
from colossalai.gemini.chunk import ChunkManager

import torch
import torch.nn as nn
import time
from typing import List

from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp, is_compatible_with_meta, parameter_size)
from torch.fx import symbolic_trace

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor


class MemStatsCollector:
    """
    A Memory statistic collector.
    It works in two phases. 
    Phase 1. Collection Phase: collect memory usage statistics of CPU and GPU.
    The first iteration of DNN training.
    Phase 2. Runtime Phase: use the read-only collected stats
    The rest iterations of DNN training.

    It has a Sampling counter which is reset after DNN training iteration.
    """

    def __init__(self) -> None:
        self._mem_monitor = SyncCudaMemoryMonitor()
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []
        self._sampling_time = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0

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

    def next_period_non_model_data_usage(self, device_type: str) -> int:
        """Get max non model data memory usage of current sampling period

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert not self._start_flag, 'Cannot get mem stats info during collection phase.'
        assert self._step_total > 0, 'Cannot get mem stats info before collection phase.'
        next_non_model_data = self.non_model_data_list(device_type)[self._step_idx]
        self._step_idx = (self._step_idx + 1) % self._step_total
        return next_non_model_data

    @property
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self.sample_overall_data()
        self._step_total = len(self._sampling_time)
        self._start_flag = False
        self._mem_monitor.finish()

    def sample_model_data(self) -> None:
        """Sampling model data statistics.
        """
        if self._start_flag:
            cuda_mem = StatefulTensor.GST_MGR.total_mem['cuda']
            cpu_mem = StatefulTensor.GST_MGR.total_mem['cpu']
            self._model_data_cuda_list.append(cuda_mem)
            self._model_data_cpu_list.append(cpu_mem)

    def sample_overall_data(self) -> None:
        """Sampling non model data statistics.
        """
        if self._start_flag:
            # overall data recording is after model data recording
            if len(self._model_data_cuda_list) == 0:
                return

            self._overall_cuda_list.append(self._mem_monitor.finish())
            self._overall_cpu_list.append(colo_device_memory_used(torch.device('cpu')))

            assert len(self._model_data_cuda_list) == len(self._overall_cuda_list)

            self._non_model_data_cuda_list.append(self._overall_cuda_list[-1] - self._model_data_cuda_list[-1])
            self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
            self._sampling_time.append(time.time())
            self._mem_monitor.start()

    def clear(self) -> None:
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cpu_list = []
        self._non_model_data_cuda_list = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0


class MemStatsCollectorV2(MemStatsCollector):

    def __init__(self, chunk_manager: ChunkManager) -> None:
        super().__init__()
        self._chunk_manager = chunk_manager

    def sample_model_data(self) -> None:
        """Sampling model data statistics.
        """
        if self._start_flag:
            cuda_mem = self._chunk_manager.total_mem['cuda']
            cpu_mem = self._chunk_manager.total_mem['cpu']
            self._model_data_cuda_list.append(cuda_mem)
            self._model_data_cpu_list.append(cpu_mem)

    @property
    def cuda_margin_mem(self) -> float:
        return colo_device_memory_capacity(get_current_device()) - max(self.overall_mem_stats('cuda'))


class MemStatsCollectorStatic(MemStatsCollector):
    """
    A Static Memory statistic collector.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def init_mem_stats(self, *inputs):

        self.module = self.module.cpu()
        self.module.train()
        data = [MetaTensor(torch.rand(inp.shape, device='meta'), fake_device='cpu') for inp in inputs]
        gm = symbolic_trace(self.module)
        interp = MetaInfoProp(gm)
        interp.propagate(*data)

        total_mem = 0
        for inp in inputs:
            total_mem += inp.numel() * 4.0
        last_node = None
        for node in gm.graph.nodes:
            total_mem = total_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            if node.op == "call_module":
                self._non_model_data_cuda_list.append(total_mem)
                last_node = node

        cur_module_mem_fwd = 0
        cur_module_mem_bwd = 0
        grad_module_out = last_node.meta["fwd_mem_out"]
        for node in gm.graph.nodes.__reversed__():
            cur_module_mem_fwd = cur_module_mem_fwd + node.meta["fwd_mem_tmp"] + node.meta["fwd_mem_out"]
            cur_module_mem_bwd = cur_module_mem_bwd + node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]
            if node.op == "call_module":
                self._non_model_data_cuda_list.append(total_mem + grad_module_out + cur_module_mem_bwd)
                total_mem = total_mem - cur_module_mem_fwd
                cur_module_mem_fwd = 0
                cur_module_mem_bwd = 0
                grad_module_out = node.meta["bwd_mem_out"]

    def next_period_non_model_data_usage(self, device_type: str) -> int:
        """Get max non model data memory usage of current sampling period

        Args:
            device_type (str): device type, can be 'cpu' or 'cuda'.

        Returns:
            int: max non model data memory usage of current sampling period
        """
        assert not self._start_flag, 'Cannot get mem stats info during collection phase.'
        assert self._step_total > 0, 'Cannot get mem stats info before collection phase.'
        next_non_model_data = self.non_model_data_list(device_type)[self._step_idx]
        # self._step_idx = (self._step_idx + 1) % self._step_total
        self._step_idx = (self._step_idx + 1) % len(self._non_model_data_cuda_list)
        return next_non_model_data


    def sample_overall_data(self) -> None:
        """Sampling non model data statistics.
        """
        if self._start_flag:
            # overall data recording is after model data recording
            if len(self._model_data_cuda_list) == 0:
                return

            self._overall_cuda_list.append(self._mem_monitor.finish())
            self._overall_cpu_list.append(colo_device_memory_used(torch.device('cpu')))

            assert len(self._model_data_cuda_list) == len(self._overall_cuda_list)

            # self._non_model_data_cuda_list.append(self._overall_cuda_list[-1] - self._model_data_cuda_list[-1])
            self._non_model_data_cpu_list.append(self._overall_cpu_list[-1] - self._model_data_cpu_list[-1])
            self._sampling_time.append(time.time())
            self._mem_monitor.start()
