from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.tensor.param_op_hook import ParamOpHook
from colossalai.gemini.tensor_utils import free_storage, alloc_storage


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class ParamTracerHook(ParamOpHook):

    def __init__(self, dtype: torch.dtype = torch.half) -> None:
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self.mem_monitor = SyncCudaMemoryMonitor()
        self._non_model_data_list = []
        self._model_data_list = []
        self.dtype = dtype

    def _free_cuda_params(self, params):
        for p in params:
            free_storage(p.data)

    def _allocate_params_on_cuda(self, params):
        for p in params:
            cur_dev = p.data.device.type
            if cur_dev == "cpu":
                # p.data = p.data.to("cuda")
                p.data = torch.randn(p.data.shape, device="cuda", dtype=self.dtype)
            elif cur_dev == "cuda":
                alloc_storage(p.data)

    def sample_model_data(self, params):
        data_volume = 0
        for p in params:
            data_volume += p.data.numel() * p.data.element_size()
        if self._training_phase == TrainingPhase.BACKWARD:
            # add param.grad, actually param.grad is None in this time
            data_volume *= 2
        self._model_data_list.append(data_volume)

    def pre_op(self, params):
        cuda_volume = self.mem_monitor.finish()
        if len(self._model_data_list):
            self._non_model_data_list.append(cuda_volume - self._model_data_list[-1])
        self._allocate_params_on_cuda(params)
        self.sample_model_data(params)
        self.mem_monitor.start()

    def post_op(self, params):
        self._free_cuda_params(params)

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    @contextmanager
    def switch_training_phase(self, training_phase: TrainingPhase = TrainingPhase.BACKWARD):
        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)
