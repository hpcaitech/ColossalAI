from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.tensor.param_op_hook import ParamOpHook


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class ParamTracerHook(ParamOpHook):

    def __init__(self) -> None:
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self.mem_monitor = SyncCudaMemoryMonitor()
        self._non_model_data_list = []
        self._model_data_list = []

    def _move_params_to_dev(self, params, dev: str) -> int:
        assert isinstance(dev, str), f"device should be a str not torch.device"
        comm_volume = 0
        for p in params:
            if p.data.device.type != dev:
                p.data = p.data.to(dev)
                comm_volume += p.data.numel() * p.data.element_size()
            if p.grad is not None:
                if p.grad.device.type != dev:
                    p.grad = p.grad.to(dev)
                    comm_volume += p.grad.numel() * p.grad.element_size()
        return comm_volume

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
        self._move_params_to_dev(params, 'cuda')
        self.sample_model_data(params)
        self.mem_monitor.start()

    def post_op(self, params):
        self._move_params_to_dev(params, 'cpu')

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