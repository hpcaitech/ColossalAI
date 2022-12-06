from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.gemini.memory_tracer.model_data_memtracer import GLOBAL_CUDA_MEM_INFO
from colossalai.gemini.tensor_utils import alloc_storage, free_storage
from colossalai.tensor.param_op_hook import ColoParamOpHook


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class GradMemTracerHook():

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.grad_hook_list = []

    def grad_handle(self, p, grad):
        assert GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag[p]
        free_storage(grad)
        GLOBAL_CUDA_MEM_INFO.unreleased_grad_volume -= grad.numel() * grad.element_size()
        GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag[p] = False

    def register_grad_hook(self):
        for p in self.module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(partial(self.grad_handle, p)))
                GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag[p] = False

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()


class ParamMemTracerHook(ColoParamOpHook):

    def __init__(self, memstats) -> None:
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self._memstats = memstats
        self.mem_monitor = SyncCudaMemoryMonitor()

    def _free_cuda_params(self, params):
        for p in params:
            if p.data.device.type == "cpu":
                raise NotImplementedError("Only free cuda memory")
            free_storage(p.data)

    def _allocate_params_on_cuda(self, params):
        for p in params:
            cur_dev = p.data.device.type
            if cur_dev == "cpu":
                if p.grad is not None and p.grad.device.type == "cpu":
                    raise NotImplementedError("Only run in forward propagation")
                p.data = torch.empty(p.data.shape,
                                     device="cuda",
                                     dtype=p.data.dtype,
                                     requires_grad=p.data.requires_grad)
            elif cur_dev == "cuda":
                alloc_storage(p.data)

    def sample_model_data(self, params):
        data_volume = GLOBAL_CUDA_MEM_INFO.unreleased_grad_volume
        for p in params:
            cur_model_data_volume = p.data.numel() * p.data.element_size()
            data_volume += cur_model_data_volume
            if self._training_phase == TrainingPhase.BACKWARD and p.requires_grad:
                # add param.grad, actually param.grad is None in this time
                data_volume += cur_model_data_volume
                if not GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag[p]:
                    GLOBAL_CUDA_MEM_INFO.unreleased_grad_volume += cur_model_data_volume
                    GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag[p] = True
        # GLOBAL_CUDA_MEM_INFO.model_data_list.append(data_volume)
        self._memstats.append_model_data('cuda', data_volume)

    def pre_op(self, params):
        cuda_volume = self.mem_monitor.finish()
        self._memstats.append_model_data('cuda', cuda_volume)
        # if len(GLOBAL_CUDA_MEM_INFO.model_data_list):
        #     GLOBAL_CUDA_MEM_INFO.non_model_data_list.append(cuda_volume - GLOBAL_CUDA_MEM_INFO.model_data_list[-1])
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
