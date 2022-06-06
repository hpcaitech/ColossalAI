import torch
import torch.distributed as dist
from enum import Enum
from torch.optim import Optimizer
from colossalai.nn.parallel.data_parallel import ColoDDPV2
from typing import Dict
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class ZeroOptimizer(ColossalaiOptimizer):

    def __init__(self,
                 optim: Optimizer,
                 module: ColoDDPV2,
                 initial_scale: float = 2**32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32):
        super().__init__(optim)
        assert isinstance(module, ColoDDPV2)
        self.module = module
        self.optim_state = OptimState.UNSCALED
        self.fp16_param_to_fp32_param: Dict[torch.Tensor, torch.Tensor] = {}
        for p, fp32_p in zip(module.parameters(), module.fp32_params):
            self.fp16_param_to_fp32_param[p] = fp32_p

        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())
        self._logger = get_dist_logger()

    def _update_params_ptr(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                if not self.module.chunk_manager.is_chunk_free(p):
                    p.data = self.fp16_param_to_fp32_param[p]
                else:
                    assert p.grad is None

    def _update_fp16_params(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                if not self.module.chunk_manager.is_chunk_free(p):
                    # TODO(ver217): copy chunk
                    fp32_p = self.fp16_param_to_fp32_param[p]
                    self.module.chunk_manager.copy_tensor_to_chunk_slice(p, fp32_p)

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        # all-reduce across global group
        dist.all_reduce(self._found_overflow)

        return self._found_overflow.item() > 0

    def _unscale_grads(self):
        assert self.optim_state == OptimState.SCALED
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
        self.optim_state = OptimState.UNSCALED

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        # unscale grads if scaled
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        found_inf = self._check_overflow()
        self.grad_scaler.update(found_inf)
        if found_inf:
            self._logger.info(f'Found overflow. Skip step')
            self.zero_grad()
            self._update_fp16_params()
            return
        self._update_params_ptr()
        ret = self.optim.step(*args, **kwargs)
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float):
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        return super().clip_grad_norm(model, max_norm)

    def backward(self, loss: torch.Tensor):
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.module.backward(loss)

    def backward_by_grad(self, tensor: torch.Tensor, grad: torch.Tensor):
        self.module.backward_by_grad(tensor, grad)
