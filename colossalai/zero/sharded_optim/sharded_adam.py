from enum import Enum
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.amp.naive_amp._fp16_optimizer import DynamicGradScaler
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.zero.sharded_model import ShardedModelV2
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.optim import Optimizer

from ._utils import has_inf_or_nan


class OptimState(Enum):
    SCALED = 1
    UNSCALED = 2


class ShardedAdam(ColossalaiOptimizer):

    def __init__(self,
                 adam_optim: Optimizer,
                 sharded_model: nn.Module,
                 cpu_offload: bool = False,
                 initial_scale: float = 2**32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: float = 1000,
                 hysteresis: float = 2,
                 max_scale: int = 2**32,
                 dp_process_group: Optional[ProcessGroup] = None,
                 mp_process_group: Optional[ProcessGroup] = None) -> None:
        super().__init__(adam_optim)
        self.model: Union[nn.Module, ShardedModelV2] = sharded_model
        self.model_is_sharded = isinstance(sharded_model, ShardedModelV2)
        self.state_device = torch.cuda.current_device() if not cpu_offload else torch.device('cpu')
        self.optim_state: OptimState = OptimState.UNSCALED
        self.dp_process_group = dp_process_group or gpc.get_group(ParallelMode.DATA)
        self.mp_process_group = mp_process_group or gpc.get_group(ParallelMode.MODEL)
        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: Tensor = torch.FloatTensor([0]).to(self.state_device)

        # Early state initialization
        for group in adam_optim.param_groups:
            for p in group['params']:
                state_shape = p.shape
                if hasattr(p, 'ca_attr'):
                    assert p.ca_attr.is_sharded, 'ShardedAdam can be only used with sharded model'
                    # TODO: use payload shape
                    state_shape = p.ca_attr.payload(self.state_device)
                state = adam_optim.state[p]
                assert len(state) == 0, 'adam optimizer initialized'
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros(state_shape,
                                               memory_format=torch.preserve_format,
                                               dtype=torch.float,
                                               device=self.state_device)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros(state_shape,
                                                  memory_format=torch.preserve_format,
                                                  dtype=torch.float,
                                                  device=self.state_device)
                if group['amsgrad']:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros(state_shape,
                                                          memory_format=torch.preserve_format,
                                                          dtype=torch.float,
                                                          device=self.state_device)

    def step(self, *args, **kwargs):
        # unscale grads if scaled
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()

        found_inf = self._check_overflow()
        self.grad_scaler.update(found_inf)

        if found_inf:
            self.zero_grad()
            return

        # Write payload back to p.data
        for group in self.optim.param_groups:
            for p in group['params']:
                data = p.data
                if hasattr(p, 'ca_attr'):
                    data = p.ca_attr.payload(self.state_device)
                if torch.is_floating_point(data) and data.dtype != torch.float:
                    data = data.to(torch.float)
                p.data = data
        ret = self.optim.step(*args, **kwargs)
        # Set p.data to None
        for group in self.optim.param_groups:
            for p in group['params']:
                p.data = None
        return ret

    def backward(self, loss: Tensor) -> None:
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        if self.model_is_sharded:
            self.model.backward(loss)
        else:
            super().backward(loss)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor) -> None:
        if self.model_is_sharded:
            self.model.backward_by_grad(tensor, grad)
        else:
            super().backward_by_grad(tensor, grad)

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        return super().clip_grad_norm(model, max_norm)

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group in self.optim.param_groups:
            for p in group['params']:
                if has_inf_or_nan(p.grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce across dp group
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        # all-reduce over model parallel group
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self.mp_process_group)

        if self._found_overflow.item() > 0:
            return True
        else:
            return False

    def _unscale_grads(self):
        assert self.optim_state == OptimState.SCALED
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
        self.optim_state = OptimState.UNSCALED
