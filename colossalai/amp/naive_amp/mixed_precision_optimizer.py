from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper

from .mixed_precision_mixin import BF16MixedPrecisionMixin, FP16MixedPrecisionMixin


class NaiveFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):

    def __init__(self,
                 working_params: List[Parameter],
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32) -> None:
        super().__init__(initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis,
                         max_scale)
        self.params = working_params

    def check_local_overflow(self) -> bool:
        for p in self.params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False


class MixedPrecisionOptimizer(OptimizerWrapper):

    def __init__(self,
                 optim: Optimizer,
                 precision: str = 'fp16',
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32,
                 max_norm: float = 0.0):
        super().__init__(optim)
        if precision == 'fp16':
            working_params = []
            for group in self.optim.param_groups:
                for p in group['params']:
                    working_params.append(p)
            self.mixed_precision = NaiveFP16MixedPrecisionMixin(working_params,
                                                                initial_scale=initial_scale,
                                                                min_scale=min_scale,
                                                                growth_factor=growth_factor,
                                                                backoff_factor=backoff_factor,
                                                                growth_interval=growth_interval,
                                                                hysteresis=hysteresis,
                                                                max_scale=max_scale)
        elif precision == 'bf16':
            self.mixed_precision = BF16MixedPrecisionMixin()
        else:
            raise ValueError(f'Unsupported precision: {precision}')
        if max_norm > 0.0:
            raise NotImplementedError('max_norm is not supported yet.')
        self.max_norm = max_norm
        self.working_to_master_map: Dict[Parameter, Tensor] = {}
        self.master_to_working_map: Dict[Tensor, Parameter] = {}

        # create master weights
        for group in self.optim.param_groups:
            master_params = []
            for p in group['params']:
                if p.requires_grad:
                    master_p = p
                    if p.dtype != torch.float:
                        master_p = p.detach().float()
                    self.working_to_master_map[p] = master_p
                    self.master_to_working_map[master_p] = p
                    master_params.append(master_p)
            group['params'] = master_params

    def backward(self, loss: Tensor, *args, **kwargs):
        loss = self.mixed_precision.pre_backward(loss)
        loss.backward(*args, **kwargs)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor):
        grad = self.mixed_precision.pre_backward_by_grad(tensor, grad)
        tensor.backward(grad)

    def zero_grad(self, *args, **kwargs):
        for p in self.working_to_master_map.keys():
            p.grad = None
        self.mixed_precision.pre_zero_grad()
        return super().zero_grad(*args, **kwargs)

    def _unscale_and_clip_grads(self, total_norm: float) -> None:
        div_scale = 1.0
        if self.mixed_precision is not None:
            div_scale = self.mixed_precision.get_grad_div_scale()

        if self.max_norm > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / div_scale) + 1e-6) / self.max_norm
            if clip > 1:
                div_scale = clip * div_scale

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data.mul_(1. / div_scale)

    def _compute_grad_norm(self) -> float:
        if self.max_norm <= 0.:
            return 0.
        grads = [p.grad for group in self.param_groups for p in group['params'] if p.grad is not None]
        if len(grads) == 0:
            return 0.
        device = grads[0].device
        # TODO(ver217): support tp
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
        return total_norm.item()

    def step(self, *args, **kwargs):
        if self.mixed_precision.should_skip_step():
            self.zero_grad()
            return
        # prepare grads
        for group in self.optim.param_groups:
            for p in group['params']:
                working_param = self.master_to_working_map[p]
                if p is working_param:
                    continue
                if working_param.grad is not None:
                    p.grad = working_param.grad.data.float()
                    working_param.grad = None
        total_norm = self._compute_grad_norm()
        self._unscale_and_clip_grads(total_norm)
        self.optim.step(*args, **kwargs)
        # update working params
        for group in self.optim.param_groups:
            for p in group['params']:
                working_param = self.master_to_working_map[p]
                if p is working_param:
                    continue
                working_param.data.copy_(p.data)
