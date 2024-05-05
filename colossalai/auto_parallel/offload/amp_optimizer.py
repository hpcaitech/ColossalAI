from enum import Enum
from typing import Dict, Tuple

import torch
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.interface import OptimizerWrapper
from colossalai.logging import get_dist_logger

from .base_offload_module import BaseOffloadModule
from .region import Region
from .region_manager import RegionManager


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class AMPOptimizer(OptimizerWrapper):
    """
    A wrapper for Optimizer.
    Code reference: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/optimizer/zero_optimizer.py

    Args:
        optimizer (Optimizer): An Optimizer instance.
        module (BaseOffloadModule): A ``BaseOffloadModule`` instance.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**16.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module: BaseOffloadModule,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        min_scale: float = 1,
        max_scale: float = 2**32,
        clipping_norm: float = 0.0,
        norm_type: float = 2.0,
    ):
        super().__init__(optimizer)

        self.module = module
        self.optim_state = OptimState.UNSCALED
        self.clipping_flag = clipping_norm > 0.0
        self.max_norm = clipping_norm

        self.region_manager: RegionManager = self.module.region_manager
        self.param_to_range: Dict[torch.nn.Parameter, Tuple[int, int]] = dict()
        self.param_to_region: Dict[torch.nn.Parameter, Region] = dict()

        self.fp32_to_fp16_params: Dict[torch.Tensor, torch.nn.Parameter] = dict()

        if self.clipping_flag:
            assert norm_type == 2.0, "AMPOptimizer only supports L2 norm now"

        self.__init__optimizer()

        # Grad scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )
        self._found_overflow: torch.Tensor = torch.zeros(
            1, dtype=torch.int64, device=get_accelerator().get_current_device()
        )
        self._logger = get_dist_logger()

    def _set_grad_ptr(self):
        for group in self.param_groups:
            for fake_param in group["params"]:
                region = self.param_to_region[fake_param]
                begin, end = self.param_to_range[fake_param]

                fake_param.data = region.cpu_grad[begin:end]
                fake_param.grad = fake_param.data
                fake_param.data = region.fp32_data[begin:end]

    def _update_fp16_params(self):
        none_tensor = torch.empty([0])
        for group in self.param_groups:
            for fake_param in group["params"]:
                assert fake_param.grad is None
                fake_param.data = none_tensor
                self.param_to_region[fake_param].cpu_grad = None

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter.item())
        return self._found_overflow.item() > 0

    def _get_combined_scale(self):
        loss_scale = 1

        if self.optim_state == OptimState.SCALED:
            loss_scale = self.loss_scale
            self.optim_state = OptimState.UNSCALED

        combined_scale = loss_scale

        if combined_scale == 1:
            return -1
        else:
            return combined_scale

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = torch.tensor([0], dtype=torch.int, device=get_accelerator().get_current_device())
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        # Copy gradients from model params to main params.
        self._set_grad_ptr()

        found_inf = self._check_overflow()
        if found_inf:
            self.optim_state = OptimState.UNSCALED  # no need to unscale grad
            self.grad_scaler.update(found_inf)  # update gradient scaler
            self._logger.info(f"Found overflow. Skip step")
            self.zero_grad()  # reset all gradients
            self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()
        self.grad_scaler.update(found_inf)

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float, norm_type: float = 2.0):
        raise NotImplementedError

    def backward(self, loss: torch.Tensor):
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.module.backward(loss)

    def __init__optimizer(self):
        for group in self.optim.param_groups:
            fake_params_list = list()

            for param in group["params"]:
                region = self.region_manager.get_region(param)
                fake_param = torch.nn.Parameter(torch.empty([0]))
                self.param_to_range[fake_param] = region.param_to_range[param]
                self.param_to_region[fake_param] = region
                fake_params_list.append(fake_param)

                # Reset existing state dict key to the new main param.
                if param in self.optim.state:
                    self.optim.state[fake_param] = self.optim.state.pop(param)

            group["params"] = fake_params_list

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optim.load_state_dict(self.optim.state_dict())
