# This test checks adam kernels
# Baseline is pure fp32 torch adam optimizer
import math
from abc import abstractmethod
from typing import Type

import pytest
import torch
from torch import Tensor

from colossalai.accelerator import get_accelerator
from colossalai.utils import multi_tensor_applier

_FUSED_ALLOWED_P_G_TYPES = [
    (torch.float, torch.half),
    (torch.float, torch.float),
    (torch.half, torch.half),
    (torch.float, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
]

_CPU_ALLOWED_P_G_TYPES = [
    (torch.float, torch.half),
    (torch.float, torch.float),
    (torch.half, torch.half),
]


class AdamKernel:
    def __init__(self, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, use_adamw: bool) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw

    @abstractmethod
    def update(self, step: int, param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor):
        pass


class TorchAdamKernel(AdamKernel):
    def update(self, step: int, param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor):
        bias_correction1 = 1 - self.beta1**step
        bias_correction2 = 1 - self.beta2**step

        if self.weight_decay != 0:
            if self.use_adamw:
                # Perform stepweight decay
                param.mul_(1 - self.lr * self.weight_decay)
            else:
                grad = grad.add(param, alpha=self.weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

        step_size = self.lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class FusedAdamKernel(AdamKernel):
    def __init__(self, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, use_adamw: bool) -> None:
        super().__init__(lr, beta1, beta2, eps, weight_decay, use_adamw)
        from colossalai.kernel.kernel_loader import FusedOptimizerLoader

        fused_optim = FusedOptimizerLoader().load()
        self.fused_adam = fused_optim.multi_tensor_adam
        self.dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=get_accelerator().get_current_device())

    def update(self, step: int, param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor):
        multi_tensor_applier(
            self.fused_adam,
            self.dummy_overflow_buf,
            [[grad], [param], [exp_avg], [exp_avg_sq]],
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            step,
            self.use_adamw,
            True,
            self.weight_decay,
            -1,
        )


class CPUAdamKernel(AdamKernel):
    def __init__(self, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, use_adamw: bool) -> None:
        super().__init__(lr, beta1, beta2, eps, weight_decay, use_adamw)
        from colossalai.kernel.kernel_loader import CPUAdamLoader

        cpu_optim = CPUAdamLoader().load()

        self.cpu_adam_op = cpu_optim.CPUAdamOptimizer(lr, beta1, beta2, eps, weight_decay, use_adamw)

    def update(self, step: int, param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor):
        self.cpu_adam_op.step(
            step,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            True,
            param.view(-1),
            grad.view(-1),
            exp_avg.view(-1),
            exp_avg_sq.view(-1),
            -1,
        )


def check_adam_kernel(
    kernel: Type[AdamKernel],
    adamw: bool,
    weight_decay: float,
    p_dtype: torch.dtype,
    g_dtype: torch.dtype,
    device: torch.device,
    n_steps: int,
    rtol: float,
    atol: float,
):
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    torch_adam = TorchAdamKernel(lr, beta1, beta2, eps, weight_decay, adamw)
    adam_kernel = kernel(lr, beta1, beta2, eps, weight_decay, adamw)
    master_p = torch.rand(64, device=device)
    master_g = torch.rand_like(master_p)
    master_exp_avg = torch.zeros_like(master_p)
    master_exp_avg_sq = torch.zeros_like(master_p)
    p = master_p.clone().to(p_dtype)
    g = master_g.clone().to(g_dtype)
    exp_avg = master_exp_avg.clone().to(p_dtype)
    exp_avg_sq = master_exp_avg_sq.clone().to(p_dtype)

    for step in range(1, 1 + n_steps):
        torch_adam.update(step, master_p, master_g, master_exp_avg, master_exp_avg_sq)
        adam_kernel.update(step, p, g, exp_avg, exp_avg_sq)
        # if overflow, the weight won't be updated. so there will be no nan in p
        assert not torch.isnan(p).any()
        assert torch.allclose(master_p, p.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("adamw", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("p_dtype, g_dtype", _FUSED_ALLOWED_P_G_TYPES)
def test_fused_adam_kernel(adamw, weight_decay, p_dtype, g_dtype):
    rtol, atol = 1e-5, 1e-8
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 1e-3, 1e-3
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 4e-3, 4e-3
    check_adam_kernel(
        FusedAdamKernel, adamw, weight_decay, p_dtype, g_dtype, get_accelerator().get_current_device(), 3, rtol, atol
    )


@pytest.mark.parametrize("adamw", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("p_dtype, g_dtype", _CPU_ALLOWED_P_G_TYPES)
def test_cpu_adam_kernel(adamw, weight_decay, p_dtype, g_dtype):
    rtol, atol = 1e-5, 1e-8
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 1e-3, 1e-3
    check_adam_kernel(CPUAdamKernel, adamw, weight_decay, p_dtype, g_dtype, torch.device("cpu"), 3, rtol, atol)
