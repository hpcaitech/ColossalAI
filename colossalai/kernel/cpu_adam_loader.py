import platform
from collections import OrderedDict

from .base_kernel_loader import BaseKernelLoader
from .extensions.cpu_adam import ArmCPUAdamExtension, X86CPUAdamExtension


class CPUAdamLoader(BaseKernelLoader):
    """
    CPU Adam Loader

    Usage:
        # init
        cpu_adam = CPUAdamLoader().load()
        cpu_adam_op = cpu_adam.CPUAdamOptimizer(
            alpha, beta1, beta2, epsilon, weight_decay, adamw_mode,
        )
        ...
        # optim step
        cpu_adam_op.step(
            step, lr, beta1, beta2, epsilon, weight_decay, bias_correction,
            params, grads, exp_avg, exp_avg_sq, loss_scale,
        )

    Args:
        func CPUAdamOptimizer:
            alpha (float): learning rate. Default to 1e-3.
            beta1 (float): coefficients used for computing running averages of gradient. Default to 0.9.
            beta2 (float): coefficients used for computing running averages of its square. Default to 0.99.
            epsilon (float): term added to the denominator to improve numerical stability. Default to 1e-8.
            weight_decay (float): weight decay (L2 penalty). Default to 0.
            adamw_mode (bool): whether to use the adamw. Default to True.
        func step:
            step (int): current step.
            lr (float): learning rate.
            beta1 (float): coefficients used for computing running averages of gradient.
            beta2 (float): coefficients used for computing running averages of its square.
            epsilon (float): term added to the denominator to improve numerical stability.
            weight_decay (float): weight decay (L2 penalty).
            bias_correction (bool): whether to use bias correction.
            params (torch.Tensor): parameter.
            grads (torch.Tensor): gradient.
            exp_avg (torch.Tensor): exp average.
            exp_avg_sq (torch.Tensor): exp average square.
            loss_scale (float): loss scale value.
    """

    def __init__(self):
        super().__init__(
            extension_map=OrderedDict(
                arm=ArmCPUAdamExtension,
                x86=X86CPUAdamExtension,
            ),
            supported_device=["cpu"],
        )

    def fetch_kernel(self):
        if platform.machine() == "x86_64":
            kernel = self._extension_map["x86"]().fetch()
        elif platform.machine() in ["aarch64", "aarch64_be", "armv8b", "armv8l"]:
            kernel = self._extension_map["arm"]().fetch()
        else:
            raise Exception("not supported")
        return kernel
