import torch
import torch.nn.functional as F
from colossalai.utils import get_current_device
from colossalai.context.moe_context import MOE_CONTEXT
from .experts import FFNExperts, TPExperts


class ForceFP32Parameter(torch.nn.Parameter):

    def half(self, memory_format=None):
        return self.data.clone()


class NormalNoiseGenerator:
    """Generates a random noisy mask for logtis tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        self.normal = torch.distributions.normal.Normal(loc=torch.tensor(0.0, device=get_current_device()),
                                                        scale=torch.tensor(1.0 / num_experts**2,
                                                                           device=get_current_device())).rsample

    def __call__(self, inputs: torch.Tensor):
        noisy = self.normal(inputs.shape)
        return inputs + noisy


class UniformNoiseGenerator:
    """Generates a random noisy mask for logtis tensor.
    copied from mesh tensorflow:
    Multiply values by a random number between :math:`1-epsilon` and :math:`1+epsilon`.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.

    Args:
        eps (float, optional): Epsilon in generator, defaults 1e-2.
    """

    def __init__(self, eps: float = 1e-2):
        self.uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - eps, device=get_current_device()),
                                                           high=torch.tensor(1.0 + eps,
                                                                             device=get_current_device())).rsample

    def __call__(self, inputs: torch.Tensor):
        noisy = self.uniform(inputs.shape)
        return inputs * noisy


def autocast_softmax(logit: torch.Tensor, dim: int):
    if logit.dtype != torch.float32:
        logit = logit.float()
    return F.softmax(logit, dim=dim)


def build_ffn_experts(num_experts: int, d_model: int, d_ff: int, activation=None, drop_rate: float = 0):
    mep_size = MOE_CONTEXT.max_ep_size
    if num_experts % mep_size == 0 or mep_size % num_experts == 0:
        return FFNExperts(num_experts, d_model, d_ff, activation, drop_rate)
    elif d_ff % mep_size == 0:
        return TPExperts(num_experts, d_model, d_ff, activation, drop_rate)
    else:
        raise NotImplementedError(f"Can not build {num_experts} experts in {mep_size} GPUS.")
