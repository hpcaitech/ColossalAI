import contextlib
from typing import Callable

import torch
import torch.nn.functional as F

from colossalai.moe.manager import MOE_MANAGER
from colossalai.utils import get_current_device


class ForceFP32Parameter(torch.nn.Parameter):

    def half(self, memory_format=None):
        return self.data.clone()


class NormalNoiseGenerator:
    """Generates a random noisy mask for logits tensor.

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
    """Generates a random noisy mask for logits tensor.
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
    return F.softmax(logit, dim=dim, detype=torch.float32)


def get_noise_generator(noise_type: str, num_experts: int) -> Callable:
    if noise_type is None:
        return None
    elif noise_type == 'Jitter':
        noisy_func = UniformNoiseGenerator()
    elif noise_type == 'Gaussian':
        noisy_func = NormalNoiseGenerator(num_experts)
    else:
        raise NotImplementedError("Unsupported input noisy policy")
    return noisy_func


def get_activation(act: str) -> Callable:
    if act is None or act == 'relu':
        return torch.nn.ReLU()
    elif act == 'gelu':
        return torch.nn.GELU()
    elif act == 'swiglu':
        return SwiGLU
    else:
        raise NotImplementedError("Unsupported activation function")


def SwiGLU(x):
    """Gated linear unit activation function.
    Args:
        x : input array
        axis: the axis along which the split should be computed (default: -1)
    """
    size = x.shape[-1]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = torch.split(x, size // 2, -1)
    return x1 * (x2 * torch.sigmoid(x2))


@contextlib.contextmanager
def skip_init():
    """
    skip param random init
    """

    def _skip_init(x, *args, **kwargs):
        return x

    # __enter__
    fn_saved = []
    init_fn_list = [
        torch.nn.init.constant_, torch.nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.xavier_uniform_,
        torch.nn.init.xavier_normal_, torch.nn.init.kaiming_uniform_, torch.nn.init.kaiming_normal_
    ]
    for fn in init_fn_list:
        fn_saved.append(fn)
        fn = _skip_init

    yield

    # __exit__
    for fn, fn_saved in zip(init_fn_list, fn_saved):
        fn = fn_saved
    return
