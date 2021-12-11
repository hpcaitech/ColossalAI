import math

from torch import Tensor
from torch.nn import init as init


def init_weight_(tensor: Tensor, fan_in: int, fan_out: int = None, init_method: str = 'torch'):
    if init_method == 'torch':
        a = math.sqrt(5)
        nonlinearity = 'leaky_relu'
        std = init.calculate_gain(nonlinearity, a) / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        init.uniform_(tensor, -bound, bound)
    elif init_method == 'jax':
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        init.uniform_(tensor, -a, a)
    elif init_method == 'jax_embed':
        std = math.sqrt(1.0 / fan_in)
        init.trunc_normal_(tensor, std=std / .87962566103423978)
    elif init_method == 'zero':
        init.zeros_(tensor)

def init_bias_(tensor: Tensor, fan_in: int, init_method: str = 'torch'):
    if init_method == 'torch':
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(tensor, -bound, bound)
    elif init_method == 'jax':
        init.normal_(tensor, std=1e-6)
    elif init_method == 'jax_embed':
        init.trunc_normal_(tensor, std=.02)
    elif init_method == 'zero':
        init.zeros_(tensor)
