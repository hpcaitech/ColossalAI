from typing import Union

import torch

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


# TODO accept nestwed inputs cmap(f)((x,y),z)
def data_frag(*args, in_dims: Union[int, tuple], num_devices: int):
    """This fucntion splits the input arguments across a given dimension.

    Args:
        args (tuple): Input arguments
        in_dims (int or tuple): The dimension(s) to split across
        num_devices (int): The number of devices to split across

    Returns:
        list: The split arguments
    """
    new_args = [[] for _ in range(num_devices)]
    if isinstance(in_dims, int):
        for a in args:
            if not isinstance(a, torch.Tensor):
                raise TypeError("Only tensors can be mapped")
            a = torch.tensor_split(a, num_devices, dim=in_dims)
            for i in range(num_devices):
                new_args[i].append(a[i])
    else:
        if len(in_dims) != len(args):
            raise ValueError("Number of in_dims must match number of args")
        for d, a in zip(in_dims, args):
            if not isinstance(a, torch.Tensor):
                raise TypeError("Only tensors can be mapped")
            if d != None:
                a = torch.tensor_split(a, num_devices, dim=d)
                for i in range(num_devices):
                    new_args[i].append(a[i])
            else:
                for i in range(num_devices):
                    new_args[i].append(a)

    return new_args


def data_to_device(*args):
    new_args = []
    for a in args:
        new_args.append(a.to(torch.cuda.current_device()))
    return new_args, kwargs


def scalar_to_vec(*args):
    return tuple(map(lambda x: x.unsqueeze(0) if x.dim() == 0 else x, args))
