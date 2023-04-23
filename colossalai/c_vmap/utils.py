from typing import Union

import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

#TODO accept nestwed inputs cmap(f)((x,y),z)
def data_frag(*args, in_dims: Union[int, tuple], num_devices: int, **kwargs):
    new_args = [[] for _ in range(num_devices)]
    new_kwargs = {i: {} for i in range(num_devices)}
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

    for k, v in kwargs.items():
        for i in range(num_devices):
            new_kwargs[i][k] = v
    return new_args, new_kwargs


def data_to_device(*args, raw_pt=False, **kwargs):
    new_args = []
    if raw_pt:
        for a in args:
            new_args.append(a.to(torch.cuda.current_device()))
        for k, v in kwargs.items():
            kwargs[k] = v.to(torch.cuda.current_device())
    else:
        for a in args:
            new_args.append(a.to(gpc.get_global_rank(ParallelMode.GLOBAL)))

        for k, v in kwargs.items():
            kwargs[k] = v.to(gpc.get_global_rank(ParallelMode.GLOBAL))
    return new_args, kwargs

def process_needed_for_function(func):
    pass
