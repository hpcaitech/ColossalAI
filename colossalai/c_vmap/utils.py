import torch
from colossalai.core import global_context as gpc

def data_frag(*args, in_dims: int, num_devices: int, **kwargs):
    new_args = [[] for _ in range(num_devices)]
    new_kwargs = {i:{} for i in range(num_devices)}
    for a in args:
        if not isinstance(a, torch.Tensor):
            raise TypeError("Only tensors can be mapped")
        a = torch.tensor_split(a, num_devices, dim=in_dims)
        for i in range(num_devices):
            new_args[i].append(a[i])
    
    for k, v in kwargs.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError("Only tensors can be mapped")
        v = torch.tensor_split(v, num_devices, dim=in_dims)
        for i in range(num_devices):
            new_kwargs[i][k] = v[i]
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