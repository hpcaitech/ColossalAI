from typing import Callable
from packaging import version

import torch
import torch.distributed as dist

from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.communication.collective import all_gather

from .utils import data_frag, data_to_device


def naive_vmap(func: Callable, in_dims: int, out_dims: int): #add option for process group
    def wrap(*args, **kwargs):
        n = args[0].shape[in_dims]
        new_args = [[] for _ in range(n)]
        new_kwargs = {i:{} for i in range(n)}
        for a in args:
            if not isinstance(a, torch.Tensor):
                raise TypeError("Only tensors can be mapped")
            a = torch.tensor_split(a, n, dim=in_dims)
            for i in range(n):
                new_args[i].append(a[i])
        
        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError("Only tensors can be mapped")
            v = torch.tensor_split(v, n, dim=in_dims)
            for i in range(n):
                new_kwargs[i][k] = v[i]
        
        streams = [torch.cuda.current_stream() for _ in range(n)]
        result = []

        for i in range(n):
            with torch.cuda.stream(streams[i]):
                result.append(func(*new_args[i], **new_kwargs[i]))
        torch.cuda.synchronize()

        return torch.cat(result, dim=out_dims)
    
    return wrap
        


def cmap(func: Callable, in_dims: int, out_dims: int, raw_pt:bool = False, group=None, dst=-1): #add option for process group
    
    if version.parse(torch.__version__) < version.parse("2.0"):
        map_fn = naive_vmap
    else:
        map_fn = torch.vmap
    
    def wrap_raw(*args, **kwargs):
        num_devices = dist.get_world_size(group=group)
        rank = dist.get_rank()

        if num_devices == 1:
            return map_fn(func, in_dims=in_dims, out_dims=out_dims)(*args, **kwargs)
        
        new_args, new_kwargs = data_frag(*args, in_dims=in_dims, num_devices=num_devices, **kwargs)
        data_to_device(*new_args[rank], raw_pt=raw_pt, **new_kwargs[rank])

        func_out = map_fn(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])
        #print(func_out)
        output_empties = [torch.zeros_like(func_out) for _ in range(num_devices)]
        if dst == -1:
            dist.all_gather(output_empties, func_out, group=group)
        else:
            dist.gather(func_out, output_empties, dst, group=group)
        torch.cuda.synchronize()
        
        return torch.cat(output_empties, dim=out_dims)
    

    def ColWrap(*args, **kwargs):
        num_devices = gpc.get_global_rank(ParallelMode.GLOBAL)
        rank = gpc.get_global_rank(ParallelMode.GLOBAL)

        new_args, new_kwargs = data_frag(*args, in_dims=in_dims, out_dims=out_dims, num_devices=num_devices, **kwargs)
        data_to_device(*new_args[rank], raw_pt=raw_pt, **new_kwargs[rank])

        func_out = map_fn(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])
        out = all_gather(tensor=func_out,dim=out_dims, parallel_mode=ParallelMode.GLOBAL)
        torch.cuda.synchronize()
        return out

    if raw_pt:
        return wrap_raw
    else:
        return ColWrap
    
if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")
    a = torch.arange(1000).reshape(10,10,10).cuda()
    b = torch.arange(1000).reshape(10,10,10).cuda()
    cmaped_fn = cmap(lambda x,y: x*y, in_dims=1, out_dims=1, raw_pt=True, dst=-1)
    out = cmaped_fn(a,b)
    if dist.get_rank() == 0:
        print(out)
