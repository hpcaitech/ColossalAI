from typing import Callable, Union
from packaging import version

import torch
import torch.distributed as dist

from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.communication.collective import all_gather, gather
from colossalai.c_vmap.utils import data_frag, data_to_device


class VersionError(Exception):
    pass


class CudaError(Exception):
    pass


# TODO add process_group argument
# TODO add support for grad functions
def cmap(func: Callable,
         in_dims: Union[int, tuple] = 0,
         out_dims: Union[int, tuple] = 0,
         raw_pt: bool = False,
         group=None, dst=-1):

    if version.parse(torch.__version__) < version.parse("2.0"):
        raise VersionError(f"torch version: {torch.__version__}: in order to cmap your function torch2.0 is required")

    elif not torch.cuda.is_available:
        raise CudaError("cuda is not available. in order to cmap your function cuda is required")

    def wrap_raw(*args, **kwargs):
        num_devices = dist.get_world_size(group=group)
        rank = dist.get_rank()

        if num_devices == 1:
            return torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*args, **kwargs)

        new_args, new_kwargs = data_frag(*args,
                                         in_dims=in_dims,
                                         num_devices=num_devices,
                                         **kwargs)
        data_to_device(*new_args[rank],
                       raw_pt=raw_pt,
                       **new_kwargs[rank])
        func_out = torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])

        if dst == -1:
            if isinstance(func_out, tuple):
                output_empties = [[torch.zeros_like(i) for _ in range(num_devices)] for i in func_out]
                for i in range(len(output_empties)):
                    dist.all_gather(output_empties[i],
                                    func_out[i],
                                    group=group)
                    torch.cuda.synchronize()
                for i in range(len(out_dims)):
                    output_empties[i] = torch.cat(output_empties[i], dim=out_dims[i])
                return tuple(output_empties)
            else:
                output_empties = list([torch.zeros_like(func_out) for _ in range(num_devices)])
                dist.all_gather(output_empties,
                                func_out,
                                group=group)
                torch.cuda.synchronize()
                return torch.cat(output_empties, dim=out_dims)

        else:
            if rank == dst:
                if isinstance(func_out, tuple):
                    output_empties = [[torch.zeros_like(i) for _ in range(num_devices)] for i in func_out]
                    for i in range(len(output_empties)):
                        dist.gather(output_empties[i],
                                    func_out[i],
                                    dst=dst,
                                    group=group)
                        torch.cuda.synchronize()
                    for i in range(len(out_dims)):
                        output_empties[i] = torch.cat(output_empties[i], dim=out_dims[i])
                    return tuple(output_empties)
                else:
                    output_empties = list([torch.zeros_like(func_out) for _ in range(num_devices)])
                    dist.gather(output_empties,
                                func_out,
                                dst=dst,
                                group=group)
                    torch.cuda.synchronize()
                    return torch.cat(output_empties, dim=out_dims)
            else:
                if isinstance(func_out, tuple):
                    for i in range(len(func_out)):
                        dist.gather(func_out[i],
                                    dst=dst,
                                    group=group)
                        torch.cuda.synchronize()
                    return None
                else:
                    dist.gather(func_out,
                                dst=dst,
                                group=group)
                    torch.cuda.synchronize()
                    return None

    def ColWrap(*args, **kwargs):
        num_devices = gpc.get_global_rank()
        rank = gpc.get_global_rank()

        new_args, new_kwargs = data_frag(*args,
                                         in_dims=in_dims,
                                         out_dims=out_dims,
                                         num_devices=num_devices,
                                         **kwargs)
        data_to_device(*new_args[rank],
                       raw_pt=raw_pt,
                       **new_kwargs[rank])

        func_out = torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])

        if dst == -1:
            if isinstance(func_out, tuple):
                results = []
                for i in range(len(func_out)):
                    results.append(all_gather(tensor=func_out[i],
                                              dim=out_dims[i],
                                              parallel_mode=ParallelMode.GLOBAL))
                    torch.cuda.synchronize()
                return tuple(results)
            else:
                out = all_gather(tensor=func_out,
                                 dim=out_dims,
                                 parallel_mode=ParallelMode.GLOBAL)
                return out
        else:
            if isinstance(func_out, tuple):
                results = []
                for i in range(len(func_out)):
                    results.append(gather(tensor=func_out[i],
                                          dim=out_dims[i],
                                          dst=dst,
                                          parallel_mode=ParallelMode.GLOBAL))
                    torch.cuda.synchronize()
                return tuple(results)
            else:
                out = gather(tensor=func_out,
                             dim=out_dims,
                             dst=dst,
                             parallel_mode=ParallelMode.GLOBAL)
                return out

    if raw_pt:
        return wrap_raw
    else:
        return ColWrap


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")

    # colossalai.launch_from_torch(config={})
    a = torch.arange(10000).reshape(10, 10, 100).cuda()
    b = torch.arange(10000).reshape(10, 10, 100).cuda()
    c = torch.arange(120).reshape(2, 3, 4, 5)

    cmaped_fn = cmap(lambda x, y: x*y, in_dims=0, out_dims=2, raw_pt=True, dst=-1)
    out = cmaped_fn(a, b)
    if dist.get_rank() == 0:
        print(out)
        print(out.shape)

        # print(vmaped_fn(c))
        # print(vmaped_fn(c).shape)
