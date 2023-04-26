from typing import Callable, Union

import torch
import torch.distributed as dist
from packaging import version

from colossalai.c_vmap.utils import data_frag, data_to_device, scalar_to_vec
from colossalai.communication.collective import all_gather, gather
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


class VersionError(Exception):
    pass


class CudaError(Exception):
    pass


# TODO add suport for multi-node cpu mapping
# TODO add support for grad functions
def cmap(func: Callable,
         in_dims: Union[int, tuple] = 0,
         out_dims: Union[int, tuple] = 0,
         raw_pt: bool = False,
         group=None,
         parallel_mode=ParallelMode.GLOBAL,
         dst=-1) -> Callable:
    """Colossal map designed to act like jax.pmap but to work with collassal AI tools(Gemini, Zero, etc)

    The purpose of cmap is to express single-program multiple-data programs. Wraping a function with cmap will split the
    input chunks and run each chunk on each cuda device. cmap is very similar to torch.vmap as both transformations map
    a function over array axes. For best performance all cuda devices should be identical. cmap can be used as an
    alternative to data parallelism

    Args:
        func: Fucntion to be mapped over argument axes, the function must return a tensor or multiple tensors
        in_dims: Specifies which dimension of the inputs should be mapped over. in_dims should have a structure like the
                 inputs. If the in_dim for a particular input is None, then that indicates there is no map dimension.
        out_dims: Specifies where the mapped dimension should appear in the outputs. If out_dims is a Tuple, then it
                  should have one element per output
        raw_pt: Whether the cmap is to be implmented using raw pytorch to be run with
                torch.distributed.init_process_group(...) or alongside colossal ai tools with colossalai.launch(...)
        group: The process group to work on. If None, the default process group will be used. for use when raw_pt=False
        parallel_mode: Parallel group mode used in this communication. for use when raw_pt=True
        dst: This kwarg determines whether the output array is scattered to all devices and gathered onto a single
             device with rank in proccess dst
    Returns:
        A parallelized version of func

    Usage:
    ---------------------------------------------
        x = torch.randn(64, 128)
        y = torch.randn(64, 128)
        batch_dot = cmap(torch.dot)
        batch_dot(x,y)
    ---------------------------------------------
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                return x

        model = Model()
        batched_model = cmap(model)
        input = torch.randn(64, 128)
        batched_model(input)
    ---------------------------------------------
        @cmap(in_dims=(0,None), dst=0)
        def batch_dot(x, y):
            return torch.dot(x, y)
        a = torch.randn(64, 128)
        b = torch.randn(128)
        out = batch_dot(a, b)  # None on all devices except 0
    ---------------------------------------------
    """

    if version.parse(torch.__version__) < version.parse("2.0"):
        raise VersionError(f"torch version: {torch.__version__}: in order to cmap your function torch2.0 is required")

    elif not torch.cuda.is_available:
        raise CudaError("cuda is not available. in order to cmap your function cuda is required")

    def wrap_raw(*args, **kwargs):
        rank = dist.get_rank(group=group)
        num_processes = dist.get_world_size(group=group)

        if rank == -1:
            return None

        if num_processes == 1:
            return torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*args, **kwargs)

        new_args, new_kwargs = data_frag(*args, in_dims=in_dims, num_devices=num_processes, **kwargs)
        data_to_device(*new_args[rank])
        func_out = torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])
        func_out = scalar_to_vec(*func_out)

        if dst == -1:
            if isinstance(func_out, tuple):
                output_empties = [[torch.empty_like(i) for _ in range(num_processes)] for i in func_out]
                if isinstance(out_dims, int):
                    out_dims = (out_dims,) * num_processes
                for i in range(len(output_empties)):
                    dist.all_gather(output_empties[i], func_out[i], group=group)
                    torch.cuda.synchronize()
                for i in range(len(out_dims)):
                    output_empties[i] = torch.cat(output_empties[i], dim=out_dims[i])
                return tuple(output_empties)
            else:
                output_empties = list([torch.empty_like(func_out) for _ in range(num_processes)])
                dist.all_gather(output_empties, func_out, group=group)
                torch.cuda.synchronize()
                return torch.cat(output_empties, dim=out_dims)

        else:
            if rank == dst:
                if isinstance(func_out, tuple):
                    output_empties = [[torch.empty_like(i) for _ in range(num_processes)] for i in func_out]
                    if isinstance(out_dims, int):
                        out_dims = (out_dims,) * num_processes
                    for i in range(len(output_empties)):
                        dist.gather(func_out[i], output_empties[i], dst=dst, group=group)
                        torch.cuda.synchronize()
                    for i in range(len(out_dims)):
                        output_empties[i] = torch.cat(output_empties[i], dim=out_dims[i])
                    return tuple(output_empties)
                else:
                    output_empties = list([torch.empty_like(func_out) for _ in range(num_processes)])
                    dist.gather(func_out, output_empties, dst=dst, group=group)
                    torch.cuda.synchronize()
                    return torch.cat(output_empties, dim=out_dims)
            else:
                if isinstance(func_out, tuple):
                    for i in range(len(func_out)):
                        dist.gather(func_out[i], dst=dst, group=group)
                        torch.cuda.synchronize()
                    return None
                else:
                    dist.gather(func_out, dst=dst, group=group)
                    torch.cuda.synchronize()
                    return None

    def ColWrap(*args, **kwargs):
        rank = gpc.get_local_rank(parallel_mode=parallel_mode)
        pg = gpc.get_group(parallel_mode=parallel_mode)
        num_processes = pg.size()

        if dist.get_rank(group=group) == -1:
            return None

        new_args, new_kwargs = data_frag(*args, in_dims=in_dims, out_dims=out_dims, num_devices=num_processes, **kwargs)
        data_to_device(*new_args[rank], raw_pt=raw_pt, **new_kwargs[rank])

        func_out = torch.vmap(func, in_dims=in_dims, out_dims=out_dims)(*new_args[rank], **new_kwargs[rank])
        func_out = scalar_to_vec(*func_out)

        if dst == -1:
            if isinstance(func_out, tuple):
                results = []
                if isinstance(out_dims, int):
                    out_dims = (out_dims,) * num_processes
                for i in range(len(func_out)):
                    results.append(all_gather(tensor=func_out[i], dim=out_dims[i], parallel_mode=parallel_mode))
                    torch.cuda.synchronize()
                return tuple(results)
            else:
                out = all_gather(tensor=func_out, dim=out_dims, parallel_mode=parallel_mode)
                torch.cuda.synchronize()
                return out
        else:
            if isinstance(func_out, tuple):
                results = []
                for i in range(len(func_out)):
                    results.append(gather(tensor=func_out[i], dim=out_dims[i], dst=dst, parallel_mode=parallel_mode))
                    torch.cuda.synchronize()
                return tuple(results)
            else:
                out = gather(tensor=func_out, dim=out_dims, dst=dst, parallel_mode=parallel_mode)
                torch.cuda.synchronize()
                return out

    if raw_pt:
        return wrap_raw
    else:
        return ColWrap
