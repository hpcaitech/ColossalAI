import argparse
import os
import warnings
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch._C._distributed_rpc import _is_current_rpc_agent_set
from torch.futures import Future

from colossalai.initialize import launch
from colossalai.pipeline.pipeline_process_group import ppg


def pyobj_map(obj: Any, fn: Callable, process_types: Union[Type, Tuple[Type]] = ()) -> Any:
    if isinstance(obj, process_types):
        return fn(obj)
    elif type(obj) is dict:
        return {k: pyobj_map(obj[k], fn, process_types) for k in obj}
    elif type(obj) is tuple:
        return tuple(pyobj_map(o, fn, process_types) for o in obj)
    elif type(obj) is list:
        return list(pyobj_map(o, fn, process_types) for o in obj)
    else:
        return obj


def pytree_map(obj: Any, fn: Callable, process_types: Union[Type, Tuple[Type]] = (), map_all: bool = False) -> Any:
    """process object recursively, like pytree

    Args:
        obj (:class:`Any`): object to process
        fn (:class:`Callable`): a function to process subobject in obj
        process_types (:class: `type | tuple[type]`): types to determine the type to process
        map_all (:class: `bool`): if map_all is True, then any type of element will use fn

    Returns:
        :class:`Any`: returns have the same structure of `obj` and type in process_types after map of `fn`
    """
    if isinstance(obj, dict):
        return {k: pytree_map(obj[k], fn, process_types, map_all) for k in obj}
    elif isinstance(obj, tuple):
        return tuple(pytree_map(o, fn, process_types, map_all) for o in obj)
    elif isinstance(obj, list):
        return list(pytree_map(o, fn, process_types, map_all) for o in obj)
    elif isinstance(obj, process_types):
        return fn(obj)
    else:
        return fn(obj) if map_all else obj


def tensor_shape_list(obj):
    return pytree_map(obj, fn=lambda x: x.shape, process_types=torch.Tensor)


def get_batch_lengths(batch):
    lengths = []
    pytree_map(batch, fn=lambda x: lengths.append(len(x)), process_types=torch.Tensor)
    return lengths


def split_batch(batch: Any, start, stop, device: str):
    if device == 'cuda':
        fn = lambda x: x[start:stop].cuda()
    else:
        fn = lambda x: x[start:stop]
    return pytree_map(batch, fn=fn, process_types=torch.Tensor)


def type_detail(obj):
    return pytree_map(obj, lambda x: type(x), map_all=True)


def pytree_filter(fn, obj, process_types):
    if obj is None:
        return None

    filters = []

    def condition_append(obj):
        if fn(obj):
            filters.append(obj)

    pytree_map(obj, fn=condition_append, process_types=process_types)
    return filters


def get_real_args_kwargs(args_or_kwargs):
    args_or_kwargs = pytree_map(args_or_kwargs, fn=lambda x: x.wait(), process_types=Future)
    # TODO : combine producer and consumer
    # by default, merge all args in the output args or kwargs
    if args_or_kwargs is not None:
        if isinstance(args_or_kwargs, dict):
            pass
        else:
            flatten_args = []
            pytree_map(args_or_kwargs, fn=lambda x: flatten_args.append(x), map_all=True)
            args_or_kwargs = flatten_args

    return args_or_kwargs


def run_worker(rank, args, master_func):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    device = args.device
    world_size = args.world_size
    dp_degree = args.dp_degree
    tp_degree = args.tp_degree
    num_worker_threads = args.num_worker_threads
    host = args.master_addr
    port = args.master_port
    backend = 'nccl' if device == 'cuda' else 'gloo'

    launch(dict(), rank, world_size, host, int(port), backend, verbose=False)
    ppg.set_global_info(rank=rank,
                        world_size=world_size,
                        dp_degree=dp_degree,
                        tp_degree=tp_degree,
                        num_worker_threads=num_worker_threads,
                        device=device)
    ppg.args = args
    # in rpc mode, only rank 0 is needed to be coded
    if rank == 0:
        master_func(args)
    # barrier here
    if _is_current_rpc_agent_set():
        rpc.shutdown()
    else:
        warnings.warn("RPC has not been initialized")


def rpc_run(args, master_func):
    world_size = args.world_size
    mp.spawn(run_worker, args=(args, master_func), nprocs=world_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dp_degree', type=int, default=1)
    parser.add_argument('--tp_degree', type=int, default=1)
    parser.add_argument('--num_microbatches', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'RMSprop'], default='SGD')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29020')
    parser.add_argument('--num_worker_threads', type=int, default=128)
    return parser.parse_args()
