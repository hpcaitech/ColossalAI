import os
import argparse
import warnings

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.optim import SGD, Adam, RMSprop, Optimizer
from torch._C._distributed_rpc import _is_current_rpc_agent_set
from colorama import Back, Style

rpc_is_initialized = _is_current_rpc_agent_set


def color_debug(text, prefix=' ', color='blue'):
    color = color.upper()
    print(getattr(Back, color), prefix, Style.RESET_ALL, text)


class RpcTestModel(nn.Module):

    def __init__(self, stage_id, actual_stage_num, feat_num, h) -> None:
        super().__init__()
        self.rank = stage_id
        self.is_last_rank = stage_id == actual_stage_num - 1
        self.linear_name = f'linear_{stage_id}'
        if stage_id == 0:
            setattr(self, self.linear_name, nn.Linear(feat_num, h))
        elif stage_id == actual_stage_num - 1:
            setattr(self, self.linear_name, nn.Linear(h, 1))
        else:
            setattr(self, self.linear_name, nn.Linear(h, h))

    def forward(self, x) -> torch.Tensor:
        linear: nn.Module = getattr(self, self.linear_name)
        out: torch.Tensor = linear(x)

        if self.is_last_rank:
            out = out.sum()
        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--num_microbatches', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'RMSprop'], default='SGD')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29020')
    parser.add_argument('--num_worker_threads', type=str, default=128)
    return parser.parse_args()


def pg_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--dp_degree', type=int, default=2)
    parser.add_argument('--tp_degree', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--num_worker_threads', type=str, default=128)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29020')
    return parser.parse_args()


def run_worker(rank, args, master_func):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # config rpc
    # if cuda is used, set_device_map is a must is configured
    # for cuda is not supported in torch rpc by default
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.num_worker_threads)

    world_size = args.world_size
    for rank_idx in range(world_size):
        options.set_device_map(f'work{rank_idx}', {rank: rank_idx})

    rpc.init_rpc(name=f'work{rank}', rank=rank, world_size=world_size, rpc_backend_options=options)

    # in rpc mode, only rank 0 is needed to be coded
    if rank == 0:
        master_func(args)
    # barrier here
    if rpc_is_initialized():
        rpc.shutdown()
    else:
        warnings.warn("RPC has not been initialized")


def rpc_run(args, master_func):
    world_size = args.world_size
    assert args.num_microbatches >= args.world_size, "num_microbatches cannot be fewer than world_size!"
    mp.spawn(run_worker, args=(args, master_func), nprocs=world_size)
