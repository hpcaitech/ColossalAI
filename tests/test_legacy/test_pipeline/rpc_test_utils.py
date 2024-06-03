import argparse
import os
import warnings

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch import nn
from torch._C._distributed_rpc import _is_current_rpc_agent_set

from colossalai.legacy import launch
from colossalai.legacy.pipeline.pipeline_process_group import ppg
from colossalai.logging import disable_existing_loggers

rpc_is_initialized = _is_current_rpc_agent_set


def color_debug(text, prefix=" ", color="blue"):
    color = color.upper()
    print(getattr(Back, color), prefix, Style.RESET_ALL, text)


class MLP(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


class DAG_MLP(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dag_layer = nn.Linear(dim, dim, bias=False)

        for _ in range(layers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
            y = self.dag_layer(y)
        return x.sum(), y.sum()


class RpcTestModel(nn.Module):
    def __init__(self, stage_id, actual_stage_num, feat_num, h) -> None:
        super().__init__()
        self.rank = stage_id
        self.is_last_rank = stage_id == actual_stage_num - 1
        self.linear_name = f"linear_{stage_id}"

        if stage_id == 0:
            linear = nn.Linear(feat_num, h)
        elif stage_id == actual_stage_num - 1:
            linear = nn.Linear(h, 1)
        else:
            linear = nn.Linear(h, h)

        setattr(self, self.linear_name, linear)

    def forward(self, x) -> torch.Tensor:
        linear: nn.Module = getattr(self, self.linear_name)
        out: torch.Tensor = linear(x)

        if self.is_last_rank:
            out = out.sum()
        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dp_degree", type=int, default=1)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--num_microbatches", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "RMSprop"], default="SGD")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29020")
    parser.add_argument("--num_worker_threads", type=str, default=128)
    return parser.parse_args()


def pg_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--dp_degree", type=int, default=2)
    parser.add_argument("--tp_degree", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--num_worker_threads", type=str, default=128)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29020")
    return parser.parse_args()


def run_worker(rank, args, master_func):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    device = args.device
    world_size = args.world_size
    dp_degree = args.dp_degree
    tp_degree = args.tp_degree
    num_worker_threads = args.num_worker_threads
    host = args.master_addr
    port = args.master_port
    backend = "nccl" if device == "cuda" else "gloo"

    disable_existing_loggers()

    launch(dict(), rank, world_size, host, int(port), backend, verbose=False)
    ppg.set_global_info(
        rank=rank,
        world_size=world_size,
        dp_degree=dp_degree,
        tp_degree=tp_degree,
        num_worker_threads=num_worker_threads,
        device=device,
    )

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
