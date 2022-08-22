import os
import argparse

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from colossalai.pipeline.rpc.PipelineBase import FillDrainPipelineEngine, OneFOneBPipelineEngine


class TestModel(nn.Module):

    def __init__(self, rank, world_size, feat_num, h) -> None:
        super().__init__()
        self.rank = rank
        self.is_last_rank = rank == world_size - 1
        self.linear_name = f'linear_{rank}'
        if rank == 0:
            setattr(self, self.linear_name, nn.Linear(feat_num, h))
        elif rank == world_size - 1:
            setattr(self, self.linear_name, nn.Linear(h, 1))
        else:
            setattr(self, self.linear_name, nn.Linear(h, h))

    def forward(self, x) -> torch.Tensor:
        linear: nn.Module = getattr(self, self.linear_name)
        out: torch.Tensor = linear(x)

        if self.is_last_rank:
            out = out.sum()
        return out


def run_main(args):
    torch.manual_seed(100)

    sample_num = 128
    feat_num = 10000
    h = 10000
    device = args.device
    world_size = args.world_size
    batch_size = 128
    assert sample_num % batch_size == 0
    batch_num = sample_num // batch_size
    num_microbatches = world_size

    input_sample = torch.randn((sample_num, feat_num), device=device)

    module_partitions = [TestModel(rank, world_size, feat_num, h) for rank in range(world_size)]

    engine = OneFOneBPipelineEngine(module_partitions=module_partitions,
                                    chunk=1,
                                    world_size=world_size,
                                    num_microbatches=num_microbatches,
                                    device=args.device,
                                    max_outstanding=world_size,
                                    use_interleave=False,
                                    checkpoint=False)

    for i in range(batch_num):
        batch = input_sample[i * batch_size:(i + 1) * batch_size]
        engine.forward_backward(batch)


def run_worker(rank, args):
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
        run_main(args)
    # barrier here
    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29020')
    parser.add_argument('--num_worker_threads', type=str, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size
    assert args.device in ['cpu', 'cuda'], "device must be cpu or cuda!"
    mp.spawn(run_worker, args=(args,), nprocs=world_size)
