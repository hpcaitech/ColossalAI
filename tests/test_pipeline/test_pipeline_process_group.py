import os

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from colossalai.pipeline.pipeline_process_group import PipelineProcessGroup
from rpc_test_utils import test_pg_parse_args, rpc_is_initialized


def run_worker(rank, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    device = args.device
    world_size = args.world_size
    dp_degree = args.dp_degree
    tp_degree = args.tp_degree
    num_worker_threads = args.num_worker_threads

    pg = PipelineProcessGroup(rank=rank,
                              world_size=world_size,
                              dp_degree=dp_degree,
                              tp_degree=tp_degree,
                              num_worker_threads=num_worker_threads,
                              device=device)

    if rpc_is_initialized():
        rpc.shutdown()


if __name__ == "__main__":
    args = test_pg_parse_args()
    world_size = args.world_size
    mp.spawn(run_worker, args=(args,), nprocs=world_size)
