import os

import torch.distributed.rpc as rpc
from rpc_test_utils import pg_parse_args, rpc_is_initialized

from colossalai.legacy.initialize import launch
from colossalai.legacy.pipeline.pipeline_process_group import ppg
from colossalai.logging import disable_existing_loggers
from colossalai.testing import spawn


def run_worker(rank, args):
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

    if rpc_is_initialized():
        rpc.shutdown()


if __name__ == "__main__":
    args = pg_parse_args()
    world_size = args.world_size
    spawn(run_worker, world_size, args=args)
