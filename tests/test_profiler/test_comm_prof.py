from functools import partial
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import colossalai
from colossalai.utils import free_port, get_current_device
from colossalai.utils.profiler import enable_communication_prof, communication_prof_show

BATCH_SIZE = 1024
D_MODEL = 1024
CONFIG = dict(parallel=dict(tensor=dict(mode='1d', size=4)))


def run_test(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    inputs = torch.randn(BATCH_SIZE, D_MODEL, dtype=torch.float32, device=get_current_device())
    outputs = torch.empty(world_size, BATCH_SIZE, D_MODEL, dtype=torch.float32, device=get_current_device())
    outputs_list = list(torch.chunk(outputs, chunks=world_size, dim=0))

    enable_communication_prof()

    op = dist.all_reduce(inputs, async_op=True)
    dist.all_gather(outputs_list, inputs)
    op.wait()
    dist.reduce_scatter(inputs, outputs_list)
    dist.broadcast(inputs, 0)
    dist.reduce(inputs, 0)

    if rank == 0:
        communication_prof_show()


def test_cc_prof():
    world_size = 4
    run_func = partial(run_test, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_cc_prof()
