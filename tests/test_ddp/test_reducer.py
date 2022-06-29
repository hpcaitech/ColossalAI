import pytest
import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from functools import partial
from colossalai.nn.parallel.reducer import Reducer
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

REDUCE_CNT = 0


def check_eq(grad, grad_clone):
    global REDUCE_CNT
    print(f'Rank{dist.get_rank()} check {REDUCE_CNT}')
    REDUCE_CNT += 1
    assert torch.allclose(grad, grad_clone)


def run_reducer():
    grads = [torch.rand(64, i + 1, device=get_current_device()) for i in range(10)]
    grads_clone = [g.clone().detach() for g in grads]
    for g in grads:
        dist.all_reduce(g)
    reducer = Reducer(bucket_size_mb=1)
    for g, g_clone in zip(grads, grads_clone):
        reducer.all_reduce_async(g_clone, _get_default_group(), partial(check_eq, g))
    reducer.flush()


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_reducer()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_reducer(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_reducer(2)
