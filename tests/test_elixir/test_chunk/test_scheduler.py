import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from colossalai.elixir.chunk import Chunk, MemoryPool
from colossalai.elixir.chunk.scheduler import FIFOScheduler, PrefetchScheduler
from colossalai.elixir.utils import init_distributed
from colossalai.testing import run_on_environment_flag


def exam_fifo(nproc, group):
    mp = MemoryPool('cuda')
    mp.allocate(public_block_number=1)
    c0 = Chunk(mp, 1024, torch.float, group)
    c1 = Chunk(mp, 1024, torch.float, group)
    c2 = Chunk(mp, 1024, torch.float, group)

    sdl = FIFOScheduler()
    sdl.reset()

    sdl.add(c0)
    sdl.add(c1)
    sdl.add(c2)
    sdl.add(c0)    # nothing happens here
    assert sdl.top() == c0

    sdl.remove(c0)
    assert sdl.top() == c1, f'{sdl.top()}'
    sdl.remove(c0)
    assert sdl.top() == c1, f'{sdl.top()}'

    sdl.add(c0)
    assert sdl.top() == c1
    sdl.remove(c1)
    assert sdl.top() == c2
    sdl.remove(c2)
    assert sdl.top() == c0


def exam_prefetch(nproc, group):
    mp = MemoryPool('cuda')
    mp.allocate()
    c0 = Chunk(mp, 1024, torch.float, group)
    c1 = Chunk(mp, 1024, torch.float, group)
    c2 = Chunk(mp, 1024, torch.float, group)

    chunk_called_per_step = [[c0], [c1], [c2], [c0], [c0], [c1], [c2], [c2], [c1], [c0]]

    sdl = PrefetchScheduler(chunk_called_per_step=chunk_called_per_step)
    print(sdl.next_step_dict)
    sdl.reset()

    sdl.step()
    sdl.add(c0)
    assert sdl.top() == c0

    sdl.step()
    sdl.add(c1)
    assert sdl.top() == c1

    sdl.step()
    sdl.add(c2)
    assert sdl.top() == c2

    sdl.remove(c0)
    sdl.step()
    sdl.add(c0)
    assert sdl.top() == c2

    sdl.remove(c0)
    sdl.step()
    sdl.add(c0)
    assert sdl.top() == c0
    sdl.remove(c0)    # notice here

    sdl.remove(c1)
    sdl.step()
    sdl.add(c1)
    assert sdl.top() == c1

    sdl.remove(c2)
    sdl.step()
    sdl.add(c2)
    assert sdl.top() == c1

    sdl.remove(c2)
    sdl.step()
    sdl.add(c2)
    assert sdl.top() == c2
    sdl.remove(c2)    # notice here
    sdl.add(c0)    # notice here

    sdl.remove(c1)
    sdl.step()
    sdl.add(c1)
    assert sdl.top() == c1
    sdl.remove(c1)    # notice here

    sdl.remove(c0)
    sdl.step()
    sdl.add(c0)
    assert sdl.top() == c0

    sdl.remove(c0)
    sdl.clear()


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_fifo(nproc=world_size, group=dist.GroupMember.WORLD)
    exam_prefetch(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@run_on_environment_flag('ELX')
def test_chunk_scheduler(world_size=1):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_scheduler()
