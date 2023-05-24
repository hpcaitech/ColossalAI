import os
from functools import partial

import pytest
import torch
import torch.distributed as dist

from colossalai.elixir.chunk import BlockRequire, ChunkGroup, MemoryPool, TensorState
from colossalai.elixir.utils import init_distributed


def exam_chunk_group_functions(nproc, group):
    a = torch.randn(3, 64, device='cuda')
    copy_a = a.clone()
    b = torch.randn(2, 32, device='cuda')
    copy_b = b.clone()
    c = torch.randn(256, device='cuda')
    copy_c = c.clone()
    d = torch.randn(2, 2, 64, device='cuda')
    copy_d = d.clone()
    e = torch.randn(2, 33, device='cuda')
    copy_e = e.clone()

    mp = MemoryPool('cuda')
    mp.allocate(public_block_size=256, public_block_number=2, private_block_list=[BlockRequire(68, torch.float)])
    cg = ChunkGroup(rcache=mp)
    c0 = cg.allocate_chunk([a, b], 256, torch.float, group)
    c1 = cg.allocate_chunk([c], 256, torch.float, group)
    c2 = cg.allocate_chunk([d], 256, torch.float, group)

    fused_config = dict(rcache_fused=True)
    c3 = cg.allocate_chunk([e], 68, torch.float, group, fused_config)

    def check_chunk_0():
        assert torch.equal(a, copy_a)
        assert torch.equal(b, copy_b)

    def check_chunk_1():
        assert torch.equal(c, copy_c)

    def check_chunk_2():
        assert torch.equal(d, copy_d)

    def check_chunk_3():
        assert torch.equal(e, copy_e)

    # check tensors_to_chunks
    chunks = cg.tensors_to_chunks([e, a])
    assert chunks[0] == c0
    assert chunks[1] == c3
    # check access_chunk for unfused chunks
    cg.access_chunk(c0)
    cg.access_chunk(c1)
    check_chunk_0()
    check_chunk_1()
    assert not cg.rcache_enough_check(c2)
    assert cg.rcache_enough_check(c3)
    # check access_chunk for fused chunks
    cg.access_chunk(c3)
    check_chunk_3()
    # check release_chunk for unfused chunks
    cg.release_chunk(c1)
    assert cg.rcache_enough_check(c2)
    # check access_chunk
    cg.access_chunk(c2)
    check_chunk_2()

    cg.tensor_trans_state(e, TensorState.COMPUTE)
    cg.tensor_trans_state(e, TensorState.HOLD_AFTER_BWD)
    cg.tensor_trans_state(e, TensorState.READY_FOR_REDUCE)
    cg.reduce_chunk(c3)
    assert not c3.is_replica

    torch.cuda.synchronize()
    print('chunk group functions are ok')


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_chunk_group_functions(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_chunk_group(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_group(world_size=2)
