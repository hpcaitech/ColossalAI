import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.elixir.chunk import BlockSpec, ChunkGroup, MemoryPool, TensorState
from colossalai.testing import run_on_environment_flag, spawn


def exam_chunk_group_functions(group):
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
    mp.allocate_public_blocks(block_num=2, block_spec=BlockSpec(numel=256, dtype=torch.float))
    mp.allocate_private_blocks([BlockSpec(68, torch.float)])

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


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    exam_chunk_group_functions(group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_chunk_group(world_size):
    spawn(run_dist, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_group(world_size=2)
