import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.elixir.chunk import BlockSpec, Chunk, MemoryPool, TensorState
from colossalai.testing import run_on_environment_flag, spawn


def exam_chunk_functions(nproc, group):
    a = torch.randn(2, 64, device='cuda')
    copy_a = a.clone()
    b = torch.randn(2, 2, 128, device='cuda')
    copy_b = b.clone()
    c = torch.randn(128, device='cuda')
    copy_c = c.clone()
    d = torch.randn(4, 32, device='cuda')
    copy_d = d.clone()

    mp = MemoryPool('cuda')
    mp.allocate_public_blocks(block_num=1)

    chunk = Chunk(mp, 1024, torch.float, group)
    chunk.l2_norm_flag = True
    assert chunk.chunk_size == 1024
    assert chunk.chunk_dtype == torch.float
    assert chunk.shard_size == 1024 // nproc

    def check_tensors():
        assert torch.equal(a, copy_a)
        assert torch.equal(b, copy_b)
        assert torch.equal(c, copy_c)
        assert torch.equal(d, copy_d)

    chunk.append_tensor(a)
    chunk.append_tensor(b)
    chunk.append_tensor(c)
    chunk.append_tensor(d)
    check_tensors()

    chunk.close_chunk()
    assert chunk.is_replica is False

    # check function: get_cpu_copy
    cpu_copys = chunk.get_cpu_copy()
    for t_gpu, t_cpu in zip([copy_a, copy_b, copy_c, copy_d], cpu_copys):
        assert t_cpu.device.type == 'cpu'
        assert torch.equal(t_gpu.cpu(), t_cpu)

    # check function: access_chunk
    block = mp.pop_public_block()
    chunk.access_chunk(block)
    assert chunk.is_replica
    assert chunk.scatter_check
    check_tensors()

    # check function: release_chunk
    chunk.optim_sync_flag = False
    block = chunk.release_chunk()
    assert block in mp.public_used_blocks
    assert chunk.is_replica is False
    assert chunk.optim_sync_flag is True

    # check function: access_chunk after release_chunk
    chunk.access_chunk(block)
    check_tensors()

    # check function: reduce_chunk
    norm = block.payload.float().norm(2)**2
    chunk.reduce_chunk()
    assert chunk.is_replica is False
    assert chunk.tensor_state_cnter[TensorState.HOLD] == 4

    test_norm = torch.Tensor([chunk.l2_norm]).cuda()
    dist.all_reduce(test_norm)
    assert torch.allclose(norm, test_norm)

    torch.cuda.synchronize()
    print('chunk functions are ok')


def exam_chunk_states(nproc, group):
    a = torch.randn(2, 64, device='cuda')
    copy_a = a.clone()
    b = torch.randn(2, 2, 128, device='cuda')
    copy_b = b.clone()
    c = torch.randn(128, device='cuda')
    copy_c = c.clone()
    d = torch.randn(4, 32, device='cuda')
    copy_d = d.clone()

    mp = MemoryPool('cuda')

    private_block_specs = [BlockSpec(1024, torch.float)]
    mp.allocate_private_blocks(private_block_specs)

    chunk = Chunk(mp, 1024, torch.float, group, rcache_fused=True)
    assert chunk.chunk_size == 1024
    assert chunk.chunk_dtype == torch.float
    assert chunk.shard_size == 1024 // nproc

    def check_tensors():
        assert torch.equal(a, copy_a)
        assert torch.equal(b, copy_b)
        assert torch.equal(c, copy_c)
        assert torch.equal(d, copy_d)

    chunk.append_tensor(a)
    chunk.append_tensor(b)
    chunk.append_tensor(c)
    chunk.append_tensor(d)
    check_tensors()

    chunk.close_chunk()
    assert chunk.is_replica is False

    chunk.access_chunk()
    assert chunk.is_replica
    check_tensors()

    assert chunk.tensor_state_cnter[TensorState.HOLD] == 4
    chunk.tensor_trans_state(a, TensorState.COMPUTE)
    assert chunk.tensor_state_cnter[TensorState.HOLD] == 3
    assert chunk.tensor_state_cnter[TensorState.COMPUTE] == 1

    tensor_list = [a, b, c, d]
    for t in tensor_list:
        chunk.tensor_trans_state(t, TensorState.COMPUTE)
        chunk.tensor_trans_state(t, TensorState.HOLD_AFTER_BWD)
        chunk.tensor_trans_state(t, TensorState.READY_FOR_REDUCE)
    assert chunk.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == 4
    assert chunk.reduce_check

    torch.cuda.synchronize()
    print('chunk states are ok')


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    exam_chunk_functions(nproc=world_size, group=dist.GroupMember.WORLD)
    exam_chunk_states(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_chunk_functions(world_size):
    spawn(run_dist, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_functions(world_size=4)
