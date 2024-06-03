import pytest
import torch
from torch.distributed.distributed_c10d import _get_default_group

import colossalai
from colossalai.tensor import ColoTensor
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.gemini.chunk import ChunkManager

CUDA_MEM_0 = {False: 512, True: 1024}
CUDA_MEM_1 = {False: 0, True: 1024}
CPU_MEM = {True: {True: 0, False: 0}, False: {True: 512, False: 0}}


@parameterize("keep_gathered", [True, False])
@parameterize("pin_memory", [True, False])
def exam_chunk_memory(keep_gathered, pin_memory):
    params = [ColoTensor(torch.rand(8, 8)) for _ in range(3)]
    config = {2: dict(chunk_size=128, keep_gathered=keep_gathered)}

    chunk_manager = ChunkManager(config)
    assert chunk_manager.total_mem["cpu"] == 0
    assert chunk_manager.total_mem["cuda"] == 0

    process_group = _get_default_group()
    for p in params:
        chunk_manager.register_tensor(p, "param", 2, process_group, pin_memory=pin_memory)
    chunk_manager.close_all_groups()
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[keep_gathered]

    chunks = chunk_manager.get_chunks(params)

    for chunk in chunks:
        chunk_manager.access_chunk(chunk)
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[True]

    for chunk in chunks:
        chunk_manager.release_chunk(chunk)

    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[keep_gathered]

    for chunk in chunks:
        chunk_manager.move_chunk(chunk, torch.device("cpu"))
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][True]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_1[keep_gathered]


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_chunk_memory()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_chunk_manager(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_chunk_manager(2)
