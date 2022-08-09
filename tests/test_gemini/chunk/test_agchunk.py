import torch
import colossalai
import pytest
import torch.multiprocessing as mp
from functools import partial
from colossalai.testing import rerun_if_address_is_in_use, parameterize
from colossalai.utils import free_port, get_current_device
from colossalai.tensor import ProcessGroup as ColoProcessGroup
from colossalai.tensor import ColoParameter
from colossalai.gemini.ag_chunk import AgChunk


def add_param(param_list, param_cp_list, *args, **kwargs):
    param = ColoParameter(torch.empty(*args, **kwargs))
    param_list.append(param)
    param_cp_list.append(param.clone())


def check_euqal(param, param_cp):
    if param.device != param_cp.device:
        temp = param.data.to(param_cp.device)
    else:
        temp = param.data
    return torch.equal(temp, param_cp.data)


@parameterize('init_device', [None, torch.device('cpu')])
@parameterize('keep_gathered', [True, False])
@parameterize('pin_memory', [True, False])
def exam_chunk_init(init_device, keep_gathered, pin_memory):
    world_size = torch.distributed.get_world_size()
    pg = ColoProcessGroup()
    my_chunk = AgChunk(
        chunk_size=1024,
        process_group=pg,
        dtype=torch.float32,
        init_device=init_device,
        keep_gathered=keep_gathered,
        pin_memory=pin_memory
    )

    param_list = []
    param_cp_list = []

    add_param(param_list, param_cp_list, 8, 8, 8, device='cuda')
    add_param(param_list, param_cp_list, 4, 4)
    add_param(param_list, param_cp_list, 4, 8, 2, device='cuda')
    add_param(param_list, param_cp_list, 1, 1, 5)

    for param in param_list:
        my_chunk.append_tensor(param)
    assert my_chunk.utilized_size == 597
    for param, param_cp in zip(param_list, param_cp_list):
        check_euqal(param, param_cp)
    my_chunk.close_chunk()

    if keep_gathered is False:
        assert my_chunk.cpu_shard.size(0) == 1024 // world_size
        my_chunk.shard_move(get_current_device())

    my_chunk.access_chunk()

    for param, param_cp in zip(param_list, param_cp_list):
        check_euqal(param, param_cp)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_chunk_init()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
@rerun_if_address_is_in_use()
def test_chunk_function(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_function(2)
