import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.tensor import ColoParameter
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.gemini import TensorState
from colossalai.zero.gemini.chunk import Chunk


def dist_sum(x):
    temp = torch.tensor([x], device=get_accelerator().get_current_device())
    dist.all_reduce(temp)
    return temp.item()


def add_param(param_list, param_cp_list, *args, **kwargs):
    param = ColoParameter(torch.randn(*args, **kwargs))
    param_list.append(param)
    param_cp_list.append(param.clone())


def check_equal(param, param_cp):
    if param.device != param_cp.device:
        temp = param.data.to(param_cp.device)
    else:
        temp = param.data
    return torch.equal(temp, param_cp.data)


@parameterize("init_device", [None, torch.device("cpu")])
@parameterize("keep_gathered", [True, False])
@parameterize("pin_memory", [True, False])
@parameterize("async_op", [True, False])
def exam_chunk_basic(init_device, keep_gathered, pin_memory, async_op):
    world_size = torch.distributed.get_world_size()
    pg = _get_default_group()
    my_chunk = Chunk(
        chunk_size=1024,
        zero_group=pg,
        dtype=torch.float32,
        init_device=init_device,
        cpu_shard_init=True,
        keep_gathered=keep_gathered,
        pin_memory=pin_memory,
    )

    param_list = []
    param_cp_list = []

    add_param(param_list, param_cp_list, 8, 8, 8, device="cuda")
    add_param(param_list, param_cp_list, 4, 4)
    add_param(param_list, param_cp_list, 4, 8, 2, device="cuda")
    add_param(param_list, param_cp_list, 1, 1, 5)

    for param in param_list:
        my_chunk.append_tensor(param)
    assert my_chunk.utilized_size == 597
    for param, param_cp in zip(param_list, param_cp_list):
        check_equal(param, param_cp)
    my_chunk.close_chunk()

    if keep_gathered is False:
        assert my_chunk.cpu_shard.size(0) == 1024 // world_size
        assert my_chunk.device_type == "cpu"
        assert my_chunk.can_move
        my_chunk.shard_move(get_accelerator().get_current_device())
    else:
        assert my_chunk.cuda_global_chunk.size(0) == 1024
        assert my_chunk.device_type == "cuda"
        assert not my_chunk.can_move

    assert dist_sum(my_chunk.valid_end) == my_chunk.utilized_size
    flag = my_chunk.has_inf_or_nan
    assert not flag, "has_inf_or_nan is {}".format(flag)

    my_chunk.access_chunk()
    assert my_chunk.device_type == "cuda"
    for param, param_cp in zip(param_list, param_cp_list):
        check_equal(param, param_cp)

    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 4
    my_chunk.tensor_trans_state(param_list[0], TensorState.COMPUTE)
    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 3
    assert my_chunk.tensor_state_cnter[TensorState.COMPUTE] == 1
    assert not my_chunk.can_release

    for param in param_list:
        my_chunk.tensor_trans_state(param, TensorState.COMPUTE)
        my_chunk.tensor_trans_state(param, TensorState.HOLD_AFTER_BWD)
        my_chunk.tensor_trans_state(param, TensorState.READY_FOR_REDUCE)

    assert my_chunk.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == 4
    assert my_chunk.can_reduce
    my_chunk.reduce(async_op)
    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 4

    if async_op:
        my_chunk.wait_async_reduce()

    if keep_gathered is False:
        assert my_chunk.cuda_shard.size(0) == 1024 // world_size
        assert my_chunk.device_type == "cuda"
        assert my_chunk.can_move
    else:
        assert my_chunk.cuda_global_chunk.size(0) == 1024
        assert my_chunk.device_type == "cuda"
        assert not my_chunk.can_move


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_chunk_basic()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2, 4])
@rerun_if_address_is_in_use()
def test_chunk_function(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_chunk_function(4)
