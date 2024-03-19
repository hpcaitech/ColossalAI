import copy
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
import colossalai.tensor.d_tensor.api as api
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import CAME
from colossalai.shardformer.layer import Linear1D_Col
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


def check_linear_1d_col(lazy_init: bool, seq_parallel: bool, overlap: bool, tp: int, zero: int):
    rtol, atol = 1e-3, 1e-3
    # create shardformer
    # ranks: [0, 1, 2, 3]
    # tp ranks = [0, 1], [2, 3]
    # dp ranks = [0, 2], [1, 3]
    dp_process_group_1 = dist.new_group([0, 2])
    dp_process_group_2 = dist.new_group([1, 3])
    tp_process_group_1 = dist.new_group([0, 1])
    tp_process_group_2 = dist.new_group([2, 3])
    rank = dist.get_rank()
    if rank == 0:
        tp_group = tp_process_group_1
        dp_group = dp_process_group_1
    if rank == 1:
        tp_group = tp_process_group_1
        dp_group = dp_process_group_2
    if rank == 2:
        tp_group = tp_process_group_2
        dp_group = dp_process_group_1
    if rank == 3:
        tp_group = tp_process_group_2
        dp_group = dp_process_group_2
    tp_rank = dist.get_rank(group=tp_group)
    dp_rank = dist.get_rank(group=dp_group)
    ctx = LazyInitContext() if lazy_init else nullcontext()
    linear = nn.Linear(32, 128).cuda()
    with ctx:
        linear_copy = nn.Linear(32, 128).cuda()
    linear_col = Linear1D_Col.from_native_module(
        linear_copy, process_group=tp_group, gather_output=True, seq_parallel=seq_parallel, overlap=overlap
    )
    api.get_device_mesh(linear_col.weight)
    api.get_sharding_spec(linear_col.weight)

    # ensure state dict is reversibly loadable
    linear.load_state_dict(linear_col.state_dict())
    linear_col.load_state_dict(linear.state_dict())
    with torch.no_grad():
        ori_dist_weight = copy.deepcopy(linear_col.weight)
        ori_target_weight = torch.chunk(linear.weight.clone(), tp, dim=0)[tp_rank]
        assert_close(ori_dist_weight, ori_target_weight)

    optim = CAME(linear.parameters(), lr=1e-3)
    dist_optim = CAME(linear_col.parameters(), lr=1e-3)

    # wrap with zero
    if zero > 1:
        optim = LowLevelZeroOptimizer(optimizer=optim, partition_grad=False, dp_process_group=dp_group)
        dist_optim = LowLevelZeroOptimizer(optimizer=dist_optim, partition_grad=False, dp_process_group=dp_group)

    # check computation correctness
    # [batch_size, seq_len, hidden_size]
    x = torch.rand(2, 4, 32).cuda()
    x_for_unshard = x.expand_as(x.clone())
    x_for_unshard.requires_grad_(True)
    x_for_shard = x.expand_as(x.clone()) if seq_parallel is False else torch.chunk(x.clone(), tp, dim=1)[tp_rank]
    x_for_shard.requires_grad_(True)

    out = linear(x_for_unshard)
    gather_out = linear_col(x_for_shard)
    assert_close(out, gather_out)

    if zero == 1:
        # check backward correctness
        out.sum().backward()
        gather_out.sum().backward()

        target_grad = torch.chunk(linear.weight.grad, tp, dim=0)[tp_rank]
        assert_close(target_grad, linear_col.weight.grad)
    else:
        out = out.sum()
        gather_out = gather_out.sum()

        optim.backward(out)
        dist_optim.backward(gather_out)

    # check optimizer correctness
    dist_optim.step()
    optim.step()
    dist_optim.zero_grad()
    optim.zero_grad()
    with torch.no_grad():
        target_weight = torch.chunk(linear.weight.clone(), tp, dim=0)[tp_rank]
    assert_close(target_weight, linear_col.weight)
    assert not torch.allclose(ori_target_weight, target_weight)
    assert not torch.allclose(ori_dist_weight, linear_col.weight)


@parameterize("lazy_init", [False])
@parameterize("seq_parallel", [False])
@parameterize("overlap", [True])
@parameterize("tp", [2])
@parameterize("zero", [1, 2])
def run_dist_linear_test(lazy_init, seq_parallel, overlap, tp, zero):
    check_linear_1d_col(lazy_init, seq_parallel, overlap, tp, zero)
    # check_linear_1d_row(lazy_init, seq_parallel)
    # check_linear_col_plus_row(lazy_init, seq_parallel, overlap)


def check_dist_linear(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_dist_linear_test()


@rerun_if_address_is_in_use()
def test_linear():
    spawn(check_dist_linear, nprocs=4)


if __name__ == "__main__":
    test_linear()
