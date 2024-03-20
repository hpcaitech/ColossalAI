import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
import colossalai.tensor.d_tensor.api as api
from colossalai.nn.optimizer import CAME
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


def check_linear_1d_col(lazy_init: bool, seq_parallel: bool, overlap: bool, tp: int, zero: int, col: bool):
    rtol, atol = 1e-3, 1e-3
    # create shardformer
    # ranks: [0, 1, 2, 3]
    # tp ranks = [0, 1], [2, 3]
    # dp ranks = [0, 2], [1, 3]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    dp = world_size // tp
    assert world_size % (tp * dp) == 0
    for i in range(tp):
        ranks = range(i, world_size, tp)
        if rank in ranks:
            dp_group = dist.new_group(ranks)
    for i in range(world_size // tp):
        ranks = range(i * tp, (i + 1) * tp)
        if rank in ranks:
            tp_group = dist.new_group(ranks)

    tp_rank = dist.get_rank(group=tp_group)
    dp_rank = dist.get_rank(group=dp_group)
    print(f"{dist.get_rank()}, tp: {dist.get_rank(tp_group)}, dp: {dist.get_rank(dp_group)}")
    if zero > 1:
        assert dist.get_world_size(group=dp_group) % zero == 0
        zero_group = dp_group
    else:
        zero_group = None

    in_features, out_features = 4, 16

    linear = nn.Linear(in_features, out_features).cuda()
    linear_copy = copy.deepcopy(linear)
    if col:
        clip_dim = 0
        linear_col = Linear1D_Col.from_native_module(
            linear_copy, process_group=tp_group, gather_output=True, seq_parallel=seq_parallel, overlap=overlap
        )
    else:
        clip_dim = -1
        linear_col = Linear1D_Row.from_native_module(
            linear_copy, process_group=tp_group, seq_parallel=seq_parallel, parallel_input=True
        )
    api.get_device_mesh(linear_col.weight)
    api.get_sharding_spec(linear_col.weight)

    # ensure state dict is reversibly loadable
    linear.load_state_dict(linear_col.state_dict())
    linear_col.load_state_dict(linear.state_dict())

    with torch.no_grad():
        ori_dist_weight = copy.deepcopy(linear_col.weight)
        ori_target_weight = torch.chunk(linear.weight.clone(), tp, dim=clip_dim)[tp_rank]
        assert_close(ori_dist_weight, ori_target_weight)

    optim = CAME(linear.parameters(), lr=0.1)
    dist_optim = CAME(linear_col.parameters(), lr=0.1, tp_process_group=tp_group, zero_process_group=zero_group)

    if zero == 1:
        # check backward correctness
        linear.weight.grad = torch.randn(out_features, in_features).cuda()
        linear_col.weight.grad = torch.chunk(linear.weight.grad.clone(), tp, dim=clip_dim)[tp_rank].cuda()
        # out.sum().backward()
        # gather_out.sum().backward()
        with torch.no_grad():
            target_grad = torch.chunk(linear.weight.grad, tp, dim=clip_dim)[tp_rank]
            assert_close(target_grad, linear_col.weight.grad)
    else:
        optim = LowLevelZeroOptimizer(optimizer=optim, partition_grad=False, dp_process_group=dp_group)
        dist_optim = LowLevelZeroOptimizer(optimizer=dist_optim, partition_grad=False, dp_process_group=dp_group)
        # check computation correctness
        # [batch_size, seq_len, hidden_size]
        x = torch.rand(2, 4, in_features).cuda()
        x_for_unshard = x.expand_as(x.clone())
        x_for_unshard.requires_grad_(True)
        x_for_shard = (
            x.expand_as(x.clone()) if seq_parallel is False else torch.chunk(x.clone(), tp, dim=1 - clip_dim)[tp_rank]
        )
        x_for_shard.requires_grad_(True)

        out = linear(x_for_unshard)
        gather_out = linear_col(x_for_shard)
        assert_close(out, gather_out)

        out = out.sum()
        gather_out = gather_out.sum()

        optim.backward(out)
        dist_optim.backward(gather_out)

    with torch.no_grad():
        target_weight = torch.chunk(linear.weight.clone(), tp, dim=clip_dim)[tp_rank]
    assert_close(target_weight, linear_col.weight)

    # check optimizer correctness
    dist_optim.step()
    optim.step()
    dist_optim.zero_grad()
    optim.zero_grad()
    with torch.no_grad():
        target_weight = torch.chunk(linear.weight.clone(), tp, dim=clip_dim)[tp_rank]
    assert_close(target_weight, linear_col.weight)
    if zero == 1:
        # 比state_dict()
        ori_state_dict = optim.state_dict()["state"][0]
        dist_state_dict = dist_optim.state_dict()["state"][0]
        for ori_state, dist_state in zip(ori_state_dict.values(), dist_state_dict.values()):
            if type(ori_state) != int:
                if ori_state.dim() == 0 or ori_state.shape == dist_state.shape:
                    assert_close(ori_state, dist_state)
                else:
                    assert_close(torch.chunk(ori_state.clone(), tp, dim=clip_dim)[tp_rank], dist_state)
    # else:
    #     ori_state_dict = optim.optim.state_dict()["state"][0]
    #     dist_state_dict = dist_optim.optim.state_dict()["state"][0]
    #     print(dist_state_dict)
    #     for ori_state, dist_state in zip(ori_state_dict.values(), dist_state_dict.values()):
    #         if type(ori_state) != int:
    #             if ori_state.dim() == 0 or ori_state.shape == dist_state.shape:
    #                 assert_close(ori_state, dist_state)
    #             else:
    #                 assert_close(torch.chunk(ori_state.clone(), zero, dim=clip_dim)[zero_rank], dist_state)

    assert not torch.allclose(ori_target_weight, target_weight)
    assert not torch.allclose(ori_dist_weight, linear_col.weight)


@parameterize("lazy_init", [False])
@parameterize("seq_parallel", [False])
@parameterize("overlap", [True])
@parameterize("tp", [2])
@parameterize("zero", [2])  # zero parallel size
@parameterize("col", [True])
def run_dist_linear_test(lazy_init, seq_parallel, overlap, tp, zero, col):
    check_linear_1d_col(lazy_init, seq_parallel, overlap, tp, zero, col)
    # check_linear_1d_row(lazy_init, seq_parallel)
    # check_linear_col_plus_row(lazy_init, seq_parallel, overlap)


def check_dist_linear(rank, world_size, port=12256):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_dist_linear_test()


@rerun_if_address_is_in_use()
def test_linear():
    spawn(check_dist_linear, nprocs=4)


if __name__ == "__main__":
    test_linear()
