import copy
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
import colossalai.tensor.d_tensor.api as api
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from colossalai.nn.optimizer.came import CAME
from colossalai.nn.optimizer.distributed_came import DistributedCAME
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor import api
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


def init_distribute():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(world_size=world_size, rank=rank, init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)


def check_dist_1d(seq_parallel, tp_size, zero_size, col, zero_stage):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    in_features, out_features = 128, 512
    assert world_size % tp_size == 0, "world size must divide tp size"
    DP_AXIS = 0
    TP_AXIS = 1
    zero_size = world_size // tp_size if zero_size == 0 else zero_size
    pg_mesh = ProcessGroupMesh(zero_size, tp_size)
    tp_group = pg_mesh.get_group_along_axis(TP_AXIS)
    dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
    # print("TP: ", dist.get_process_group_ranks(tp_group), "DP: ", dist.get_process_group_ranks(dp_group))
    ori_model = nn.Linear(in_features, out_features).to(rank)
    if col:
        clip_dim = 0
        shard_model = Linear1D_Col.from_native_module(
            copy.deepcopy(ori_model),
            process_group=tp_group,
            gather_output=True,
            seq_parallel=seq_parallel,
            overlap=False,
        ).to(rank)
    else:
        clip_dim = -1
        shard_model = Linear1D_Row.from_native_module(
            copy.deepcopy(ori_model), process_group=tp_group, seq_parallel=seq_parallel, parallel_input=False
        ).to(rank)
    tp_rank = dist.get_rank(group=tp_group)

    # check original weight and bias
    with torch.no_grad():
        ori_dist_weight = copy.deepcopy(shard_model.weight)
        ori_target_weight = torch.chunk(ori_model.weight.clone(), tp_size, dim=clip_dim)[tp_rank]
        ori_dist_bias = copy.deepcopy(shard_model.bias)
        ori_target_bias = torch.chunk(ori_model.bias.clone(), tp_size if col else 1, dim=clip_dim)[
            tp_rank if col else 0
        ]
        assert_close(ori_dist_weight, ori_target_weight)
        assert_close(ori_dist_bias, ori_target_bias)

    # plugin = HybridParallelPlugin(tp_size=tp_size, pp_size=1, zero_stage=zero_stage)
    # booster = Booster(plugin=plugin)

    optim = CAME(ori_model.parameters(), lr=0.1)
    dist_optim = DistributedCAME(shard_model.parameters(), lr=0.1)

    # shard_model, dist_optim, _, _, _ = booster.boost(shard_model, dist_optim)

    if zero_size > 1:
        dist_optim = LowLevelZeroOptimizer(
            optimizer=dist_optim, partition_grad=(zero_stage == 2), dp_process_group=dp_group
        )
    master_to_working_map = (
        dist_optim.get_master_to_working_map() if isinstance(dist_optim, LowLevelZeroOptimizer) else None
    )
    True if isinstance(dist_optim, LowLevelZeroOptimizer) else False
    if isinstance(dist_optim, LowLevelZeroOptimizer):
        dist_optim.optim.setup_distributed(master_to_working_map, tp_group, dp_group)
    else:
        dist_optim.setup_distributed(master_to_working_map, tp_group, dp_group)

    ori_model.weight.grad = torch.randn(out_features, in_features).cuda()
    ori_model.bias.grad = torch.randn(out_features).cuda()
    shard_model.weight.grad = torch.chunk(ori_model.weight.grad.clone(), tp_size, dim=clip_dim)[tp_rank].cuda()
    shard_model.bias.grad = torch.chunk(ori_model.bias.grad.clone(), tp_size if col else 1, dim=0)[
        tp_rank if col else 0
    ].cuda()

    # check grad assign
    with torch.no_grad():
        target_grad = torch.chunk(ori_model.weight.grad.clone(), tp_size, dim=clip_dim)[tp_rank]
        target_bias_grad = torch.chunk(ori_model.bias.grad.clone(), tp_size if col else 1, dim=0)[tp_rank if col else 0]
        assert_close(target_grad, shard_model.weight.grad)
        assert_close(target_bias_grad, shard_model.bias.grad)

    if zero_size > 1:
        x = torch.rand(2, 4, in_features).cuda()
        x_for_unshard = x.expand_as(x.clone())
        x_for_unshard.requires_grad_(True)
        x_for_shard = (
            x.expand_as(x.clone())
            if seq_parallel is False
            else torch.chunk(x.clone(), tp_size, dim=1 - clip_dim)[tp_rank]
        )
        x_for_shard.requires_grad_(True)

        out = ori_model(x_for_unshard)
        gather_out = shard_model(x_for_shard)
        assert_close(out, gather_out)

        out = out.sum()
        gather_out = gather_out.sum()
        out.backward()
        dist_optim.backward(gather_out)

    # check before optim.step()
    with torch.no_grad():
        target_weight = torch.chunk(ori_model.weight.clone(), tp_size, dim=clip_dim)[tp_rank]
        target_bias = torch.chunk(ori_model.bias.clone(), tp_size if col else 1, dim=0)[tp_rank if col else 0]
        assert_close(target_weight, shard_model.weight)
        assert_close(target_bias, shard_model.bias)

    optim.step()
    dist_optim.step()
    dist_optim.zero_grad()
    optim.zero_grad()
    with torch.no_grad():
        target_weight = torch.chunk(ori_model.weight.clone(), tp_size, dim=clip_dim)[tp_rank]
        target_bias = torch.chunk(ori_model.bias.clone(), tp_size if col else 1, dim=0)[tp_rank if col else 0]
    # check after optim.step()
    assert not torch.allclose(ori_target_weight, target_weight)
    assert not torch.allclose(ori_dist_weight, shard_model.weight)
    assert_close(target_weight, shard_model.weight)
    assert_close(target_bias, shard_model.bias)


@parameterize("seq_parallel", [False])
@parameterize("tp_size", [4])
@parameterize("zero_size", [0])  # zero parallel size, 0 means world_size // tp_size
@parameterize("col", [True, False])
@parameterize("zero_stage", [2])
def run_dist_linear_test(seq_parallel, tp_size, zero_size, col, zero_stage):
    check_dist_1d(seq_parallel, tp_size, zero_size, col, zero_stage)
    api.clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_dist_linear(rank, world_size, port=12256):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_dist_linear_test()


@rerun_if_address_is_in_use()
def test_dist_came():
    spawn(check_dist_linear, nprocs=4)


if __name__ == "__main__":
    test_dist_came()
