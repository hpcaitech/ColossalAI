import os
import tempfile
from contextlib import nullcontext
from typing import Tuple

import pytest
import torch
import torch.distributed as dist

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("fused MoE experts only exist in transformers v5", allow_module_level=True)

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_shardformer.test_model.test_moe_optimizer_coverage import NUM_TOKENS, _build


def _state_snapshot(optimizer):
    """Tensor optimizer states keyed by working-parameter name order (ZeRO keys state by master params)."""
    inner = optimizer.unwrap() if hasattr(optimizer, "unwrap") else optimizer.optim
    return [
        {k: v.detach().clone() for k, v in inner.state[p].items() if isinstance(v, torch.Tensor)}
        for group in inner.param_groups
        for p in group["params"]
        if p in inner.state
    ]


def run_optimizer_checkpoint(config: Tuple):
    """Sharded optimizer save/load must round-trip Adam state for fused EP/TP-sharded experts.

    The upstream MoE checkpoint test saves a momentum-free SGD, whose state is empty, so the expert
    optimizer-state gather/scatter path is never exercised; Adam keeps exp_avg/exp_avg_sq per param.
    """
    model_name, zero_stage, ep_size, tp_size = config
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=tp_size,
        ep_size=ep_size,
        zero_stage=zero_stage,
        precision="bf16",
        initial_scale=1,
        overlap_communication=False,
        find_unused_parameters=True,
    )
    booster = Booster(plugin=plugin)
    seed_all(1234)
    model = _build(model_name).to(torch.bfloat16).cuda()
    hidden_size = model.config.hidden_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, *_ = booster.boost(model, optimizer)

    # Ranks of one TP group must see the same inputs; otherwise their replicated (non TP-sharded) parameters get
    # different gradients and optimizer states, while a checkpoint keeps one copy per TP group.
    seed_all(4321 + dist.get_rank(plugin.dp_group))
    inputs = torch.rand(2, NUM_TOKENS // 2, hidden_size, device="cuda", dtype=torch.bfloat16)
    loss = model(inputs_embeds=inputs).last_hidden_state.float().pow(2).mean()
    booster.backward(loss, optimizer)
    optimizer.step()
    optimizer.zero_grad()

    expected = _state_snapshot(optimizer)
    assert expected, "optimizer has no tensor state; the round trip would be vacuous"

    context = tempfile.TemporaryDirectory() if rank == 0 else nullcontext()
    with context as tmp:
        shared = [tmp]
        dist.broadcast_object_list(shared, src=0)
        optim_dir = os.path.join(shared[0], "optim")
        booster.save_optimizer(optimizer, optim_dir, shard=True)
        dist.barrier()
        inner = optimizer.unwrap() if hasattr(optimizer, "unwrap") else optimizer.optim
        for state in inner.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    value.zero_()
        booster.load_optimizer(optimizer, optim_dir)
        dist.barrier()

    loaded = _state_snapshot(optimizer)
    assert len(loaded) == len(expected)
    for index, (left, right) in enumerate(zip(expected, loaded)):
        assert left.keys() == right.keys(), f"param {index}: state keys differ"
        for key in left:
            assert torch.equal(left[key], right[key].to(left[key].device)), f"param {index}: {key} not restored"


@parameterize(
    "config",
    [
        # (model, zero_stage, ep_size, tp_size) on 4 ranks
        ("mixtral", 1, 4, 1),
        ("mixtral", 1, 2, 2),
        ("deepseek_v3", 1, 4, 1),
        # ep_size 2 on 4 ranks gives moe_dp_size 2, so expert states are also sharded by Zero.
        ("mixtral", 1, 2, 1),
        ("deepseek_v3", 1, 2, 1),
        # expert sizes whose Zero padding differs between moe_dp_size and the global dp size
        ("mixtral_odd", 1, 2, 1),
    ],
)
def run_all(config: Tuple):
    run_optimizer_checkpoint(config)


def check_optimizer_checkpoint(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_all()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_moe_optimizer_checkpoint():
    spawn(check_optimizer_checkpoint, 4)


if __name__ == "__main__":
    test_moe_optimizer_checkpoint()
