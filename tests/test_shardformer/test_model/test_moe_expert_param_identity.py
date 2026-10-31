import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("fused MoE experts only exist in transformers v5", allow_module_level=True)

from transformers import DeepseekV3Config, MixtralConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from colossalai.shardformer.modeling.deepseek_v3 import EpDeepseekV3MoE
from colossalai.shardformer.modeling.mixtral import EPMixtralSparseMoeBlock
from colossalai.testing import rerun_if_address_is_in_use, spawn


@pytest.fixture
def single_rank_group():
    # Optimizers are created from the unsharded model and the MoE plugin only prunes optimizer params
    # when PP/EP > 1, so the EP/TP expert slicing must keep the original Parameter objects.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def _expert_params(module):
    return module.experts.gate_up_proj, module.experts.down_proj


def test_mixtral_ep_keeps_expert_parameter_objects(single_rank_group):
    config = MixtralConfig(hidden_size=32, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2)
    module = MixtralSparseMoeBlock(config)
    before = _expert_params(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    module = EPMixtralSparseMoeBlock.from_native_module(
        module, tp_group=single_rank_group, moe_dp_group=single_rank_group, ep_group=single_rank_group
    )

    after = _expert_params(module)
    assert all(a is b for a, b in zip(after, before))
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimizer_params for p in after)


def _check_mixtral_tp_keeps_expert_parameter_objects(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        own_group = [dist.new_group(ranks=[r]) for r in range(world_size)][rank]
        torch.manual_seed(0)
        config = MixtralConfig(hidden_size=32, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2)
        module = MixtralSparseMoeBlock(config)
        # v5 allocates fused experts with torch.empty (weights are initialised by the full model), so
        # fill them with known values before comparing the local shards.
        with torch.no_grad():
            for param in _expert_params(module):
                param.copy_(torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape))
        full_gate_up = module.experts.gate_up_proj.detach().clone()
        full_down = module.experts.down_proj.detach().clone()
        before = _expert_params(module)
        optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

        # TP over the expert intermediate dimension; EP = 1 on each rank.
        module = EPMixtralSparseMoeBlock.from_native_module(
            module, tp_group=dist.group.WORLD, moe_dp_group=own_group, ep_group=own_group
        )

        after = _expert_params(module)
        assert all(a is b for a, b in zip(after, before))
        optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in optimizer_params for p in after)
        # The in-place storage swap must still hold this rank's gate/up/down shards.
        part = 64 // world_size
        expected_gate_up = torch.cat(
            (
                full_gate_up[:, rank * part : (rank + 1) * part],
                full_gate_up[:, 64 + rank * part : 64 + (rank + 1) * part],
            ),
            dim=1,
        )
        assert torch.equal(module.experts.gate_up_proj.detach(), expected_gate_up)
        assert torch.equal(module.experts.down_proj.detach(), full_down[:, :, rank * part : (rank + 1) * part])
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_mixtral_tp_keeps_expert_parameter_objects():
    spawn(_check_mixtral_tp_keeps_expert_parameter_objects, nprocs=2)


def test_deepseek_v3_ep_keeps_expert_parameter_objects(single_rank_group):
    config = DeepseekV3Config(
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )
    module = DeepseekV3MoE(config)
    before = _expert_params(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=1.0)

    module = EpDeepseekV3MoE.from_native_module(module, moe_dp_group=single_rank_group, ep_group=single_rank_group)

    after = _expert_params(module)
    assert all(a is b for a, b in zip(after, before))
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimizer_params for p in after)
