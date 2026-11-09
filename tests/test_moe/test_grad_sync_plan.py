"""The gradient sync plan of the MoE plugin, on CPU with gloo (4 ranks: ep 2 x moe_dp 2).

Rank layout: ``ep_group`` = {0,1} / {2,3}, ``moe_dp_group`` = {0,2} / {1,3}, ``dp_group`` = all ranks.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from colossalai.moe import EPMoELayer, ReferenceExperts
from colossalai.tensor.moe_tensor.api import ExpertGradScale, set_moe_tensor_ep_group
from colossalai.tensor.moe_tensor.grad_sync import GradSyncSpec, build_grad_sync_plan, has_reduction_scaled_experts
from colossalai.testing import rerun_if_address_is_in_use, spawn

NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, LOCAL_TOKENS = 4, 2, 8, 4, 6
EP = 2


def _init_gloo(rank, world_size, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    # every rank creates every group, in the same order
    ep_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    moe_dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    return dist.group.WORLD, ep_groups[rank // EP], moe_dp_groups[rank % EP]


def _run_plan_structure(rank, world_size, port):
    dp_group, ep_group, moe_dp_group = _init_gloo(rank, world_size, port)
    try:
        dense = nn.Parameter(torch.zeros(2))
        # an expert of the existing EP models: only the ep group is tagged, the layer applies 1 / ep_size itself
        legacy_expert = nn.Parameter(torch.zeros(2))
        set_moe_tensor_ep_group(legacy_expert, ep_group)
        in_forward = EPMoELayer(NUM_EXPERTS, HIDDEN, INTERMEDIATE, ep_group=ep_group, experts=ReferenceExperts())
        in_reduction = EPMoELayer(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            ep_group=ep_group,
            experts=ReferenceExperts(),
            expert_grad_scale=ExpertGradScale.IN_REDUCTION,
        )

        plan = build_grad_sync_plan(
            [dense, legacy_expert, *in_forward.parameters(), *in_reduction.parameters()], dp_group, moe_dp_group
        )
        assert plan[dense] == GradSyncSpec(dp_group, 4)
        assert plan[legacy_expert] == GradSyncSpec(moe_dp_group, 2)
        for param in in_forward.parameters():
            assert plan[param] == GradSyncSpec(moe_dp_group, 2) and plan[param].extra_divisor == 1
        for param in in_reduction.parameters():
            # summed over the 2 replicas, divided by moe_dp_size * ep_size
            assert plan[param] == GradSyncSpec(moe_dp_group, 4) and plan[param].extra_divisor == 2

        assert not has_reduction_scaled_experts([dense, legacy_expert, *in_forward.parameters()])
        assert has_reduction_scaled_experts(in_reduction.parameters())
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_grad_sync_plan_structure():
    spawn(_run_plan_structure, 4)


def _route(tokens, router):
    weights, idx = torch.topk((tokens @ router.t()).softmax(-1), TOP_K, dim=-1)
    return idx, weights / weights.sum(-1, keepdim=True)


def _reference(tokens, router, gate_up, down):
    """SwiGLU experts, one expert at a time, on a single process."""
    idx, weights = _route(tokens, router)
    out = torch.zeros_like(tokens)
    for expert in range(NUM_EXPERTS):
        token, slot = torch.where(idx == expert)
        gate, up = F.linear(tokens[token], gate_up[expert]).chunk(2, dim=-1)
        out = out.index_add(0, token, F.linear(F.silu(gate) * up, down[expert]) * weights[token, slot, None])
    return out


def _run_plan_gradient(rank, world_size, port, grad_scale):
    """Reducing by the plan (sum over the replica group, then divide) gives the gradient of the mean-over-ranks loss."""
    dp_group, ep_group, moe_dp_group = _init_gloo(rank, world_size, port)
    try:
        generator = torch.Generator().manual_seed(0)  # same weights and data on every rank
        gate_up = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        down = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
        router = torch.randn(NUM_EXPERTS, HIDDEN, generator=generator)
        x = torch.randn(world_size * LOCAL_TOKENS, HIDDEN, generator=generator)
        upstream = torch.randn(world_size * LOCAL_TOKENS, HIDDEN, generator=generator)

        # reference: all tokens on one process, loss = mean over the ranks' sums
        full = {k: v.clone().requires_grad_() for k, v in {"gate_up": gate_up, "down": down, "router": router}.items()}
        (_reference(x, full["router"], full["gate_up"], full["down"]) * upstream).sum().backward()

        layer = EPMoELayer(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            ep_group=ep_group,
            experts=ReferenceExperts(),
            expert_grad_scale=grad_scale,
        )
        layer.load_full_weights(gate_up, down)
        router_param = nn.Parameter(router.clone())
        mine = slice(rank * LOCAL_TOKENS, (rank + 1) * LOCAL_TOKENS)
        idx, weights = _route(x[mine], router_param)
        (layer(x[mine], idx, weights) * upstream[mine]).sum().backward()

        plan = build_grad_sync_plan([router_param, *layer.parameters()], dp_group, moe_dp_group)
        local = layer.local_expert_range
        for name, param, expected in (
            ("router", router_param, full["router"].grad),
            ("gate_up_proj", layer.gate_up_proj, full["gate_up"].grad[local.start : local.stop]),
            ("down_proj", layer.down_proj, full["down"].grad[local.start : local.stop]),
        ):
            spec = plan[param]
            grad = param.grad.clone()
            dist.all_reduce(grad, group=spec.replica_group)
            grad /= spec.sum_divisor
            assert torch.allclose(grad, expected / world_size, rtol=1e-4, atol=1e-6), f"{grad_scale.value} {name}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("grad_scale", [ExpertGradScale.IN_FORWARD, ExpertGradScale.IN_REDUCTION])
@rerun_if_address_is_in_use()
def test_grad_sync_plan_gives_exact_gradient(grad_scale):
    spawn(_run_plan_gradient, 4, grad_scale=grad_scale)
