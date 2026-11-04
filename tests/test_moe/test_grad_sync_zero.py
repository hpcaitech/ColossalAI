"""Exercise the gradient sync plan through the real MoE plugin and ZeRO optimizer on CUDA/NCCL."""

import faulthandler
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import colossalai
from colossalai.accelerator import set_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import MoeHybridParallelPlugin
from colossalai.moe import EPMoELayer
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.tensor.moe_tensor.api import ExpertGradScale
from colossalai.testing import rerun_if_address_is_in_use, spawn

WORLD_SIZE = 4
EP_SIZE = 2
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN = 8
INTERMEDIATE = 8
LOCAL_TOKENS = 6
GRAD_RTOL = 5e-2


class NoOpPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self):
        return {}

    def postprocess(self):
        return self.model


class TinyMoeModel(nn.Module):
    def __init__(self, ep_group, grad_scale, empty_experts=False):
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.router_weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, generator=generator))
        self.router_bias = nn.Parameter(torch.zeros(NUM_EXPERTS))
        self.experts = EPMoELayer(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            ep_group=ep_group,
            expert_grad_scale=grad_scale,
        )
        gate_up = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        down = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
        self.experts.load_full_weights(gate_up, down)
        self.top_k = 1 if empty_experts else TOP_K
        if empty_experts:
            with torch.no_grad():
                self.router_weight.zero_()
                self.router_bias.copy_(torch.tensor([8.0, -8.0, -8.0, -8.0]))

    def forward(self, tokens):
        logits = F.linear(tokens, self.router_weight, self.router_bias)
        weights, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return self.experts(tokens, idx, weights)


def _reference(tokens, router_weight, router_bias, gate_up, down, top_k):
    logits = F.linear(tokens, router_weight, router_bias)
    weights, idx = torch.topk(logits.softmax(-1), top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = torch.zeros_like(tokens)
    for expert in range(NUM_EXPERTS):
        token, slot = torch.where(idx == expert)
        gate, up = F.linear(tokens[token], gate_up[expert]).chunk(2, dim=-1)
        out = out.index_add(0, token, F.linear(F.silu(gate) * up, down[expert]) * weights[token, slot, None])
    return out


def _full_reference(model, inputs, upstreams):
    params = {name: param.detach().clone().float().requires_grad_() for name, param in model.named_parameters()}
    for name in ("experts.gate_up_proj", "experts.down_proj"):
        local = params[name]
        shards = [torch.empty_like(local) for _ in range(EP_SIZE)]
        dist.all_gather(shards, local, group=model.experts.ep_group)
        params[name] = torch.cat(shards).detach().requires_grad_()
    for tokens, upstream in zip(inputs, upstreams):
        device = params["router_weight"].device
        tokens, upstream = tokens.to(device), upstream.to(device)
        out = _reference(
            tokens,
            params["router_weight"],
            params["router_bias"],
            params["experts.gate_up_proj"],
            params["experts.down_proj"],
            model.top_k,
        )
        (out * upstream).sum().backward()
    return params


def _run_case(stage, grad_scale, overlap, empty_experts):
    rank = dist.get_rank()
    plugin = MoeHybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        ep_size=EP_SIZE,
        zero_stage=stage,
        precision="bf16",
        overlap_communication=overlap,
        custom_policy=NoOpPolicy(),
    )
    if stage == 2 or overlap:
        assert plugin.zero_stage == 1
        assert plugin.zero_config["overlap_communication"] is False
    model = TinyMoeModel(plugin.ep_group, grad_scale, empty_experts=empty_experts)
    booster = Booster(plugin=plugin)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    generator = torch.Generator().manual_seed(23)
    inputs = [torch.randn(WORLD_SIZE * LOCAL_TOKENS, HIDDEN, generator=generator) for _ in range(2)]
    upstreams = [torch.randn(WORLD_SIZE * LOCAL_TOKENS, HIDDEN, generator=generator) for _ in range(2)]
    microbatches = 1 if stage == 2 else 2
    reference = _full_reference(model.unwrap(), inputs[:microbatches], upstreams[:microbatches])

    for tokens, upstream in zip(inputs[:microbatches], upstreams[:microbatches]):
        mine = slice(rank * LOCAL_TOKENS, (rank + 1) * LOCAL_TOKENS)
        sync_context = optimizer.no_sync() if microbatches == 2 and tokens is inputs[0] else nullcontext()
        with sync_context:
            out = model(tokens[mine].cuda())
            optimizer.backward((out * upstream[mine].cuda()).sum())

    errors = []
    local = model.unwrap().experts.local_expert_range
    for name, param in model.unwrap().named_parameters():
        got = optimizer.get_param_grad(param)
        expected = reference[name].grad.detach().cpu() / WORLD_SIZE
        if name.startswith("experts."):
            expected = expected[local.start : local.stop]
        got = torch.zeros_like(expected) if got is None else got.detach().cpu().float()
        error = (got - expected).norm()
        allowance = GRAD_RTOL * expected.norm() + 1e-6
        if error > allowance:
            errors.append(
                f"{name}: relative L2={error.item() / max(expected.norm().item(), 1e-12):.4f}, "
                f"got/reference norm={got.norm().item() / max(expected.norm().item(), 1e-12):.4f}"
            )
    assert (
        not errors
    ), f"stage={stage}, scale={grad_scale.value}, overlap={overlap}, empty={empty_experts}: " + "; ".join(errors)


CASES = [
    (1, ExpertGradScale.IN_FORWARD, False, False),
    (1, ExpertGradScale.IN_REDUCTION, False, False),
    (1, ExpertGradScale.IN_REDUCTION, True, False),
    (2, ExpertGradScale.IN_REDUCTION, False, False),
    (1, ExpertGradScale.IN_REDUCTION, False, True),
]


def _run_dist(rank, world_size, port, case):
    set_accelerator("cuda")
    torch.cuda.set_device(rank)
    torch.empty(1, device=f"cuda:{rank}")
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        colossalai.launch(
            rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl", local_rank=rank
        )
        print(f"START rank={rank} case={case}", flush=True)
        _run_case(*case)
        print(f"END rank={rank} case={case}", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.dist
@pytest.mark.parametrize("case", CASES)
@rerun_if_address_is_in_use()
def test_grad_sync_zero(case):
    spawn(_run_dist, WORLD_SIZE, case=case)


if __name__ == "__main__":
    for case in CASES:
        test_grad_sync_zero(case=case)
