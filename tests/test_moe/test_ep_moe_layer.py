import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import colossalai
from colossalai.moe import EPMoELayer, GroupedMMExperts, NCCLAllToAllDispatcher, ReferenceExperts
from colossalai.tensor.moe_tensor.api import ExpertGradScale, get_ep_group, get_expert_grad_scale, is_moe_tensor
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

# Relative L2 error allowed between the bf16 layer and an fp32 single-GPU reference. The largest error over all checks
# here is 6e-3; dropping the EP grad scaler, for example, gives 0.9.
RTOL = 2e-2

NUM_EXPERTS, TOP_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_TOKENS = 8, 2, 64, 32, 64


def ep_groups(ep_size: int):
    """Split the world into consecutive groups of ep_size ranks; return this rank's group."""
    world, rank = dist.get_world_size(), dist.get_rank()
    mine = None
    for start in range(0, world, ep_size):
        group = dist.new_group(list(range(start, start + ep_size)))
        if start <= rank < start + ep_size:
            mine = group
    return mine


def route(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor):
    """Softmax top-k router with renormalised fp32 weights (Mixtral / Qwen style)."""
    probs = (F.linear(x.float(), gate) + bias).softmax(-1)
    weights, idx = torch.topk(probs, TOP_K, dim=-1)
    return idx, weights / weights.sum(-1, keepdim=True)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@parameterize("ep_size", [1, 2, 4])
def run_dispatch_contract(ep_size: int):
    """Rows reach the right experts, grouped by expert and in global token order; combine undoes dispatch."""
    group = ep_groups(ep_size)
    ep_rank = dist.get_rank(group)
    dispatcher = NCCLAllToAllDispatcher(NUM_EXPERTS, group)
    num_local = NUM_EXPERTS // ep_size

    # ep rank 1 sends nothing, and expert 1 is never chosen
    num_tokens = 0 if ep_rank == 1 else NUM_TOKENS
    generator = torch.Generator(device="cuda").manual_seed(dist.get_rank())
    choices = torch.tensor([e for e in range(NUM_EXPERTS) if e != 1], device="cuda")
    scores = torch.rand(num_tokens, choices.numel(), device="cuda", generator=generator)
    topk_idx = choices[scores.argsort(-1)[:, :TOP_K]]
    topk_weights = torch.rand(num_tokens, TOP_K, device="cuda", generator=generator)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    x = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", generator=generator)
    x[:, 0] = torch.arange(num_tokens, device="cuda") + ep_rank * NUM_TOKENS  # global token id within the ep group

    dispatched = dispatcher.dispatch(x, topk_idx, topk_weights)
    counts = dispatched.tokens_per_expert.tolist()
    assert dispatched.tokens_per_expert_cpu == counts
    assert dispatched.tokens.size(0) == sum(counts)

    # expected rows of each local expert: ids of the tokens that chose it, in ep-group token order
    gathered = [None] * ep_size
    dist.all_gather_object(gathered, (x[:, 0].long().tolist(), topk_idx.tolist()), group=group)
    for local, rows in enumerate(dispatched.tokens.split(counts)):
        expert = ep_rank * num_local + local
        expected = [tid for ids, idx in gathered for tid, chosen in zip(ids, idx) if expert in chosen]
        assert rows[:, 0].long().tolist() == expected, f"expert {expert}: rows out of place"

    # identity experts: combine must give back sum_j w_j * x = x
    y = dispatcher.combine(dispatched.tokens, dispatched)
    assert y.shape == x.shape
    torch.testing.assert_close(y, x, rtol=1e-6, atol=1e-6)


@parameterize("ep_size", [1, 2, 4])
@parameterize("backend", ["grouped_mm", "reference"])
@parameterize("grad_scale", [ExpertGradScale.IN_FORWARD, ExpertGradScale.IN_REDUCTION])
@parameterize("skewed", [False, True])
def run_layer_vs_reference(ep_size: int, backend: str, grad_scale: ExpertGradScale, skewed: bool):
    """Output, input grad and expert weight grads against an fp32 single-GPU computation."""
    group = ep_groups(ep_size)
    experts = GroupedMMExperts() if backend == "grouped_mm" else ReferenceExperts()
    layer = EPMoELayer(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        ep_group=group,
        experts=experts,
        expert_grad_scale=grad_scale,
        device="cuda",
        dtype=torch.bfloat16,
    )

    generator = torch.Generator(device="cuda").manual_seed(0)  # same weights on every rank
    gate = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, device="cuda", generator=generator) * HIDDEN_SIZE**-0.5
    full_gate_up = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device="cuda", generator=generator) * (
        HIDDEN_SIZE**-0.5
    )
    full_down = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device="cuda", generator=generator) * (
        INTERMEDIATE_SIZE**-0.5
    )
    # skewed: the second half of the experts receives no token at all
    bias = torch.zeros(NUM_EXPERTS, device="cuda")
    if skewed:
        bias[NUM_EXPERTS // 2 :] = -1e4
    layer.load_full_weights(full_gate_up, full_down)

    generator = torch.Generator(device="cuda").manual_seed(1 + dist.get_rank())
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device="cuda", generator=generator)
    upstream = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device="cuda", generator=generator)

    # layer (bf16), router in fp32 as models do
    x_layer = x.to(torch.bfloat16).requires_grad_()
    topk_idx, topk_weights = route(x_layer, gate, bias)
    y = layer(x_layer, topk_idx, topk_weights)
    (y.float() * upstream).sum().backward()

    # fp32 reference over this rank's tokens; expert grads summed over the ep group, as the layer computes them
    x_ref = x.to(torch.bfloat16).float().requires_grad_()
    gate_up_ref = full_gate_up.to(torch.bfloat16).float().requires_grad_()
    down_ref = full_down.to(torch.bfloat16).float().requires_grad_()
    ref_idx, ref_weights = route(x_ref, gate, bias)
    y_ref = torch.zeros_like(x_ref)
    for expert in range(NUM_EXPERTS):
        token, slot = torch.where(ref_idx == expert)
        g, u = F.linear(x_ref[token], gate_up_ref[expert]).chunk(2, dim=-1)
        out = F.linear(F.silu(g) * u, down_ref[expert]) * ref_weights[token, slot, None]
        y_ref = y_ref.index_add(0, token, out)
    (y_ref * upstream).sum().backward()
    dist.all_reduce(gate_up_ref.grad, group=group)
    dist.all_reduce(down_ref.grad, group=group)
    local = slice(layer.local_expert_range.start, layer.local_expert_range.stop)

    # IN_FORWARD divides expert grads by ep_size inside the layer; IN_REDUCTION leaves that to the reduction
    factor = ep_size if grad_scale == ExpertGradScale.IN_FORWARD else 1
    config = f"ep={ep_size} {backend} {grad_scale.value} skewed={skewed}"
    assert rel(y, y_ref) < RTOL, f"{config}: output {rel(y, y_ref):.3e}"
    assert rel(x_layer.grad, x_ref.grad) < RTOL, f"{config}: input grad {rel(x_layer.grad, x_ref.grad):.3e}"
    for name, grad, ref in [
        ("gate_up_proj", layer.gate_up_proj.grad, gate_up_ref.grad[local]),
        ("down_proj", layer.down_proj.grad, down_ref.grad[local]),
    ]:
        assert grad is not None, f"{config}: {name} has no grad"
        for expert, (g, r) in enumerate(zip(grad, ref)):
            if r.abs().max() == 0:  # expert received no token anywhere in the ep group
                assert g.abs().max() == 0, f"{config}: {name}[{expert}] should be zero"
            else:
                err = rel(g.float() * factor, r)
                assert err < RTOL, f"{config}: {name}[{expert}] grad {err:.3e}"


def run_param_contract():
    """Expert params carry their EP tags, and loading weights keeps the parameter objects."""
    group = ep_groups(2)
    layer = EPMoELayer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, ep_group=group, device="cuda")
    assert layer.num_local_experts == NUM_EXPERTS // 2
    assert layer.local_expert_range == range(dist.get_rank(group) * 4, dist.get_rank(group) * 4 + 4)
    params = (layer.gate_up_proj, layer.down_proj)
    for param in params:
        assert is_moe_tensor(param) and get_ep_group(param) is group
        assert get_expert_grad_scale(param) == ExpertGradScale.IN_FORWARD
    ids = [id(p) for p in params]
    layer.load_full_weights(
        torch.ones(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device="cuda"),
        torch.ones(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device="cuda"),
    )
    assert [id(p) for p in (layer.gate_up_proj, layer.down_proj)] == ids
    assert get_expert_grad_scale(torch.zeros(1)) is None


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.cuda.set_device(rank)
    run_dispatch_contract()
    run_layer_vs_reference()
    run_param_contract()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_ep_moe_layer(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_ep_moe_layer(world_size=4)
