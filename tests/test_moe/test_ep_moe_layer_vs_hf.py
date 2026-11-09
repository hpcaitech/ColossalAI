"""EPMoELayer against the native HF Qwen3-MoE sparse block (CPU only).

The layer is a drop-in for ``Qwen3MoeSparseMoeBlock.experts``: HF calls ``experts(hidden_states, top_k_index,
top_k_weights)`` and the layer takes the same three tensors, with the same parameter names. The router stays HF's.
Everything is judged with the baseline-error rule of ``tests.kit.numerics``: float64 HF is the truth, HF at the
tested precision is the reference whose error sets the allowance, and the swapped model is the test.
"""

import copy
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from colossalai._compat import is_transformers_v5
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import Trace, assert_aligned, run_fwd_bwd

pytestmark = pytest.mark.skipif(not is_transformers_v5(), reason="needs the transformers v5 fused-experts layout")

# the shapes can be overridden to approach a real model (Qwen3-30B-A3B: 128 experts, top-8), e.g.
# MOE_TEST_NUM_EXPERTS=64 MOE_TEST_TOP_K=8; spawned workers inherit the environment
NUM_EXPERTS = int(os.environ.get("MOE_TEST_NUM_EXPERTS", 8))
TOP_K = int(os.environ.get("MOE_TEST_TOP_K", 2))
HIDDEN, INTER = 32, 16
BACKENDS = [
    pytest.param(
        "grouped_mm",
        marks=pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="needs torch._grouped_mm (torch >= 2.8)"),
    ),
    "reference",
]


def _tiny_model(seed: int = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=INTER,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        mlp_only_layers=[],
        max_position_embeddings=64,
        use_cache=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        experts_implementation="eager",
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config).float()
    # HF initialises the router to a tiny scale and the experts to 0.02: use sizeable values so that routing is
    # not near-uniform and the experts matter
    for name, param in model.named_parameters():
        if name.endswith("mlp.gate.weight"):
            param.data.normal_()
        elif name.endswith("experts.gate_up_proj"):
            param.data.normal_(std=HIDDEN**-0.5)
        elif name.endswith("experts.down_proj"):
            param.data.normal_(std=INTER**-0.5)
    return model


def _swap_experts(module, backend, ep_group=None):
    """Replace every HF experts module below ``module`` by an EPMoELayer holding the same weights."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    from colossalai.moe import EPMoELayer, GroupedMMExperts, ReferenceExperts

    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Qwen3MoeExperts):
                layer = EPMoELayer(
                    child.num_experts,
                    child.hidden_dim,
                    child.intermediate_dim,
                    ep_group=ep_group,
                    act_fn=F.silu,
                    experts=GroupedMMExperts() if backend == "grouped_mm" else ReferenceExperts(),
                )
                layer.load_full_weights(child.gate_up_proj.data, child.down_proj.data)
                setattr(parent, name, layer)


def _block_inputs(batch: int = 4):
    generator = torch.Generator().manual_seed(1)
    return {"hidden_states": torch.randn(batch, 16, HIDDEN, generator=generator, dtype=torch.float64)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", BACKENDS)
def test_block_matches_hf(dtype, backend):
    """Layer level: the router sees the same input in all three runs, so there are no routing flips to explain."""
    block = _tiny_model().model.layers[0].mlp
    new = copy.deepcopy(block)
    _swap_experts(new, backend)
    inputs = _block_inputs()
    truth = run_fwd_bwd(block, inputs, torch.float64)
    ref = run_fwd_bwd(block, inputs, dtype)
    if backend == "grouped_mm":
        with patch.object(torch, "_grouped_mm", wraps=torch._grouped_mm) as grouped_mm:
            test = run_fwd_bwd(new, inputs, dtype)
        assert grouped_mm.call_count > 0, "the test never executed torch._grouped_mm"
    else:
        test = run_fwd_bwd(new, inputs, dtype)
    assert_aligned(test, ref, truth)


def _init_gloo(rank, world_size, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)


def _gather(tensor, group=None):
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, 0)


def _run_ep_block(rank, world_size, port, backend):
    _init_gloo(rank, world_size, port)
    try:
        ep_group = dist.new_group(list(range(world_size)))
        block = _tiny_model().model.layers[0].mlp
        local = 2
        inputs = _block_inputs(batch=world_size * local)
        truth = run_fwd_bwd(block, inputs, torch.float64)
        ref = run_fwd_bwd(block, inputs, torch.float32)

        new = copy.deepcopy(block)
        _swap_experts(new, backend, ep_group)
        x = inputs["hidden_states"]
        mine = x[rank * local : (rank + 1) * local].float().requires_grad_()
        out = new(mine)
        # the same upstream gradient as run_fwd_bwd draws for the full output
        upstream = torch.randn(x.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        (out * upstream[rank * local : (rank + 1) * local].float()).sum().backward()

        router_grad = new.gate.weight.grad.clone()
        dist.all_reduce(router_grad)  # the router is replicated: its gradient is summed over the ranks' tokens
        # the layer divides expert gradients by ep_size in backward (ExpertGradScale.IN_FORWARD), so undo that to
        # compare with the full-batch gradient
        test = Trace(
            {
                "out": _gather(out.detach()),
                "grad/hidden_states": _gather(mine.grad),
                "grad/param/experts.gate_up_proj": _gather(new.experts.gate_up_proj.grad) * world_size,
                "grad/param/experts.down_proj": _gather(new.experts.down_proj.grad) * world_size,
                "grad/param/gate.weight": router_grad,
            }
        )
        assert_aligned(test, ref, truth)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ep_size", [2, 4])
@rerun_if_address_is_in_use()
def test_ep_block_matches_hf(backend, ep_size):
    spawn(_run_ep_block, ep_size, backend=backend)


def _run_dp_ep_grad_scale(rank, world_size, port):
    """ep 2 x moe_dp 2, reduced the way the current ZeRO code does: dense gradients are averaged over all dp ranks,
    expert gradients only over the moe_dp group (zero/low_level/low_level_optim.py divides by that group's size)."""
    _init_gloo(rank, world_size, port)
    try:
        ep = 2
        ep_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        moe_dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        ep_group, moe_dp_group = ep_groups[rank // ep], moe_dp_groups[rank % ep]

        block = _tiny_model().model.layers[0].mlp
        local = 4
        x = _block_inputs(batch=world_size * local)["hidden_states"].float()
        upstream = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))

        exact = copy.deepcopy(block)
        (exact(x.clone()) * upstream).sum().backward()
        target = {
            "gate_up": exact.experts.gate_up_proj.grad / world_size,
            "down": exact.experts.down_proj.grad / world_size,
            "router": exact.gate.weight.grad / world_size,
        }

        new = copy.deepcopy(block)
        _swap_experts(new, "reference", ep_group)
        mine = x[rank * local : (rank + 1) * local]
        (new(mine) * upstream[rank * local : (rank + 1) * local]).sum().backward()

        got = {}
        router = new.gate.weight.grad.clone()
        dist.all_reduce(router)
        got["router"] = router / world_size
        for key, param in (("gate_up", new.experts.gate_up_proj), ("down", new.experts.down_proj)):
            grad = param.grad.clone()
            dist.all_reduce(grad, group=moe_dp_group)
            grad /= dist.get_world_size(moe_dp_group)
            got[key] = _gather(grad, ep_group)
        for key in target:
            rel = ((got[key] - target[key]).norm() / target[key].norm()).item()
            assert rel < 1e-4, f"{key}: relative error {rel:.2e} against the full-batch gradient"
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_dp_ep_expert_gradient_scale():
    """With the default ExpertGradScale.IN_FORWARD the current reduction rule gives the exact gradient."""
    spawn(_run_dp_ep_grad_scale, 4)
