"""Gemma shared grouped GEMM validation on allocated CUDA devices."""

import copy
from unittest.mock import patch

import pytest
import torch

from colossalai.shardformer.modeling.gemma4 import Gemma4GroupedExperts
from tests.kit.model_zoo.transformers.gemma4 import tiny_gemma4
from tests.kit.numerics.scenarios.independent_impl import check_fwd_bwd


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty_experts", [False, True])
def test_gemma4_grouped_experts(dtype, empty_experts):
    assert torch.cuda.is_available(), "requires an allocated CUDA device"
    torch.manual_seed(42)
    reference = tiny_gemma4("26b_a4b").model.layers[0].experts.cuda()
    candidate = Gemma4GroupedExperts.from_native_module(copy.deepcopy(reference))
    tokens = 19
    # Skewed routes, odd per-expert row counts and experts receiving no tokens.
    ids = torch.arange(tokens, device="cuda") % (2 if empty_experts else 8)
    indices = torch.stack((ids, (ids + 1) % (2 if empty_experts else 8)), -1)
    weights = torch.rand(tokens, 2, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    inputs = dict(
        hidden_states=torch.randn(tokens, 32, device="cuda"),
        top_k_index=indices,
        top_k_weights=weights,
    )
    report = check_fwd_bwd(candidate, reference, inputs, dtype, expect_calls=((torch, "_grouped_mm"),))
    assert report.passed, report


def test_gemma4_grouped_model_checkpointed():
    assert torch.cuda.is_available(), "requires an allocated CUDA device"
    from colossalai.shardformer.modeling.gemma4 import replace_gemma4_experts
    from tests.kit.model_zoo.transformers.gemma4 import gemma4_data

    model = replace_gemma4_experts(tiny_gemma4("26b_a4b")).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with patch.object(torch, "_grouped_mm", wraps=torch._grouped_mm) as grouped:
        for _ in range(3):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**{n: v.cuda() for n, v in gemma4_data().items()}).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        assert grouped.call_count >= 3 * 6 * 2 * 2  # two GEMMs, every layer, forward plus recomputation
    for pattern in (
        "router.proj",
        "router.scale",
        "router.per_expert_scale",
        "experts.gate_up_proj",
        "experts.down_proj",
        "mlp.",
    ):
        assert any(pattern in n and not torch.equal(p, before[n]) for n, p in model.named_parameters()), pattern
