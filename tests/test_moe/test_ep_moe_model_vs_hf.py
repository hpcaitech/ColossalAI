"""Whole-model Qwen3-MoE checks kept outside the CPU CI layer.

The layer, EP and DP-contract tests live in ``test_ep_moe_layer_vs_hf.py`` and are collected by the CPU batch. These
checks exercise the complete HF model and remain available for local L2 reports without making the CPU CI job a model
level acceptance gate.
"""

import copy

import pytest
import torch

from colossalai._compat import is_transformers_v5
from tests.kit.numerics import assert_aligned, run_fwd_bwd
from tests.test_moe.test_ep_moe_layer_vs_hf import BACKENDS, _swap_experts, _tiny_model

pytestmark = pytest.mark.skipif(not is_transformers_v5(), reason="needs the transformers v5 fused-experts layout")


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_matches_hf_fp32(backend):
    """Whole model in fp32: logits, loss, aux loss, router logits and every parameter gradient."""
    model = _tiny_model()
    new = copy.deepcopy(model)
    _swap_experts(new, backend)
    ids = torch.randint(1, 64, (2, 16), generator=torch.Generator().manual_seed(2))
    inputs = {"input_ids": ids, "labels": ids, "output_router_logits": True, "use_cache": False}
    truth = run_fwd_bwd(model, inputs, torch.float64)
    ref = run_fwd_bwd(model, inputs, torch.float32)
    test = run_fwd_bwd(new, inputs, torch.float32)
    assert_aligned(test, ref, truth)


@pytest.mark.parametrize("backend", BACKENDS)
def test_greedy_generation_matches_hf(backend):
    """Prefill plus KV-cache decoding, including single-token steps where most experts receive no row."""
    model = _tiny_model().eval()
    new = copy.deepcopy(model)
    _swap_experts(new, backend)
    ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(3))
    kwargs = dict(max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=0)
    with torch.no_grad():
        expected = model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
        actual = new.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
    assert torch.equal(actual, expected)
