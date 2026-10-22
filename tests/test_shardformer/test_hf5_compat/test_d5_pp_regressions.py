"""Serial PP contract regressions; real distributed acceptance runs separately."""

import copy
from types import SimpleNamespace

import pytest

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("HF5 pipeline contracts", allow_module_level=True)

import torch
from transformers import MixtralConfig, MixtralModel

from colossalai.shardformer import ShardConfig
from colossalai.shardformer.modeling._mixtral_tf5 import MixtralPipelineForwards
from colossalai.shardformer.policies._deepseek_v3_tf5 import DeepseekV3ModelPolicy
from colossalai.shardformer.shard.utils import set_tensors_to_none
from tests.test_shardformer.test_hf5_compat.test_integrated_core import model_and_data


def stage(index, total=2):
    return SimpleNamespace(
        is_interleave=False,
        use_zbv=False,
        is_first_stage=lambda **kwargs: index == 0,
        is_last_stage=lambda **kwargs: index == total - 1,
        distribute_layers=lambda count: [count // total] * total,
        get_stage_index=lambda counts: (sum(counts[:index]), sum(counts[: index + 1])),
    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_mixtral_pp_causal_mask_with_flash_installed(monkeypatch, backend):
    from colossalai.shardformer.modeling import _mixtral_tf5 as implementation

    monkeypatch.setattr(implementation, "is_flash_attn_2_available", lambda: True)
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        use_cache=False,
        attn_implementation=backend,
    )
    model = MixtralModel(config).eval()
    candidate = copy.deepcopy(model)
    tokens = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])
    expected = model(input_ids=tokens).last_hidden_state

    def run(ids):
        first = MixtralPipelineForwards.mixtral_model_forward(
            candidate, input_ids=ids, stage_manager=stage(0), stage_index=(0, 1), use_cache=False
        )
        return MixtralPipelineForwards.mixtral_model_forward(
            candidate, hidden_states=first["hidden_states"], stage_manager=stage(1), stage_index=(1, 2), use_cache=False
        ).last_hidden_state

    actual = run(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)
    changed = tokens.clone()
    changed[:, -1] = 10
    # Expert token grouping can change GEMM rounding even for an unchanged
    # prefix; use the existing FP32 gradient-level bound, not bitwise equality.
    torch.testing.assert_close(run(changed)[:, :-1], actual[:, :-1], atol=1e-6, rtol=1e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for ref, value in zip(model.parameters(), candidate.parameters()):
        assert (ref.grad is None) == (value.grad is None)
        if ref.grad is not None:
            torch.testing.assert_close(value.grad, ref.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("index", [0, 1])
def test_deepseek_pp_preserves_rotary_buffer(index):
    model, _ = model_and_data("deepseek_v3")
    model = model.model
    policy = DeepseekV3ModelPolicy()
    policy.set_model(model)
    config = ShardConfig(enable_tensor_parallelism=False)
    config.pipeline_stage_manager = stage(index)
    policy.set_shard_config(config)
    held = policy.get_held_layers()
    assert model.rotary_emb in held
    set_tensors_to_none(model, exclude=set(held))
    assert model.rotary_emb.inv_freq is not None
    cos, sin = model.rotary_emb(torch.zeros(1, 4, 32), position_ids=torch.arange(4).unsqueeze(0))
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
