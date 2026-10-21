import copy
import importlib
from types import SimpleNamespace

import pytest

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("HF5 core migration contracts", allow_module_level=True)

import torch
import transformers as hf

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer._utils import check_hf5_policy
from colossalai.shardformer.policies.auto_policy import get_autopolicy


def model_and_data(family):
    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        use_cache=False,
        attn_implementation="eager",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    if family == "bert":
        model = hf.BertForMaskedLM(
            hf.BertConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                max_position_embeddings=16,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
                attn_implementation="eager",
            )
        )
    elif family == "llama":
        model = hf.LlamaForCausalLM(hf.LlamaConfig(**common))
    elif family == "mixtral":
        model = hf.MixtralForCausalLM(hf.MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2))
    else:
        common["num_key_value_heads"] = 4
        model = hf.DeepseekV3ForCausalLM(
            hf.DeepseekV3Config(
                **common,
                q_lora_rank=None,
                kv_lora_rank=8,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=8,
                n_routed_experts=4,
                n_shared_experts=1,
                num_experts_per_tok=2,
                n_group=1,
                topk_group=1,
                moe_intermediate_size=16,
                first_k_dense_replace=0,
            )
        )
    tokens = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])
    return model.eval(), dict(input_ids=tokens, labels=tokens.clone())


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("family", ["bert", "llama", "mixtral", "deepseek_v3"])
def test_integrated_non_ep_training(family, seed):
    torch.manual_seed(seed)
    reference, data = model_and_data(family)
    candidate = copy.deepcopy(reference)
    policy = get_autopolicy(candidate)
    assert type(policy).__module__.endswith(f"_{family}_tf5")
    assert importlib.import_module("colossalai.shardformer.policies." + family).__dict__[type(policy).__name__] is type(
        policy
    )
    candidate, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(candidate)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
    params = dict(candidate.named_parameters())
    ref_params = dict(reference.named_parameters())
    assert params.keys() == ref_params.keys()
    optimizer_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert optimizer_ids == {id(param) for param in candidate.parameters()}
    for step in range(2):
        ref_optimizer.zero_grad()
        optimizer.zero_grad()
        if step == 1:
            reference.gradient_checkpointing_enable()
            candidate.gradient_checkpointing_enable()
            reference.train()
            candidate.train()
        expected, actual = reference(**data), candidate(**data)
        assert torch.isfinite(actual.logits).all()
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=1e-3)
        expected.loss.backward()
        actual.loss.backward()
        for name, parameter in ref_params.items():
            other = params[name]
            assert (parameter.grad is None) == (other.grad is None), name
            if parameter.grad is not None:
                assert torch.isfinite(other.grad).all(), name
                torch.testing.assert_close(other.grad, parameter.grad, atol=1e-6, rtol=1e-4)
        ref_optimizer.step()
        optimizer.step()
        for name, parameter in ref_params.items():
            torch.testing.assert_close(params[name], parameter, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("family", ["mixtral", "deepseek_v3"])
def test_repaired_ep_policy_is_not_blanket_gated(family):
    check_hf5_policy(family, SimpleNamespace(expert_parallel_size=2))


@pytest.mark.parametrize("family", ["mixtral", "deepseek_v3"])
def test_explicit_dispatched_ep_policy_passes_compatibility_check(family):
    model, _ = model_and_data(family)
    policy = get_autopolicy(model)
    config = ShardConfig(enable_tensor_parallelism=False)
    config._expert_parallel_size = 2
    before = {name: id(parameter) for name, parameter in model.named_parameters()}
    # Policy selection must not mutate weights; actual EP training/checkpoint
    # acceptance belongs to the distributed optimizer regressions.
    check_hf5_policy(family, config)
    assert type(policy).__module__.endswith(f"_{family}_tf5")
    assert before == {name: id(parameter) for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("v5", [False, True])
def test_dispatch_selects_actual_implementation(monkeypatch, v5):
    from colossalai import _compat

    calls = []
    exported = SimpleNamespace(public_name=object(), _private_name=object(), __name__="implementation")
    monkeypatch.setattr(_compat, "is_transformers_v5", lambda: v5)

    def import_impl(name):
        calls.append(name)
        return exported

    monkeypatch.setattr(_compat, "import_module", import_impl)
    namespace = {}
    _compat.reexport_transformers_impl(namespace, "example", "_llama")
    assert calls == ["example._llama_tf5" if v5 else "example._llama_tf4"]
    assert namespace["public_name"] is exported.public_name
    assert namespace["_private_name"] is exported._private_name
    assert "__name__" not in namespace
