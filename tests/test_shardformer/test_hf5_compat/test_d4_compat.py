"""D4 tiny CPU integration and early support-boundary errors."""

import copy
from types import SimpleNamespace

import pytest

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("HF5 long-tail policy contracts", allow_module_level=True)

import torch
import transformers as hf

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer._utils import check_hf5_policy, hf5_policy_error
from colossalai.shardformer.policies.auto_policy import PolicyLocation, import_policy
from tests.test_shardformer.test_hf5_compat.bf16_checks import compare_parameters


def model_and_data(family, backend):
    if family == "opt":
        config = hf.OPTConfig(
            vocab_size=64,
            hidden_size=32,
            word_embed_proj_dim=32,
            ffn_dim=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            dropout=0,
            attention_dropout=0,
            use_cache=False,
            attn_implementation=backend,
        )
        model = hf.OPTForCausalLM(config)
    elif family == "blip2":
        config = hf.Blip2Config(
            vision_config=dict(
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=16,
                patch_size=8,
                attention_dropout=0,
            ),
            qformer_config=dict(
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                encoder_hidden_size=32,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0,
            ),
            text_config=dict(
                model_type="opt",
                vocab_size=64,
                hidden_size=32,
                ffn_dim=48,
                num_hidden_layers=1,
                num_attention_heads=4,
                word_embed_proj_dim=32,
                dropout=0,
                attention_dropout=0,
                use_cache=False,
            ),
            num_query_tokens=2,
            image_token_id=63,
        )
        config._attn_implementation = backend
        model = hf.Blip2ForConditionalGeneration(config).eval()
        inputs = torch.tensor([[2, 3, 4, 5], [2, 6, 7, 8]])
        inputs[:, :2] = 63
        labels = inputs.clone()
        labels[:, :2] = -100
        return model, dict(pixel_values=torch.randn(2, 3, 16, 16), input_ids=inputs, labels=labels)
    else:
        config = hf.CohereConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            attention_dropout=0,
            use_cache=False,
            attn_implementation=backend,
        )
        model = hf.CohereForCausalLM(config)
    inputs = torch.tensor([[2, 3, 4, 5], [2, 6, 7, 8]])
    return model.eval(), dict(input_ids=inputs, labels=inputs.clone(), use_cache=False)


@pytest.mark.parametrize("family", ["opt", "command", "blip2"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_training_integration(family, backend):
    reference, data = model_and_data(family, backend)
    candidate = copy.deepcopy(reference)
    candidate, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(candidate)
    expected, actual = reference(**data), candidate(**data)
    torch.testing.assert_close(actual.logits, expected.logits, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(actual.loss, expected.loss, atol=1e-5, rtol=1e-3)
    expected.loss.backward()
    actual.loss.backward()
    compare_parameters(reference, candidate)
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.optim.SGD(candidate.parameters(), lr=0.01).step()
    compare_parameters(reference, candidate, gradients=False)


@pytest.mark.parametrize("family", ["opt", "command", "blip2"])
@pytest.mark.parametrize(
    "flag", ["enable_tensor_parallelism", "enable_flash_attention", "enable_jit_fused", "pipeline_stage_manager"]
)
def test_unsupported_configuration_error(family, flag):
    config = SimpleNamespace(**{flag: True})
    with pytest.raises(NotImplementedError, match="HF5-COMPATIBILITY") as exc:
        check_hf5_policy(family, config)
    assert family in str(exc.value)
    assert hf.__version__ in str(exc.value)


@pytest.mark.parametrize("family", ["chatglm2", "deepseek"])
def test_remote_code_gate_before_import(family, monkeypatch):
    from colossalai.shardformer.policies import auto_policy

    def forbidden_import(*args, **kwargs):
        pytest.fail("Denied policy must not be imported")

    monkeypatch.setattr(auto_policy.importlib, "import_module", forbidden_import)
    with pytest.raises(NotImplementedError, match="remote-code"):
        import_policy(PolicyLocation(family, "UnusedPolicy"))


def test_legacy_version_not_gated(monkeypatch):
    from colossalai.shardformer import _utils

    monkeypatch.setattr(_utils, "is_hf5", lambda: False)
    assert hf5_policy_error("command", SimpleNamespace(enable_tensor_parallelism=True)) is None


@pytest.mark.parametrize("family", ["bert", "llama", "qwen2", "qwen3", "mixtral", "deepseek_v3", "vit", "whisper"])
def test_core_not_blocked_by_tail_gate(family):
    assert hf5_policy_error(family, SimpleNamespace(enable_tensor_parallelism=True)) is None


def test_unknown_model_keeps_original_error():
    from colossalai.shardformer.policies.auto_policy import get_autopolicy

    with pytest.raises(NotImplementedError, match="Auto policy"):
        get_autopolicy(torch.nn.Linear(2, 2))


def test_blip2_obsolete_dropout_targets_removed():
    from colossalai.shardformer.policies.auto_policy import get_autopolicy

    model, _ = model_and_data("blip2", "eager")
    policy = get_autopolicy(model)
    config = ShardConfig(enable_tensor_parallelism=False)
    config.enable_tensor_parallelism = True
    config._tensor_parallel_size = 2
    policy.set_model(model)
    policy.set_shard_config(config)
    policy.preprocess()
    descriptions = policy.module_policy()
    removed = {"self_attn.dropout", "attention.attention.dropout", "crossattention.attention.dropout"}
    assert all(
        replacement.suffix not in removed
        for description in descriptions.values()
        for replacement in description.sub_module_replacement or []
    )


def test_explicit_builtin_policy_gate():
    from colossalai.shardformer.policies.opt import OPTPolicy

    model, _ = model_and_data("opt", "eager")
    config = ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(NotImplementedError, match="enable_jit_fused"):
        ShardFormer(config).optimize(model, policy=OPTPolicy())
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_custom_policy_not_gated():
    from colossalai.shardformer.policies.base_policy import Policy

    class CustomPolicy(Policy):
        def config_sanity_check(self):
            pass

        def preprocess(self):
            return self.model

        def module_policy(self):
            return {}

        def postprocess(self):
            return self.model

    model, _ = model_and_data("opt", "eager")
    ShardFormer(ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)).optimize(model, CustomPolicy())
