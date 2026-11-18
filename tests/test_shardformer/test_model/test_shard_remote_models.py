"""Pinned remote-code adapter contracts on CPU, without pretrained weights."""

import copy
import os
import sys
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub.errors import LocalEntryNotFoundError

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.ouro import get_ouro_model_class
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from tests.kit.remote_models import (
    REMOTE_MODELS,
    build_remote_model,
    forward_and_loss,
    load_remote_classes,
    tiny_config,
    tiny_data,
)

pytestmark = pytest.mark.skipif(transformers.__version__ != "5.17.0", reason="Pinned HF5.17 remote-code contracts")


@pytest.fixture(scope="module", autouse=True)
def offline_snapshots():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for key in REMOTE_MODELS:
            try:
                load_remote_classes(key)
            except LocalEntryNotFoundError:
                if os.environ.get("COLOSSALAI_REQUIRE_REMOTE_MODELS") == "1":
                    pytest.fail(f"Missing pinned remote-code snapshot for {key}")
                pytest.skip(f"Prepare the pinned remote-code snapshots first: missing {key}")
        yield
    finally:
        torch.set_num_threads(previous)


def manual_causal_mask(*, input_embeds, attention_mask=None, **kwargs):
    """Independent eager reference, without using HF's new mask helper."""
    batch, length = input_embeds.shape[:2]
    allowed = torch.ones(length, length, device=input_embeds.device, dtype=torch.bool).tril()
    allowed = allowed[None, None].expand(batch, 1, length, length)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()
    return torch.zeros_like(allowed, dtype=input_embeds.dtype).masked_fill(
        ~allowed, torch.finfo(input_embeds.dtype).min
    )


@contextmanager
def original_reference(key, monkeypatch):
    if not key.startswith("ouro"):
        yield
        return
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    _, _, original = load_remote_classes(key)
    module = sys.modules[original.__module__]
    with monkeypatch.context() as patch:
        # Test-only bridge to instantiate the unchanged official reference.
        # Production execution takes place after these patches have been removed.
        compute = Qwen2RotaryEmbedding.compute_default_rope_parameters
        patch.setattr(module, "ROPE_INIT_FUNCTIONS", {**module.ROPE_INIT_FUNCTIONS, "default": compute})
        patch.setattr(
            module.OuroRotaryEmbedding, "compute_default_rope_parameters", staticmethod(compute), raising=False
        )
        patch.setattr(module, "create_causal_mask", manual_causal_mask)
        yield


def compare_trainable(reference, candidate, *, gradients=False):
    expected, actual = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert expected.keys() == actual.keys()
    compared = 0
    for name, parameter in expected.items():
        other = actual[name]
        left, right = (parameter.grad, other.grad) if gradients else (parameter, other)
        assert (left is None) == (right is None), name
        if left is None:
            continue
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5, msg=name)
        compared += 1
    assert compared > 0


@pytest.mark.parametrize("key", REMOTE_MODELS)
def test_native_policy_training(key, monkeypatch):
    torch.manual_seed(42)
    with original_reference(key, monkeypatch):
        reference = build_remote_model(key, adapted=False).train()
        data = tiny_data(key)
        expected, expected_loss = forward_and_loss(reference, key, data)
        expected_loss.backward()

    candidate = build_remote_model(key).train()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    before_ids = {name: id(parameter) for name, parameter in candidate.named_parameters()}
    expected_family = "llada" if key == "llada" else "llada2_moe" if key.startswith("llada") else "ouro"
    assert get_autopolicy(candidate).family == expected_family
    candidate, shared = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(candidate)
    assert shared == []
    assert before_ids == {name: id(parameter) for name, parameter in candidate.named_parameters()}
    calls = {}
    handles = []
    for name, module in candidate.named_modules():
        if type(module).__name__ in {"OuroDecoderLayer", "LLaDA2MoeSparseMoeBlock"}:
            calls[name] = 0

            def record(module, inputs, output, name=name):
                calls[name] += 1

            handles.append(module.register_forward_hook(record))
    try:
        actual, actual_loss = forward_and_loss(candidate, key, data)
        actual_loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert actual.logits.shape == (2, 16, 128)
    assert torch.isfinite(actual_loss)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-5)
    if key.startswith("ouro"):
        assert len(calls) == 2 and set(calls.values()) == {2}
        assert candidate.model.early_exit_gate.weight.grad.abs().sum() > 0
        manual_loss = F.cross_entropy(actual.logits[:, :-1].reshape(-1, 128), data["labels"][:, 1:].reshape(-1))
    else:
        if key != "llada":
            assert len(calls) == 1 and set(calls.values()) == {1}
        manual_loss = F.cross_entropy(actual.logits.reshape(-1, 128), data["labels"].reshape(-1))
    torch.testing.assert_close(actual_loss, manual_loss)
    compare_trainable(reference, candidate, gradients=True)
    before = {name: parameter.detach().clone() for name, parameter in candidate.named_parameters()}
    for model in (reference, candidate):
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0).step()
    compare_trainable(reference, candidate)
    assert any(not torch.equal(before[name], parameter) for name, parameter in candidate.named_parameters())


@pytest.mark.parametrize("key", REMOTE_MODELS)
def test_save_pretrained_roundtrip(key, tmp_path):
    model = build_remote_model(key).eval()
    data = tiny_data(key)
    with torch.no_grad():
        expected, expected_loss = forward_and_loss(model, key, data)
    model.save_pretrained(tmp_path, max_shard_size="100KB")
    config = type(model.config).from_pretrained(tmp_path, local_files_only=True)
    restored, info = type(model).from_pretrained(
        tmp_path,
        config=config,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
        output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"], info
    assert model.state_dict().keys() == restored.state_dict().keys()
    assert get_autopolicy(restored).family == get_autopolicy(model).family
    with torch.no_grad():
        actual, actual_loss = forward_and_loss(restored.eval(), key, data)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    restored.train()
    _, loss = forward_and_loss(restored, key, data)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in restored.parameters())


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_ouro_isolation_mask_and_causality(key, padding_side, monkeypatch):
    _, _, original = load_remote_classes(key)
    module = sys.modules[original.__module__]
    registry, mask, constructor = module.ROPE_INIT_FUNCTIONS, module.create_causal_mask, original.__init__
    adapter = get_ouro_model_class(original)
    assert get_ouro_model_class(original) is adapter
    with original_reference(key, monkeypatch):
        reference = build_remote_model(key, adapted=False).eval()
        data = tiny_data(key)
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        padding = slice(-3, None) if padding_side == "right" else slice(0, 3)
        data["attention_mask"][:, padding] = 0
        data["labels"][:, padding] = -100
        with torch.no_grad():
            expected, _ = forward_and_loss(reference, key, data)

    candidate = adapter(tiny_config(key)).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        actual, _ = forward_and_loss(candidate, key, data)
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
        inputs = data["input_ids"]
        plain = candidate(input_ids=inputs).logits
        changed = inputs.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 128
        future_changed = candidate(input_ids=changed).logits
        torch.testing.assert_close(plain[:, :8], future_changed[:, :8], atol=0, rtol=0)
        via_embeds = candidate(inputs_embeds=candidate.model.embed_tokens(inputs)).logits
        torch.testing.assert_close(plain, via_embeds, atol=0, rtol=0)
    assert module.ROPE_INIT_FUNCTIONS is registry and module.create_causal_mask is mask
    assert original.__init__ is constructor
    assert "compute_default_rope_parameters" not in module.OuroRotaryEmbedding.__dict__
    # Creating / using the adapter must not silently fix the original class.
    with pytest.raises(KeyError, match="default"):
        original(tiny_config(key))


def test_two_ouro_snapshots_coexist():
    models = [build_remote_model(key).eval() for key in ("ouro14", "ouro26")]
    assert type(models[0]) is not type(models[1])
    inputs = tiny_data("ouro14")["input_ids"]
    with torch.no_grad():
        before = models[0](inputs).logits.clone()
        assert torch.isfinite(models[1](inputs).logits).all()
        torch.testing.assert_close(models[0](inputs).logits, before, atol=0, rtol=0)


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
def test_ouro_tied_weights_roundtrip(key, tmp_path):
    config = tiny_config(key)
    config.tie_word_embeddings = True
    model = build_remote_model(key, config=config).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.save_pretrained(tmp_path)
    restored = (
        type(model)
        .from_pretrained(
            tmp_path,
            config=copy.deepcopy(config),
            attn_implementation="eager",
            local_files_only=True,
            dtype=torch.float32,
        )
        .eval()
    )
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    inputs = tiny_data(key)["input_ids"]
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).logits, restored(inputs).logits, atol=0, rtol=0)


@pytest.mark.parametrize("key", ["llada", "llada21", "ouro14"])
@pytest.mark.parametrize(
    "flag",
    [
        "enable_tensor_parallelism",
        "enable_sequence_parallelism",
        "enable_flash_attention",
        "enable_jit_fused",
        "enable_fused_normalization",
        "enable_all_optimization",
        "fp8_communication",
        "pipeline_stage_manager",
        "ep_group",
        "moe_dp_group",
        "gradient_checkpoint_config",
        "extra_kwargs",
    ],
)
def test_policy_rejects_unvalidated_modes_before_mutation(key, flag):
    model = build_remote_model(key)
    config = ShardConfig(enable_tensor_parallelism=False)
    # Set after construction to avoid requiring a distributed process group.
    setattr(config, flag, True if flag != "extra_kwargs" else {"unknown_option": True})
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(NotImplementedError, match=flag):
        ShardFormer(config).optimize(model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"use_cache": True}, "use_cache"),
        ({"use_sliding_window": True}, "sliding"),
        ({"_attn_implementation": "sdpa"}, "eager"),
        ({"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}}, "RoPE"),
        ({"rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.5, "rope_theta": 10000.0}}, "RoPE"),
    ],
)
def test_ouro_rejects_unsupported_config(updates, message):
    config = tiny_config("ouro14")
    config.update(updates)
    with pytest.raises(NotImplementedError, match=message):
        build_remote_model("ouro14", config=config)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"use_cache": True}, "KV cache"),
        ({"past_key_values": object()}, "KV cache"),
        ({"cache_position": torch.arange(16) + 1}, "cache_position"),
        ({"position_ids": (torch.arange(16) + 1)[None]}, "position_ids"),
        ({"attention_mask": {"full_attention": None}}, "2D"),
        ({"attention_mask": torch.full((2, 16), 0.5)}, "zeros and ones"),
    ],
)
def test_ouro_rejects_unsupported_runtime(kwargs, message):
    model = build_remote_model("ouro14")
    with pytest.raises((NotImplementedError, ValueError), match=message):
        model(input_ids=tiny_data("ouro14")["input_ids"], **kwargs)


def test_explicit_policy_and_version_boundary(monkeypatch):
    model = build_remote_model("llada")
    policy = get_autopolicy(model)
    with pytest.raises(NotImplementedError, match="enable_jit_fused"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)).optimize(model, policy=policy)
    monkeypatch.setattr(sys.modules["transformers"], "__version__", "5.18.0")
    with pytest.raises(NotImplementedError, match="5.17.0"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)


@pytest.mark.parametrize("key", ["llada", "llada21", "llada22"])
def test_llada_tied_weights_roundtrip(key, tmp_path):
    config = tiny_config(key)
    if key == "llada":
        config.weight_tying = True
    else:
        config.tie_word_embeddings = True
    model = build_remote_model(key, config=config).eval()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    model.save_pretrained(tmp_path)
    restored = (
        type(model)
        .from_pretrained(
            tmp_path,
            config=copy.deepcopy(config),
            attn_implementation="eager",
            local_files_only=True,
            dtype=torch.float32,
        )
        .eval()
    )
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    with torch.no_grad():
        expected, _ = forward_and_loss(model, key, tiny_data(key))
        actual, _ = forward_and_loss(restored, key, tiny_data(key))
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)


def test_llada_adapter_preserves_official_initialization():
    torch.manual_seed(101)
    reference = build_remote_model("llada", adapted=False)
    torch.manual_seed(101)
    candidate = build_remote_model("llada")
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("key", ["llada", "llada21", "llada22"])
def test_original_class_requires_loading_adapter(key):
    model = build_remote_model(key, adapted=False)
    with pytest.raises(NotImplementedError, match="Construct the model with"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
