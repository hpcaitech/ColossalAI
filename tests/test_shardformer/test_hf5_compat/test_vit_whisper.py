"""HF 5.17 CPU contracts. Serial stages are NOT distributed PP validation.

Flash tests replace only the GPU kernel with explicit CPU attention; the real
ColossalAI projection/cache/mask/return adapters run, but no GPU kernel runs.
"""

import copy
import importlib
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("HF5 ViT and Whisper contracts", allow_module_level=True)

import torch
import transformers as hf

from colossalai.shardformer.shard import ShardConfig


def local(family):
    return importlib.import_module("colossalai.shardformer.modeling." + family)


def stage(index, total=2):
    return SimpleNamespace(
        stage=index,
        num_stages=total,
        is_interleave=False,
        use_zbv=False,
        is_first_stage=lambda **kw: index == 0,
        is_last_stage=lambda **kw: index == total - 1,
    )


def pair(family, variant="base", backend="eager"):
    if family == "vit":
        config = hf.ViTConfig(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
            encoder_stride=8,
            num_labels=3,
            attn_implementation=backend,
        )
        cls = {
            "base": hf.ViTModel,
            "classification": hf.ViTForImageClassification,
            "masked": hf.ViTForMaskedImageModeling,
        }[variant]
    else:
        config = hf.WhisperConfig(
            vocab_size=32,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=2,
            encoder_ffn_dim=48,
            decoder_ffn_dim=48,
            num_mel_bins=8,
            max_source_positions=8,
            max_target_positions=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            num_labels=3,
            classifier_proj_size=16,
            use_cache=False,
            apply_spec_augment=False,
            attn_implementation=backend,
        )
        cls = {
            "base": hf.WhisperModel,
            "generation": hf.WhisperForConditionalGeneration,
            "classification": hf.WhisperForAudioClassification,
        }[variant]
    reference = cls(config).eval()
    return reference, copy.deepcopy(reference)


def data(family, variant="base"):
    if family == "vit":
        result = {"pixel_values": torch.randn(2, 3, 16, 16)}
        if variant == "classification":
            result["labels"] = torch.tensor([0, 2])
        if variant == "masked":
            result["bool_masked_pos"] = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)
    else:
        result = {"input_features": torch.randn(2, 8, 16)}
        if variant != "classification":
            result.update(decoder_input_ids=torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]]), use_cache=False)
        if variant == "generation":
            result["labels"] = torch.tensor([[4, 5, 6, 2], [7, 8, 9, 2]])
        if variant == "classification":
            result["labels"] = torch.tensor([0, 2])
    return result


def close(a, b, atol=1e-5, rtol=1e-3):
    assert a.shape == b.shape
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def compare_training(reference, actual, ref_out, out, record_property, logits_mask=None):
    for key in ("last_hidden_state", "pooler_output", "logits", "reconstruction", "loss", "encoder_last_hidden_state"):
        if key in ref_out:
            value, expected = out[key], ref_out[key]
            if key == "logits" and logits_mask is not None:
                # Fully masked padding queries have no supervised target; the
                # custom-mask kernel unblocks these rows. Compare valid tokens.
                value, expected = value[logits_mask], expected[logits_mask]
            close(value, expected)
            record_property(key + "_max_abs_error", (value - expected).abs().max().item())

    def objective(outputs):
        if "loss" in outputs:
            return outputs.loss
        result = outputs.last_hidden_state.square().mean()
        if "pooler_output" in outputs:
            result = result + outputs.pooler_output.square().mean()
        return result

    objective(ref_out).backward()
    objective(out).backward()
    compared = 0
    max_error = 0.0
    ref_params = dict(reference.named_parameters())
    for name, p in actual.named_parameters():
        r = ref_params[name]
        assert (p.grad is None) == (r.grad is None), name
        if p.grad is not None:
            close(p.grad, r.grad, atol=1e-6, rtol=1e-4)
            compared += 1
            max_error = max(max_error, (p.grad - r.grad).abs().max().item())
    assert compared > 10
    record_property("gradient_parameters_compared", compared)
    record_property("gradient_max_abs_error", max_error)
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.optim.SGD(actual.parameters(), lr=0.01).step()
    for name, p in actual.named_parameters():
        close(p, ref_params[name], atol=1e-6, rtol=1e-4)


@pytest.fixture
def cpu_flash(monkeypatch):
    from colossalai.shardformer.layer import ColoAttention

    calls = []

    def attention(q, k, v, attention_mask=None, attention_mask_type=None, dropout_p=0.0, scale=None, **kw):
        calls.append((tuple(q.shape), tuple(k.shape), dropout_p, scale))
        scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5 if scale is None else scale)
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=dropout_p > 0)
        return weights @ v

    monkeypatch.setattr(ColoAttention, "attention", staticmethod(attention))
    return calls


def patch_flash(model, family):
    module = local(family)
    if family == "vit":
        from transformers.models.vit.modeling_vit import ViTAttention

        for child in model.modules():
            if isinstance(child, ViTAttention):
                child.forward = MethodType(module.get_vit_flash_self_attention_forward(), child)
    else:
        from transformers.models.whisper.modeling_whisper import WhisperAttention, WhisperDecoder

        for child in model.modules():
            if isinstance(child, WhisperAttention):
                child.forward = MethodType(module.get_whisper_flash_attention_forward(), child)
            if isinstance(child, WhisperDecoder):
                child.forward = MethodType(
                    module.get_whisper_decoder_forward_for_flash_attention(
                        ShardConfig(enable_tensor_parallelism=False, enable_flash_attention=True)
                    ),
                    child,
                )


@pytest.mark.parametrize("family", ["vit", "whisper"])
def test_clean_process_import(family):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import colossalai.shardformer.modeling.{family}; import colossalai.shardformer.policies.{family}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "family,variant",
    [
        ("vit", "base"),
        ("vit", "classification"),
        ("vit", "masked"),
        ("whisper", "base"),
        ("whisper", "generation"),
        ("whisper", "classification"),
    ],
)
@pytest.mark.parametrize("tp", [False, True])
def test_policy_target_paths(family, variant, tp, record_property):
    reference, _ = pair(family, variant)
    pm = importlib.import_module("colossalai.shardformer.policies." + family)
    policy = getattr(pm, type(reference).__name__ + "Policy")()
    config = ShardConfig(enable_tensor_parallelism=False, enable_flash_attention=True)
    config.enable_tensor_parallelism = tp
    config._tensor_parallel_size = 2 if tp else 1
    policy.set_model(reference)
    policy.set_shard_config(config)
    policy.preprocess()
    descriptions = policy.module_policy()
    checked = 0
    for cls, desc in descriptions.items():
        instances = [m for m in reference.modules() if isinstance(m, cls)]
        # AudioClassification has only an encoder; the shared policy also
        # describes decoder types used by the other two Whisper variants.
        if (
            family == "whisper"
            and variant == "classification"
            and cls.__name__ in ("WhisperDecoder", "WhisperDecoderLayer")
        ):
            assert not instances
            continue
        assert instances, cls
        for m in instances:
            for replacement in desc.sub_module_replacement or []:
                assert m.get_submodule(replacement.suffix) is not None
                checked += 1
            for path in desc.attribute_replacement or {}:
                parent, _, attr = path.rpartition(".")
                assert hasattr(m.get_submodule(parent) if parent else m, attr), path
    assert checked > 0 or (family == "vit" and not tp)
    record_property("replacement_targets", checked)
    if family == "whisper" and tp and variant != "classification":
        from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer

        assert descriptions[WhisperDecoderLayer].attribute_replacement["encoder_attn.num_heads"] == 1


@pytest.mark.parametrize("variant", ["base", "classification", "masked"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_vit_pipeline_serial(variant, checkpoint, record_property):
    reference, actual = pair("vit", variant)
    module = local("vit")
    inputs = data("vit", variant)
    if checkpoint:
        actual.train()
        reference.train()
        actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vit = actual if variant == "base" else actual.vit
    result = None
    for i in range(2):
        manager = stage(i)
        indices = [i, i + 1]
        vit.forward = MethodType(module.ViTModel_pipeline_forward(manager, indices), vit)
        extra = {} if i == 0 else {"hidden_states": result["hidden_states"]}
        if variant == "base":
            result = actual(**inputs, **extra)
        else:
            forward = getattr(module, type(actual).__name__ + "_pipeline_forward")(manager, indices)
            result = forward(actual, **inputs, **extra)
    compare_training(reference, actual, reference(**inputs), result, record_property)


@pytest.mark.parametrize("variant", ["base", "generation", "classification"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_whisper_pipeline_serial(variant, checkpoint, record_property):
    reference, actual = pair("whisper", variant)
    forwards = local("whisper").WhisperPipelineForwards
    inputs = data("whisper", variant)
    if checkpoint:
        actual.train()
        reference.train()
        actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    function = {
        "base": forwards.whisper_model_forward,
        "generation": forwards.whisper_for_conditional_generation_forward,
        "classification": forwards.whisper_for_audio_classification_forward,
    }[variant]
    total = 2 if variant == "classification" else 4
    result = {}
    for i in range(total):
        result = function(
            actual,
            **inputs,
            **result,
            stage_manager=stage(i, total),
            stage_index=[i % 2, i % 2 + 1],
            decoder_starting_stage=2,
            shard_config=ShardConfig(enable_tensor_parallelism=False),
        )
        if i != total - 1:
            assert isinstance(result, dict)
    compare_training(reference, actual, reference(**inputs), result, record_property)


@pytest.mark.parametrize(
    "family,variant",
    [
        ("vit", "base"),
        ("vit", "classification"),
        ("vit", "masked"),
        ("whisper", "base"),
        ("whisper", "generation"),
        ("whisper", "classification"),
    ],
)
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_flash_adapter_training_cpu_kernel(family, variant, backend, cpu_flash, record_property):
    reference, actual = pair(family, variant, backend)
    patch_flash(actual, family)
    inputs = data(family, variant)
    compare_training(reference, actual, reference(**inputs), actual(**inputs), record_property)
    assert len(cpu_flash) >= 2


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", ["none", "left", "right"])
@pytest.mark.parametrize("jit", [False, True])
def test_whisper_cache_decode(backend, padding, jit, cpu_flash, record_property):
    reference, actual = pair("whisper", "generation", backend)
    patch_flash(actual, "whisper")
    if jit:
        from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func

        for layer in actual.model.decoder.layers:
            layer.forward = MethodType(local("whisper").get_jit_fused_whisper_decoder_layer_forward(), layer)
            layer.dropout_add = MethodType(get_jit_fused_dropout_add_func(), layer)
    features = torch.randn(2, 8, 16)
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9], [1, 9, 8, 7, 6, 5, 4]])
    mask = torch.ones_like(ids)
    if padding == "left":
        mask[0, :2] = 0
    if padding == "right":
        mask[0, 2:4] = 0
    caches = [None, None]
    counts = [0, 0]
    handles = []
    for i, layer in enumerate(actual.model.decoder.layers):

        def counter(m, a, o, idx=i):
            counts[idx] += 1

        handles.append(layer.encoder_attn.k_proj.register_forward_hook(counter))
    first_cross = []
    with torch.no_grad():
        encoders = [m.get_encoder()(features) for m in (reference, actual)]
        for end in range(4, 8):
            start = 0 if end == 4 else end - 1
            outputs = []
            for j, m in enumerate((reference, actual)):
                out = m(
                    encoder_outputs=encoders[j],
                    decoder_input_ids=ids[:, start:end],
                    decoder_attention_mask=mask[:, :end],
                    past_key_values=caches[j],
                    use_cache=True,
                )
                caches[j] = out.past_key_values
                outputs.append(out)
            full = reference(
                encoder_outputs=encoders[0],
                decoder_input_ids=ids[:, :end],
                decoder_attention_mask=mask[:, :end],
                use_cache=False,
            )
            valid = mask[:, end - 1].bool()
            close(outputs[1].logits[valid, -1], outputs[0].logits[valid, -1])
            close(outputs[1].logits[valid, -1], full.logits[valid, -1])
            for i in range(2):
                assert caches[1].self_attention_cache.get_seq_length(i) == end
                assert caches[1].cross_attention_cache.get_seq_length(i) == 8
                cross = caches[1].cross_attention_cache.layers[i].keys
                if end == 4:
                    first_cross.append(cross.clone())
                else:
                    torch.testing.assert_close(cross, first_cross[i], atol=0, rtol=0)
    assert counts == [1, 1]
    for h in handles:
        h.remove()
    record_property("self_cache_lengths", "4,5,6,7")
    record_property("cross_projection_calls", str(counts))


@pytest.mark.parametrize("family", ["vit", "whisper"])
def test_jit_layer_contract(family, record_property):
    from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func

    reference, actual = pair(family, "classification" if family == "vit" else "generation")
    module = local(family)
    for m in actual.modules():
        if family == "vit" and type(m).__name__ == "ViTLayer":
            factory = module.get_jit_fused_vit_layer_forward
        elif family == "whisper" and type(m).__name__ == "WhisperEncoderLayer":
            factory = module.get_jit_fused_whisper_encoder_layer_forward
        elif family == "whisper" and type(m).__name__ == "WhisperDecoderLayer":
            factory = module.get_jit_fused_whisper_decoder_layer_forward
        else:
            continue
        m.forward = MethodType(factory(), m)
        m.dropout_add = MethodType(get_jit_fused_dropout_add_func(), m)
    inputs = data(family, "classification" if family == "vit" else "generation")
    compare_training(reference, actual, reference(**inputs), actual(**inputs), record_property)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_whisper_mask_helper(backend):
    reference, _ = pair("whisper", backend=backend)
    module = local("whisper")
    hidden = torch.randn(2, 4, 32)
    config = ShardConfig(enable_tensor_parallelism=False, enable_flash_attention=True)
    mask = module._get_attention_mask(reference.decoder, config, hidden, 0, torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))
    assert mask.shape == (2, 1, 4, 4)
    assert mask[0, 0, 3, 3] < -1e20
    assert mask[1, 0, 1, 2] < -1e20
    assert mask[1, 0, 3, 0] == 0
    assert reference.config._attn_implementation == backend


def test_decoder_cache_allocation():
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    from colossalai.shardformer._utils import prepare_hf_encoder_decoder_cache

    ref, _ = pair("whisper")
    ref.config.encoder_layers = 4
    cache = prepare_hf_encoder_decoder_cache(ref.config)
    assert isinstance(cache, EncoderDecoderCache)
    assert len(cache.self_attention_cache.layers) == 2
    assert ref.config.encoder_layers == 4
    assert prepare_hf_encoder_decoder_cache(ref.config, cache) is cache
    assert isinstance(prepare_hf_encoder_decoder_cache(ref.config, DynamicCache()), EncoderDecoderCache)
    with pytest.raises(TypeError):
        prepare_hf_encoder_decoder_cache(ref.config, ())


@pytest.mark.parametrize(
    "family,variant",
    [
        ("vit", "base"),
        ("vit", "classification"),
        ("vit", "masked"),
        ("whisper", "base"),
        ("whisper", "generation"),
        ("whisper", "classification"),
    ],
)
def test_shardformer_optimize_flash_cpu_kernel(family, variant, cpu_flash, record_property):
    from colossalai.shardformer import ShardFormer

    reference, actual = pair(family, variant)
    sharder = ShardFormer(
        ShardConfig(enable_tensor_parallelism=False, enable_flash_attention=True, make_vocab_size_divisible_by=1)
    )
    actual, _ = sharder.optimize(actual)
    inputs = data(family, variant)
    compare_training(reference, actual, reference(**inputs), actual(**inputs), record_property)
    assert len(cpu_flash) >= 2


@pytest.mark.parametrize("padding", ["left", "right"])
def test_whisper_decoder_checkpoint_and_capture(padding, cpu_flash, record_property):
    reference, actual = pair("whisper", "generation")
    patch_flash(actual, "whisper")
    reference.train()
    actual.train()
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    inputs = data("whisper", "generation")
    mask = torch.ones(2, 4, dtype=torch.long)
    mask[0, 0 if padding == "left" else -1] = 0
    inputs["decoder_attention_mask"] = mask
    inputs["labels"] = inputs["labels"].masked_fill(mask == 0, -100)
    # Explicitly requesting cache while checkpointing must not mutate a cache.
    out = actual(**{**inputs, "use_cache": True}, output_hidden_states=True)
    ref = reference(**inputs, output_hidden_states=True)
    assert out.past_key_values is None
    assert len(out.decoder_hidden_states) == len(ref.decoder_hidden_states) == 3
    compare_training(reference, actual, ref, out, record_property, logits_mask=mask.bool())


def test_whisper_cross_mask_attention(cpu_flash):
    ref, _ = pair("whisper")
    reference = ref.decoder.layers[0].encoder_attn
    actual = copy.deepcopy(reference)
    actual.forward = MethodType(local("whisper").get_whisper_flash_attention_forward(), actual)
    hidden = torch.randn(2, 3, 32)
    audio = torch.randn(2, 8, 32)
    mask = torch.zeros(2, 1, 3, 8)
    mask[0, :, :, 5:] = torch.finfo(torch.float32).min
    expected = reference(hidden, key_value_states=audio, attention_mask=mask)[0]
    out = actual(hidden, key_value_states=audio, attention_mask=mask)[0]
    close(out, expected)
    audio[0, 5:] = 10000
    close(actual(hidden, key_value_states=audio, attention_mask=mask)[0], out)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_vit_padding_mask_and_tuple(backend, cpu_flash):
    ref, actual = pair("vit", backend=backend)
    patch_flash(actual, "vit")
    pixels = torch.randn(2, 3, 16, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    expected = ref(pixels, attention_mask=mask, return_dict=False)
    out = actual(pixels, attention_mask=mask, return_dict=False)
    assert isinstance(out, tuple) and len(out) == len(expected)
    for a, b in zip(out, expected):
        close(a, b)
    module = local("vit")
    state = module.ViTModel_pipeline_forward(stage(0), [0, 1])(actual, pixels, attention_mask=mask)
    result = module.ViTModel_pipeline_forward(stage(1), [1, 2])(
        actual, hidden_states=state["hidden_states"], attention_mask=mask, return_dict=False
    )
    for a, b in zip(result, expected):
        close(a, b)


def test_whisper_audio_weighted_pipeline(record_property):
    reference, actual = pair("whisper", "classification")
    # Construct with the feature enabled so HF creates trainable layer_weights.
    config = copy.deepcopy(reference.config)
    config.use_weighted_layer_sum = True
    reference = hf.WhisperForAudioClassification(config).eval()
    actual = copy.deepcopy(reference)
    inputs = data("whisper", "classification")
    fn = local("whisper").WhisperPipelineForwards.whisper_for_audio_classification_forward
    state = fn(actual, **inputs, stage_manager=stage(0), stage_index=[0, 1], decoder_starting_stage=2)
    assert len(state["encoder_states"]) == 1
    out = fn(actual, **inputs, **state, stage_manager=stage(1), stage_index=[1, 2], decoder_starting_stage=2)
    compare_training(reference, actual, reference(**inputs), out, record_property)


@pytest.mark.parametrize("family", ["vit", "whisper"])
def test_pipeline_held_layers(family):
    from colossalai.pipeline.stage_manager import PipelineStageManager

    reference, _ = pair(family)
    pm = importlib.import_module("colossalai.shardformer.policies." + family)
    total = 2 if family == "vit" else 4
    held = []
    for i in range(total):
        manager = stage(i, total)
        manager.num_model_chunks = 1
        manager.num_layers_per_stage = None
        manager.distribute_layers = MethodType(PipelineStageManager.distribute_layers, manager)
        manager.get_stage_index = MethodType(PipelineStageManager.get_stage_index, manager)
        config = ShardConfig(enable_tensor_parallelism=False, pipeline_stage_manager=manager)
        policy = getattr(pm, type(reference).__name__ + "Policy")()
        policy.set_model(reference)
        policy.set_shard_config(config)
        policy.preprocess()
        policy.module_policy()
        held.extend(policy.get_held_layers())
    layers = (
        list(reference.layers) if family == "vit" else list(reference.encoder.layers) + list(reference.decoder.layers)
    )
    for layer in layers:
        assert sum(x is layer for x in held) == 1


def test_whisper_jit_stays_disabled():
    pm = importlib.import_module("colossalai.shardformer.policies.whisper")
    ref, _ = pair("whisper")
    policy = pm.WhisperModelPolicy()
    config = ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)
    policy.set_model(ref)
    policy.set_shard_config(config)
    policy.preprocess()
    with pytest.warns(UserWarning, match="doesn't support jit"):
        policy.module_policy()
    assert not config.enable_jit_fused


@pytest.mark.parametrize("variant", ["base", "generation"])
def test_whisper_pipeline_tuple_and_default_cache(variant):
    reference, actual = pair("whisper", variant)
    actual.config.use_cache = True
    inputs = data("whisper", variant)
    inputs.pop("use_cache")
    fn = getattr(
        local("whisper").WhisperPipelineForwards,
        "whisper_model_forward" if variant == "base" else "whisper_for_conditional_generation_forward",
    )
    state = {}
    for i in range(2):
        state = fn(
            actual,
            **inputs,
            **state,
            return_dict=False,
            stage_manager=stage(i),
            stage_index=[0, 2],
            decoder_starting_stage=1,
            shard_config=ShardConfig(enable_tensor_parallelism=False),
        )
    expected = reference(**inputs, use_cache=False, return_dict=False)
    assert isinstance(state, tuple) and len(state) == len(expected)
    for a, b in zip(state, expected):
        close(a, b)


def test_vit_jit_policy_keeps_exact_activation(record_property):
    from colossalai.shardformer import ShardFormer

    reference, actual = pair("vit", "classification")
    # Large enough activations to expose replacing erf GELU by tanh GELU.
    with torch.no_grad():
        for layer in reference.vit.layers:
            layer.mlp.fc1.weight.mul_(10)
    actual.load_state_dict(reference.state_dict())
    actual, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False, enable_jit_fused=True)).optimize(actual)
    assert all(hasattr(layer, "dropout_add") for layer in actual.vit.layers)
    inputs = data("vit", "classification")
    compare_training(reference, actual, reference(**inputs), actual(**inputs), record_property)


def test_vit_interleaved_stage_functions_serially(record_property):
    reference, _ = pair("vit")
    config = copy.deepcopy(reference.config)
    config.num_hidden_layers = 4
    reference = hf.ViTModel(config).eval()
    actual = copy.deepcopy(reference)
    pixels = torch.randn(2, 3, 16, 16)
    state = {}
    for logical in range(4):
        physical = logical % 2
        chunk = logical // 2
        manager = stage(physical)
        manager.is_interleave = True
        manager.model_chunk_id = chunk
        manager.is_first_stage = lambda **kw: logical == 0
        manager.is_last_stage = lambda **kw: logical == 3
        indices = [[physical, physical + 1], [physical + 2, physical + 3]]
        state = local("vit").ViTModel_pipeline_forward(manager, indices)(actual, pixel_values=pixels, **state)
    compare_training(reference, actual, reference(pixels), state, record_property)


@pytest.mark.parametrize("family", ["vit", "whisper"])
def test_flash_training_dropout_argument(family, cpu_flash):
    reference, actual = pair(family)
    patch_flash(actual, family)
    attn = actual.layers[0].attention if family == "vit" else actual.encoder.layers[0].self_attn
    if family == "vit":
        attn.attention_dropout = 0.2
    else:
        attn.dropout = 0.2
    hidden = torch.randn(2, 4, 32)
    attn.train()
    attn(hidden)
    assert cpu_flash[-1][2] == 0.2
    attn.eval()
    attn(hidden)
    assert cpu_flash[-1][2] == 0.0
