"""CPU regression tests of real ShardFormer Qwen forwards (no GPU kernels or collectives).

The pipeline checks execute stage functions serially; they do not validate a
distributed pipeline schedule. Run this directory with --confcutdir to avoid the
repository-wide accelerator fixture on CPU-only development machines.
"""

import copy
import importlib
from types import MethodType, SimpleNamespace

import pytest
import torch
import transformers
from packaging.version import Version

if Version(transformers.__version__) < Version("5.17.0"):
    pytest.skip("HF 5.17 contract", allow_module_level=True)

from transformers.cache_utils import DynamicCache

from colossalai.shardformer.shard import ShardConfig

pytestmark = pytest.mark.skipif(Version(transformers.__version__) < Version("5.17.0"), reason="HF 5.17 contract")


def make_pair(family, *, head_dim=8, sliding=False, backend="eager", rope=False, dropout=0.0, head="ForCausalLM"):
    prefix = "Qwen2" if family == "qwen2" else "Qwen3"
    config = getattr(transformers, prefix + "Config")(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=dropout,
        attn_implementation=backend,
        use_cache=False,
        **({"head_dim": head_dim} if family == "qwen3" else {}),
        **(
            {"use_sliding_window": True, "sliding_window": 3, "layer_types": ["full_attention", "sliding_attention"]}
            if sliding
            else {}
        ),
        **({"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}} if rope else {}),
    )
    reference = getattr(transformers, prefix + head)(config).eval()
    return reference, copy.deepcopy(reference)


def local_module(family):
    return importlib.import_module("colossalai.shardformer.modeling." + family)


def patch_model(model, family, *, attention=True):
    module = local_module(family)
    config = ShardConfig(enable_tensor_parallelism=False)
    model.forward = MethodType(getattr(module, f"get_{family}_model_forward_for_flash_attn")(config), model)
    if attention:
        for layer in model.layers:
            layer.self_attn.forward = MethodType(
                getattr(module, f"get_{family}_flash_attention_forward")(config), layer.self_attn
            )


def inputs(padding="none"):
    ids = torch.tensor([[3, 7, 9, 11, 13, 15], [4, 8, 10, 12, 14, 16]])
    if padding == "left":
        ids[0, :2] = 0
    elif padding == "right":
        ids[0, -2:] = 0
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    labels = ids.masked_fill(mask == 0, -100)
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions, labels=labels, use_cache=False)


def close(actual, reference, *, atol=1e-5, rtol=1e-3):
    assert torch.isfinite(actual).all()
    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "padding,backend,sliding,rope",
    [
        ("none", "eager", False, False),
        ("left", "eager", False, False),
        ("right", "sdpa", False, False),
        ("none", "eager", True, False),
        ("left", "eager", False, True),
    ],
)
def test_forward_loss_gradients_and_step(family, padding, backend, sliding, rope):
    reference, actual = make_pair(family, backend=backend, sliding=sliding, rope=rope, head_dim=16)
    patch_model(actual.model, family)
    data = inputs(padding)
    ref_out, out = reference(**data), actual(**data)
    close(out.logits[data["attention_mask"].bool()], ref_out.logits[data["attention_mask"].bool()])
    close(out.loss, ref_out.loss)
    ref_out.loss.backward()
    out.loss.backward()
    compared = 0
    for (name, param), (ref_name, ref_param) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        assert param.grad is not None and ref_param.grad is not None, name
        close(param.grad, ref_param.grad, atol=1e-6, rtol=1e-4)
        compared += 1
    assert compared > 0
    torch.optim.SGD(actual.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    for param, ref_param in zip(actual.parameters(), reference.parameters()):
        close(param, ref_param, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("sliding", [False, True])
def test_cache_prefill_and_decode(family, sliding):
    reference, actual = make_pair(family, sliding=sliding, head_dim=16)
    patch_model(actual.model, family)
    ids = inputs()["input_ids"]
    ids = torch.cat((ids, torch.tensor([[17], [18]])), dim=1)
    caches = [DynamicCache(config=m.config) for m in (reference, actual)]
    with torch.no_grad():
        for end in range(4, 8):
            start = 0 if end == 4 else end - 1
            outputs = [
                m(input_ids=ids[:, start:end], attention_mask=torch.ones(2, end), past_key_values=cache, use_cache=True)
                for m, cache in zip((reference, actual), caches)
            ]
            full = reference(input_ids=ids[:, :end], use_cache=False)
            close(outputs[1].logits[:, -1], outputs[0].logits[:, -1])
            close(outputs[1].logits[:, -1], full.logits[:, -1])
            assert outputs[1].past_key_values is caches[1]
            for layer_idx in range(2):
                assert caches[1].get_seq_length(layer_idx) == caches[0].get_seq_length(layer_idx) == end


def test_automatic_cache_and_output_capture(family):
    reference, actual = make_pair(family)
    patch_model(actual.model, family)
    kwargs = dict(input_ids=inputs()["input_ids"], use_cache=True, output_hidden_states=True, output_attentions=True)
    ref_out, out = reference(**kwargs), actual(**kwargs)
    assert isinstance(out.past_key_values, DynamicCache)
    assert len(out.hidden_states) == len(ref_out.hidden_states) == 3
    assert len(out.attentions) == len(ref_out.attentions) == 2
    for a, b in zip(out.hidden_states + out.attentions, ref_out.hidden_states + ref_out.attentions):
        close(a, b)
    tup = actual.model(input_ids=inputs()["input_ids"], use_cache=False, return_dict=False)
    assert isinstance(tup, tuple) and tup[0].shape == (2, 6, 32)


@pytest.mark.parametrize("reentrant", [False, True])
def test_gradient_checkpointing(family, reentrant):
    reference, actual = make_pair(family)
    patch_model(actual.model, family)
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    actual.train()
    reference.train()
    ref_out, out = reference(**inputs()), actual(**inputs())
    close(out.loss, ref_out.loss)
    ref_out.loss.backward()
    out.loss.backward()
    for a, b in zip(actual.parameters(), reference.parameters()):
        assert a.grad is not None and b.grad is not None
        close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def stage(index):
    return SimpleNamespace(
        stage=index,
        num_stages=2,
        is_interleave=False,
        num_model_chunks=1,
        is_first_stage=lambda: index == 0,
        is_last_stage=lambda: index == 1,
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_pipeline_stage_functions_serially(family, checkpoint):
    reference, actual = make_pair(family)
    config = ShardConfig(enable_tensor_parallelism=False)
    forward = getattr(
        getattr(local_module(family), "Qwen" + family[-1] + "PipelineForwards"), f"{family}_model_forward"
    )
    if checkpoint:
        actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        actual.train()
    data = inputs("left")
    data.pop("labels")
    ref_out = reference.model(**data)
    first = forward(actual.model, **data, stage_manager=stage(0), stage_index=[0, 1], shard_config=config)
    out = forward(
        actual.model,
        hidden_states=first["hidden_states"],
        attention_mask=data["attention_mask"],
        use_cache=False,
        stage_manager=stage(1),
        stage_index=[1, 2],
        shard_config=config,
        position_ids=data["position_ids"],
    )
    close(out.last_hidden_state, ref_out.last_hidden_state)
    out.last_hidden_state.square().mean().backward()
    ref_out.last_hidden_state.square().mean().backward()
    for a, b in zip(actual.model.parameters(), reference.model.parameters()):
        assert a.grad is not None and b.grad is not None
        close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def test_attention_training_dropout(family):
    reference, actual = make_pair(family, dropout=0.2)
    patch_model(actual.model, family)
    reference.train()
    actual.train()
    torch.manual_seed(123)
    ref_out = reference(**inputs())
    torch.manual_seed(123)
    out = actual(**inputs())
    close(out.loss, ref_out.loss)


def test_pipeline_sequence_classification_left_padding(family):
    reference, actual = make_pair(family, head="ForSequenceClassification")
    forward = getattr(
        getattr(local_module(family), "Qwen" + family[-1] + "PipelineForwards"),
        f"{family}_for_sequence_classification_forward",
    )
    data = inputs("left")
    data["labels"] = torch.tensor([0, 1])
    manager = SimpleNamespace(is_first_stage=lambda: True, is_last_stage=lambda: True)
    out = forward(
        actual,
        **data,
        stage_manager=manager,
        stage_index=[0, 2],
        shard_config=ShardConfig(enable_tensor_parallelism=False),
    )
    ref_out = reference(**data)
    close(out.logits, ref_out.logits)
    close(out.loss, ref_out.loss)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_precomputed_mask_mapping(family, backend):
    from transformers.masking_utils import create_causal_mask

    reference, actual = make_pair(family, backend=backend)
    patch_model(actual.model, family)
    ids = inputs()["input_ids"]
    mask = create_causal_mask(reference.config, reference.model.embed_tokens(ids), None, None)
    out = actual(input_ids=ids, attention_mask={"full_attention": mask}, use_cache=False)
    ref_out = reference(input_ids=ids, use_cache=False)
    close(out.logits, ref_out.logits)


def test_left_padding_chunked_decode(family):
    reference, actual = make_pair(family)
    patch_model(actual.model, family)
    data = inputs("left")
    cache = None
    for start, end in ((0, 4), (4, 6)):
        out = actual(
            input_ids=data["input_ids"][:, start:end],
            attention_mask=data["attention_mask"][:, :end],
            position_ids=data["position_ids"][:, start:end],
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        ref_out = reference(
            input_ids=data["input_ids"][:, :end],
            attention_mask=data["attention_mask"][:, :end],
            position_ids=data["position_ids"][:, :end],
            use_cache=False,
        )
        close(out.logits[:, -1], ref_out.logits[:, -1])
        assert cache.get_seq_length() == end


def test_pipeline_default_positions_and_checkpoint_selection(family):
    from colossalai.shardformer.shard.grad_ckpt_config import PipelineGradientCheckpointConfig

    reference, actual = make_pair(family)
    actual.train()
    actual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    config = ShardConfig(
        enable_tensor_parallelism=False,
        gradient_checkpoint_config=PipelineGradientCheckpointConfig(num_ckpt_layers_per_stage=[1, 0]),
    )
    calls = [0, 0]

    def count(index):
        def hook(module, args):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_pre_hook(count(i)) for i, layer in enumerate(actual.model.layers)]
    forward = getattr(
        getattr(local_module(family), "Qwen" + family[-1] + "PipelineForwards"), f"{family}_model_forward"
    )
    first = forward(
        actual.model,
        input_ids=inputs()["input_ids"],
        use_cache=False,
        stage_manager=stage(0),
        stage_index=[0, 1],
        shard_config=config,
    )
    out = forward(
        actual.model,
        hidden_states=first["hidden_states"],
        use_cache=False,
        stage_manager=stage(1),
        stage_index=[1, 2],
        shard_config=config,
    )
    ref_out = reference.model(input_ids=inputs()["input_ids"], use_cache=False)
    close(out.last_hidden_state, ref_out.last_hidden_state)
    out.last_hidden_state.square().mean().backward()
    assert calls == [2, 1]  # Only the selected first layer is recomputed, exactly once.
    for handle in handles:
        handle.remove()


def test_lm_wrapper_generation_arguments(family):
    reference, actual = make_pair(family)
    patch_model(actual.model, family)
    actual.forward = MethodType(
        local_module(family).get_lm_forward_with_dist_cross_entropy(ShardConfig(enable_tensor_parallelism=False)),
        actual,
    )
    kwargs = dict(input_ids=inputs()["input_ids"], logits_to_keep=1, use_cache=True)
    ref_out, out = reference(**kwargs), actual(**kwargs)
    assert out.logits.shape == (2, 1, 64)
    close(out.logits, ref_out.logits)
    assert out.past_key_values.get_seq_length() == 6


def test_zbv_policy_without_tensor_parallelism(family):
    reference, _ = make_pair(family)
    module = importlib.import_module("colossalai.shardformer.policies." + family)
    policy = getattr(module, "Qwen" + family[-1] + "Policy")()
    policy.set_model(reference)
    policy.set_shard_config(
        ShardConfig(enable_tensor_parallelism=False, pipeline_stage_manager=SimpleNamespace(use_zbv=True))
    )
    policy.preprocess()
    descriptions = policy.module_policy()
    replacements = descriptions[type(reference.model.layers[0])].sub_module_replacement
    assert any(replacement.suffix == "self_attn.q_proj" for replacement in replacements)


def test_cache_length_helper(family):
    from colossalai.shardformer._utils import get_hf_cache_seq_length

    reference, _ = make_pair(family)
    cache = DynamicCache(config=reference.config)
    assert get_hf_cache_seq_length(None) == get_hf_cache_seq_length(cache) == 0
    kv = torch.randn(2, 2, 4, 8)
    for idx in range(2):
        cache.update(kv, kv, idx)
        assert get_hf_cache_seq_length(cache, idx) == 4
        cache.update(kv[:, :, :1], kv[:, :, :1], idx)
        assert get_hf_cache_seq_length(cache, idx) == 5


def test_flash_mask_contract(family):
    from colossalai.shardformer._utils import get_hf_flash_attention_kwargs, prepare_hf_causal_masks
    from colossalai.shardformer.layer import AttnMaskType

    reference, _ = make_pair(family, sliding=True)
    data = inputs("left")
    masks = prepare_hf_causal_masks(
        reference.config,
        reference.model.embed_tokens(data["input_ids"]),
        data["attention_mask"],
        None,
        data["position_ids"],
        eager=True,
    )
    for layer_type, mask in masks.items():
        kwargs = get_hf_flash_attention_kwargs(mask)
        assert kwargs["attention_mask_type"] == AttnMaskType.CUSTOM
        # This checks the real mask bridge only; it never substitutes a fake GPU kernel.
        close(kwargs["attention_mask"][0, :, 2:], mask[0, :, 2:])
        assert (kwargs["attention_mask"][0, :, :2] == 0).all()
        assert mask[1, 0, 2, 3] < -1e20  # Future token stays masked.
        if layer_type == "sliding_attention":
            assert mask[1, 0, 5, 1] < -1e20  # Token outside the window stays masked.


def test_shardformer_cpu_policy_integration(family):
    from colossalai.shardformer import ShardFormer

    reference, actual = make_pair(family)
    actual, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(actual)
    out, ref_out = actual(**inputs()), reference(**inputs())
    close(out.logits, ref_out.logits)
    close(out.loss, ref_out.loss)
    out.loss.backward()
    ref_out.loss.backward()
    for a, b in zip(actual.parameters(), reference.parameters()):
        assert a.grad is not None and b.grad is not None
        close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
