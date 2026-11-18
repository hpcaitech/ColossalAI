"""2D2 objective integration, not full-model or distributed L1/L2 acceptance."""

import copy
import os

import pytest
import torch
import transformers
from huggingface_hub.errors import LocalEntryNotFoundError

from colossalai.nn.diffusion import prepare_block_diffusion_inputs, sample_masked_tokens
from colossalai.nn.loss.diffusion import masked_diffusion_loss, masked_token_cross_entropy
from colossalai.shardformer import ShardConfig, ShardFormer
from tests.kit.remote_models import build_remote_model, load_remote_classes, tiny_config

from .reference import llada_sequence_loss, official_function, official_masked_ce

pytestmark = pytest.mark.skipif(transformers.__version__ != "5.17.0", reason="Pinned Transformers 5.17.0 adapters")


@pytest.fixture(scope="module", autouse=True)
def cached_models():
    """Require pinned local code in acceptance runs; never download from tests."""
    for key in ("llada", "llada21", "llada22"):
        try:
            load_remote_classes(key)
        except LocalEntryNotFoundError:
            if os.environ.get("COLOSSALAI_REQUIRE_REMOTE_MODELS") == "1":
                pytest.fail(f"Missing pinned local snapshot: {key}")
            pytest.skip(f"Missing pinned local snapshot: {key}")


def model_pair(key):
    """Create a native official reference and an independently adapted candidate."""
    torch.manual_seed(1984)
    config = tiny_config(key)
    reference = build_remote_model(key, adapted=False, config=copy.deepcopy(config)).train()
    candidate = build_remote_model(key, config=copy.deepcopy(config)).train()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    candidate, shared = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(candidate)
    assert shared == []
    return reference, candidate


def compare_parameters(reference, candidate, *, gradients=False):
    left, right = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert left.keys() == right.keys()
    count = 0
    maximum = 0.0
    for name, param in left.items():
        expected = param.grad if gradients else param
        actual = right[name].grad if gradients else right[name]
        assert (expected is None) == (actual is None), name
        if expected is None:
            continue
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=name)
        maximum = max(maximum, float((actual - expected).detach().abs().max()))
        count += 1
    assert count > 0
    return count, maximum


@pytest.mark.parametrize("key", ["llada", "llada21", "llada22"])
@pytest.mark.parametrize("sft", [False, True])
def test_tiny_full_attention_three_training_steps(key, sft, record_property):
    reference, candidate = model_pair(key)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0) for model in (reference, candidate)]
    ids = torch.randint(5, 128, (2, 12), generator=torch.Generator().manual_seed(2026))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, -2:] = False
    target = valid.clone()
    if sft:
        target[0, :3] = False
        target[1, :5] = False
    initial = {n: p.detach().clone() for n, p in candidate.named_parameters()}
    history = []
    for step in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        batch = sample_masked_tokens(
            ids,
            2,
            target_mask=target,
            valid_mask=valid,
            timesteps=torch.tensor([0.3 + step * 0.1, 0.8]),
            generator=torch.Generator().manual_seed(55 + step),
        )
        kwargs = dict(input_ids=batch.input_ids, attention_mask=valid.long(), use_cache=False)
        expected = reference(**kwargs).logits
        actual = candidate(**kwargs).logits
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        expected_loss = llada_sequence_loss(expected, batch.labels, batch.mask_probability, target).mean()
        actual_loss = masked_diffusion_loss(actual, batch.labels, batch.mask_probability, target)
        torch.testing.assert_close(actual_loss, expected_loss)
        expected_loss.backward()
        actual_loss.backward()
        count, error = compare_parameters(reference, candidate, gradients=True)
        for optimizer in optimizers:
            optimizer.step()
        compare_parameters(reference, candidate)
        history.append({"loss": float(actual_loss.detach()), "gradient_tensors": count, "max_gradient_error": error})
    assert any(not torch.equal(initial[n], p) for n, p in candidate.named_parameters())
    record_property("training_steps", history)
    record_property("scope", "CPU FP32 tiny, full attention, same-position inverse-probability objective")


def sequential_block_logits(model, batch, block_size):
    """Independent oracle: one forward per block with clean preceding context.

    This uses the ordinary block-causal prefix layout, not the doubled-sequence
    mask helper, so a wrong noisy/clean cross-attention edge cannot be hidden by
    sharing the production mask in both paths.
    """
    length = batch.input_ids.shape[1]
    outputs = []
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        ids = torch.cat((batch.clean_input_ids[:, :start], batch.input_ids[:, start:end]), dim=1)
        positions = torch.arange(end)
        blocks = positions // block_size
        allowed = blocks[:, None] >= blocks[None, :]
        bias = torch.zeros(end, end).masked_fill(~allowed, float("-inf"))[None, None]
        logits = model(
            input_ids=ids,
            position_ids=positions[None].expand(ids.shape[0], -1),
            attention_mask=bias.expand(ids.shape[0], -1, -1, -1),
            use_cache=False,
        ).logits
        outputs.append(logits[:, start:end])
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("key", ["llada21", "llada22"])
@pytest.mark.parametrize("objective", ["llada", "dfactory"])
@pytest.mark.parametrize("layout", ["official_single_pass", "sequential"])
def test_tiny_block_training(key, objective, layout, record_property):
    reference, candidate = model_pair(key)
    ids = torch.randint(5, 128, (2, 8), generator=torch.Generator().manual_seed(801))
    target = torch.ones_like(ids, dtype=torch.bool)
    target[:, :2] = False
    batch = sample_masked_tokens(
        ids,
        2,
        target_mask=target,
        timesteps=torch.tensor([0.5, 0.8]),
        generator=torch.Generator().manual_seed(63),
    )
    if layout == "sequential":
        expected = sequential_block_logits(reference, batch, 4)
    else:
        allowed = official_function("block_diffusion_mask")(
            None, None, torch.arange(16)[:, None], torch.arange(16)[None, :], block_size=4, n=8
        )
        bias = torch.zeros(16, 16).masked_fill(~allowed, float("-inf"))[None, None]
        expected = reference(
            input_ids=torch.cat((batch.input_ids, ids), dim=1),
            position_ids=torch.arange(8).repeat(2)[None].expand(2, -1),
            attention_mask=bias.expand(2, 1, 16, 16),
            use_cache=False,
        ).logits[:, :8]
    inputs = prepare_block_diffusion_inputs(batch, 4)
    actual = candidate(**inputs, use_cache=False).logits[:, :8]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    if objective == "llada":
        # Generic 1/p objective combined with the block layout. This is not a
        # claim that LLaDA2.1/2.2's token-editing training objective is reproduced.
        expected_loss = llada_sequence_loss(expected, batch.labels, batch.mask_probability, target).mean()
        actual_loss = masked_diffusion_loss(actual, batch.labels, batch.mask_probability, target)
    else:
        expected_loss = official_masked_ce(expected, batch.labels)
        actual_loss = masked_token_cross_entropy(actual, batch.labels)
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    count, error = compare_parameters(reference, candidate, gradients=True)
    # AdamW magnifies tiny near-zero gradient differences between GEMM layouts.
    # Keep the independent sequential oracle for logits/gradients and an SGD
    # update; validate default AdamW against the official single-pass layout.
    # See scripts/phase2/2d2/probe_block_roundoff.py for the measured baseline.
    optimizer_class = torch.optim.SGD if layout == "sequential" else torch.optim.AdamW
    for model in (reference, candidate):
        optimizer_class(model.parameters(), lr=1e-3, weight_decay=0).step()
    compare_parameters(reference, candidate)
    record_property("optimizer", optimizer_class.__name__)
    record_property("reference_layout", layout)
    record_property("gradient_tensors", count)
    record_property("max_gradient_error", error)
    record_property("max_logit_error", float((actual - expected).detach().abs().max()))


@pytest.mark.parametrize("key", ["llada21", "llada22"])
def test_real_tiny_attention_cannot_read_current_clean_answers(key):
    _, model = model_pair(key)
    model.eval()
    ids = torch.randint(5, 128, (1, 8), generator=torch.Generator().manual_seed(451))
    batch = sample_masked_tokens(ids, 2, timesteps=torch.ones(1))
    inputs = prepare_block_diffusion_inputs(batch, 4)
    with torch.no_grad():
        expected = model(**inputs, use_cache=False).logits[:, :8]
        changed = {name: value.clone() for name, value in inputs.items()}
        # Change the entire clean current/future suffix. Noisy block 0 must not
        # observe it through either direct or multi-layer attention.
        changed["input_ids"][:, 8:] = (changed["input_ids"][:, 8:] + 17) % 128
        actual = model(**changed, use_cache=False).logits[:, :8]
    torch.testing.assert_close(actual[:, :4], expected[:, :4], atol=0, rtol=0)
    # Later block legitimately conditions on clean block 0: the edge is useful.
    assert not torch.equal(actual[:, 4:8], expected[:, 4:8])
