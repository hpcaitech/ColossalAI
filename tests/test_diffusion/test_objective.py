import pytest
import torch

from colossalai.nn.diffusion import block_diffusion_attention_mask, prepare_block_diffusion_inputs, sample_masked_tokens
from colossalai.nn.loss.diffusion import masked_diffusion_loss, masked_token_cross_entropy

from .reference import llada_sequence_loss, official_function, official_masked_ce


def example():
    ids = torch.arange(24).reshape(3, 8).remainder(13)
    valid = torch.tensor([[1] * 8, [1] * 6 + [0] * 2, [1] * 8], dtype=torch.bool)
    target = valid.clone()
    target[0, :2] = False
    target[1, :3] = False
    target[2] = False
    batch = sample_masked_tokens(
        ids,
        15,
        target_mask=target,
        valid_mask=valid,
        timesteps=torch.tensor([0.2, 0.8, 0.5]),
        mask_random=torch.arange(24).reshape(3, 8).remainder(7).float() / 7,
    )
    return ids, batch


def test_sampling_replay_and_llada_schedule():
    ids = torch.arange(35).reshape(5, 7)
    generator = torch.Generator().manual_seed(19)
    state = generator.get_state()
    actual = sample_masked_tokens(ids, 40, generator=generator)
    generator.set_state(state)
    # Official LLaDA GUIDELINES.md forward_process, independently replayed.
    times = torch.rand(5, generator=generator)
    probability = ((1 - 1e-3) * times + 1e-3)[:, None].repeat(1, 7)
    selected = torch.rand((5, 7), generator=generator) < probability
    assert torch.equal(actual.masked_indices, selected)
    torch.testing.assert_close(actual.mask_probability, probability, atol=0, rtol=0)
    assert torch.equal(actual.input_ids, torch.where(selected, 40, ids))
    generator.set_state(state)
    replayed = sample_masked_tokens(ids, 40, generator=generator)
    assert torch.equal(replayed.input_ids, actual.input_ids)
    assert torch.equal(ids, torch.arange(35).reshape(5, 7))


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_dfactory_noise_matches_official(seed):
    ids = torch.arange(16)
    target = torch.arange(16) >= 5
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected = official_function("sft_noise_transition")(ids, (0.3, 0.8), target, 21)
    actual = sample_masked_tokens(
        ids[None],
        21,
        target_mask=target[None],
        probability_range=(0.3, 0.8),
        generator=torch.Generator().manual_seed(seed),
    )
    assert torch.equal(actual.input_ids[0], expected)


def test_prompt_padding_and_original_mask_token_are_explicit():
    ids, batch = example()
    assert torch.equal(batch.input_ids[~batch.target_mask], ids[~batch.target_mask])
    assert (batch.labels[~batch.masked_indices] == -100).all()
    assert torch.equal(batch.labels[batch.masked_indices], ids[batch.masked_indices])
    original = torch.tensor([[2, 3, 4]])
    draw = sample_masked_tokens(
        original,
        2,
        timesteps=torch.zeros(1),
        mask_random=torch.full((1, 3), 0.9),
    )
    assert draw.input_ids[0, 0] == 2
    assert not draw.masked_indices.any()
    assert (draw.labels == -100).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_weighted_loss_and_gradients_against_formula(dtype, reduction):
    _, batch = example()
    logits = torch.randn(3, 8, 16, generator=torch.Generator().manual_seed(81), dtype=dtype).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    actual = masked_diffusion_loss(logits, batch.labels, batch.mask_probability, batch.target_mask, reduction=reduction)
    expected = llada_sequence_loss(reference, batch.labels, batch.mask_probability, batch.target_mask)
    if reduction != "none":
        expected = expected.mean() if reduction == "mean" else expected.sum()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert (logits.grad[~batch.masked_indices] == 0).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unweighted_loss_matches_dfactory_not_llada(dtype):
    _, batch = example()
    logits = torch.randn(3, 8, 16, generator=torch.Generator().manual_seed(25), dtype=dtype).requires_grad_()
    reference = logits.detach().float().requires_grad_()
    actual = masked_token_cross_entropy(logits, batch.labels)
    expected = official_masked_ce(reference, batch.labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad.float(), reference.grad.to(dtype).float())
    weighted = masked_diffusion_loss(logits, batch.labels, batch.mask_probability, batch.target_mask)
    assert not torch.isclose(weighted, actual)
    assert torch.isclose(
        masked_token_cross_entropy(logits, batch.labels, reduction="sum"), actual * batch.masked_indices.sum()
    )


@pytest.mark.parametrize("empty_targets", [False, True])
@pytest.mark.parametrize("objective", ["llada", "masked_ce"])
def test_no_sampled_tokens_yields_differentiable_zero(empty_targets, objective):
    ids = torch.arange(8).reshape(2, 4)
    batch = sample_masked_tokens(
        ids,
        9,
        target_mask=torch.full_like(ids, not empty_targets, dtype=torch.bool),
        timesteps=torch.zeros(2),
        mask_random=torch.full((2, 4), 0.9),
    )
    logits = torch.randn(2, 4, 10, requires_grad=True)
    loss = (
        masked_diffusion_loss(logits, batch.labels, batch.mask_probability, batch.target_mask)
        if objective == "llada"
        else masked_token_cross_entropy(logits, batch.labels)
    )
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_sequence_normalization_and_no_label_shift():
    logits = torch.zeros(2, 4, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, -100, -100, -100], [2, 3, -100, -100]])
    target = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    prob = torch.tensor([[0.25] * 4, [0.5] * 4])
    loss = masked_diffusion_loss(logits, labels, prob, target)
    # Row 0 contributes log(5), row 1 contributes 2*log(5); average is 1.5*log(5).
    torch.testing.assert_close(loss, 1.5 * torch.tensor(5.0, dtype=torch.float64).log())
    loss.backward()
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[1, 1, 3] < 0
    assert (logits.grad[labels == -100] == 0).all()


@pytest.mark.parametrize("objective", ["llada", "masked_ce"])
def test_unequal_microbatch_accumulation_uses_global_denominator(objective):
    _, batch = example()
    full = torch.randn(3, 8, 16, dtype=torch.float64, requires_grad=True)
    micro = full.detach().clone().requires_grad_()
    if objective == "llada":
        masked_diffusion_loss(full, batch.labels, batch.mask_probability, batch.target_mask).backward()
        denom = len(batch.labels)
    else:
        masked_token_cross_entropy(full, batch.labels).backward()
        denom = batch.masked_indices.sum()
    for selection in (slice(0, 1), slice(1, 3)):
        if objective == "llada":
            loss = masked_diffusion_loss(
                micro[selection],
                batch.labels[selection],
                batch.mask_probability[selection],
                batch.target_mask[selection],
                reduction="sum",
            )
        else:
            loss = masked_token_cross_entropy(micro[selection], batch.labels[selection], reduction="sum")
        (loss / denom).backward()
    torch.testing.assert_close(full.grad, micro.grad)


@pytest.mark.parametrize("length", [1, 7, 8])
@pytest.mark.parametrize("block_size", [1, 2, 4, 16])
def test_block_mask_matches_official_function(length, block_size):
    expected = official_function("block_diffusion_mask")(
        None,
        None,
        torch.arange(2 * length)[:, None],
        torch.arange(2 * length)[None, :],
        block_size=block_size,
        n=length,
    )
    actual = block_diffusion_attention_mask(length, block_size)
    assert torch.equal(actual[0, 0], expected)
    assert actual.dtype == torch.bool


def test_block_mask_prevents_answer_leakage_and_handles_padding():
    valid = torch.tensor([[1] * 8, [0, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    mask = block_diffusion_attention_mask(8, 4, valid_mask=valid)[:, 0]
    # Noisy block 1 sees noisy block 1 and clean block 0; no current clean answers.
    assert mask[0, 4:8, 4:8].all()
    assert mask[0, 4:8, 8:12].all()
    assert not mask[0, 4:8, :4].any()
    assert not mask[0, 4:8, 12:16].any()
    # Clean tokens never see noisy tokens; block causality is not token causality.
    assert not mask[0, 8:, :8].any()
    assert mask[0, 8, 11]
    assert not mask[0, 8, 12]
    doubled = valid[1].repeat(2)
    assert not mask[1][doubled][:, ~doubled].any()
    for index in (~doubled).nonzero().flatten():
        assert mask[1, index].sum() == 1
        assert mask[1, index, index]
    assert mask.any(-1).all()


def test_block_inputs_repeat_positions_and_supervise_noisy_half_only():
    _, batch = example()
    inputs = prepare_block_diffusion_inputs(batch, 3, dtype=torch.bfloat16)
    assert torch.equal(inputs["input_ids"][:, :8], batch.input_ids)
    assert torch.equal(inputs["input_ids"][:, 8:], batch.clean_input_ids)
    assert torch.equal(inputs["position_ids"], torch.arange(8).repeat(2)[None].expand(3, -1))
    assert inputs["attention_mask"].shape == (3, 1, 16, 16)
    assert inputs["attention_mask"].dtype == torch.bfloat16
    assert not torch.isnan(inputs["attention_mask"]).any()
    assert set(inputs) == {"input_ids", "position_ids", "attention_mask"}


@pytest.mark.parametrize(
    "change",
    [
        {"probability_range": (0.0, 1.0)},
        {"probability_range": (0.8, 0.3)},
        {"probability_range": (0.1, float("nan"))},
        {"mask_token_id": -1},
        {"timesteps": torch.tensor([float("nan")])},
        {"timesteps": torch.tensor([1.1])},
        {"timesteps": torch.tensor([0])},
        {"timesteps": torch.zeros(1, 1)},
        {"mask_random": torch.ones(1, 3)},
        {"mask_random": torch.zeros(1, 2)},
        {"target_mask": torch.ones(1, 3)},
        {"valid_mask": torch.ones(1, 3)},
        {"valid_mask": torch.zeros(1, 3, dtype=torch.bool), "target_mask": torch.ones(1, 3, dtype=torch.bool)},
    ],
)
def test_invalid_corruption_inputs_are_rejected(change):
    args = {"mask_token_id": 5, **change}
    with pytest.raises(ValueError):
        sample_masked_tokens(torch.tensor([[1, 2, 3]]), **args)


@pytest.mark.parametrize("fault", ["prob_zero", "prob_nan", "wrong_target", "label_oob", "reduction", "shape"])
def test_invalid_loss_metadata_is_rejected(fault):
    logits = torch.randn(1, 2, 5)
    labels = torch.tensor([[1, -100]])
    prob = torch.full((1, 2), 0.5)
    target = torch.ones(1, 2, dtype=torch.bool)
    reduction = "mean"
    if fault == "prob_zero":
        prob[0, 0] = 0
    elif fault == "prob_nan":
        prob[0, 1] = float("nan")
    elif fault == "wrong_target":
        target[0, 0] = False
    elif fault == "label_oob":
        labels[0, 0] = 5
    elif fault == "reduction":
        reduction = "token_mean"
    elif fault == "shape":
        prob = prob[:, :1]
    with pytest.raises(ValueError):
        masked_diffusion_loss(logits, labels, prob, target, reduction=reduction)


@pytest.mark.parametrize("length,block_size", [(0, 4), (8, 0), (8, -1), (8, 2.5)])
def test_invalid_block_sizes_are_rejected(length, block_size):
    with pytest.raises(ValueError):
        block_diffusion_attention_mask(length, block_size)
