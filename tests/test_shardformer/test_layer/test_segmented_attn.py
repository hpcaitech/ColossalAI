import pytest
import torch
import torch.nn.functional as F

from colossalai.shardformer.layer import SegmentMaskSpec, SegmentRule, build_segment_mask, naive_segment_attention
from colossalai.shardformer.layer.attn import invert_mask
from tests.kit.numerics import assert_aligned, run_fwd_bwd

BATCH = 2
HEADS = 2
HEAD_DIM = 16

# The shape of a world-model training step: three autoregressive tokens that only see their own prefix, followed by
# five diffusion tokens that see the whole sequence. Both parts see each other, so the whole row is visible from the
# diffusion part but the autoregressive part stays triangular.
EIGHT_TOKEN_LENGTHS = [3, 5]
EIGHT_TOKEN_RULES = [
    [SegmentRule.CAUSAL, SegmentRule.FULL],
    [SegmentRule.FULL, SegmentRule.FULL],
]
EIGHT_TOKEN_MASK = [
    [1, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def _eight_token_spec():
    return SegmentMaskSpec(EIGHT_TOKEN_LENGTHS, EIGHT_TOKEN_RULES)


def _random_inputs(seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (BATCH, HEADS, sum(EIGHT_TOKEN_LENGTHS), HEAD_DIM)
    return {name: torch.randn(*shape, generator=generator, dtype=torch.float64) for name in ("q", "k", "v")}


def sdpa_attention(q, k, v, attention_mask):
    """An implementation that shares no code with the module under test."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(0).unsqueeze(0))


def test_eight_token_mask_is_the_hand_checked_pattern():
    mask = build_segment_mask(_eight_token_spec())
    expected = torch.tensor(EIGHT_TOKEN_MASK, dtype=torch.bool)

    assert mask.dtype is torch.bool
    assert mask.shape == (8, 8)
    assert torch.equal(mask, expected), f"got\n{mask.int()}\nexpected\n{expected.int()}"


def test_blocks_do_not_leak_across_segments():
    spec = SegmentMaskSpec(
        EIGHT_TOKEN_LENGTHS,
        [[SegmentRule.CAUSAL, SegmentRule.NONE], [SegmentRule.FULL, SegmentRule.FULL]],
    )
    mask = build_segment_mask(spec)

    assert not mask[:3, 3:].any(), "the causal part must not see the bidirectional part"
    assert mask[3:, :3].all(), "the bidirectional part must see the causal part"


@pytest.mark.parametrize(
    "segment_lengths, rules, message",
    [
        ([], [], "at least one segment"),
        ([2, 0], [[SegmentRule.FULL, SegmentRule.FULL], [SegmentRule.FULL, SegmentRule.FULL]], "must be positive"),
        ([2, 2], [[SegmentRule.FULL, SegmentRule.FULL]], "square matrix"),
        (
            [2, 2],
            [[SegmentRule.NONE, SegmentRule.FULL], [SegmentRule.FULL, SegmentRule.FULL]],
            "no visible key",
        ),
    ],
)
def test_invalid_spec_is_rejected(segment_lengths, rules, message):
    with pytest.raises(ValueError, match=message):
        SegmentMaskSpec(segment_lengths, rules)


def test_spec_does_not_alias_its_inputs():
    segment_lengths = list(EIGHT_TOKEN_LENGTHS)
    rules = [list(row) for row in EIGHT_TOKEN_RULES]
    spec = SegmentMaskSpec(segment_lengths, rules)

    segment_lengths[0] = 99
    rules[0][0] = SegmentRule.FULL

    assert spec.segment_lengths == [3, 5]
    assert spec.rules[0][0] is SegmentRule.CAUSAL


def test_offsets_and_total_length_describe_the_packed_sequence():
    spec = _eight_token_spec()

    assert spec.total_length == 8
    assert spec.offsets == [0, 3]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_naive_attention_matches_sdpa_within_baseline(dtype):
    inputs = dict(_random_inputs(), attention_mask=build_segment_mask(_eight_token_spec()))

    truth = run_fwd_bwd(sdpa_attention, inputs, torch.float64)
    reference = run_fwd_bwd(sdpa_attention, inputs, dtype)
    test = run_fwd_bwd(naive_segment_attention, inputs, dtype)

    report = assert_aligned(test, reference, truth)
    assert {entry.name for entry in report.entries} == {"out", "grad/q", "grad/k", "grad/v"}


def test_masked_positions_are_removed_before_the_softmax():
    spec = _eight_token_spec()
    q, k, v = (tensor.float() for tensor in _random_inputs().values())
    _, scores = naive_segment_attention(q, k, v, spec, return_scores=True)
    mask = build_segment_mask(spec)

    assert torch.isneginf(scores[..., ~mask]).all(), "every masked position must be -inf before the softmax"
    assert torch.isfinite(scores[..., mask]).all(), "no visible position may be masked"


def test_boolean_mask_must_follow_the_correct_convention():
    q, k, v = (tensor.float() for tensor in _random_inputs().values())
    integer_mask = build_segment_mask(_eight_token_spec()).to(torch.int32)

    with pytest.raises(TypeError, match="boolean mask"):
        naive_segment_attention(q, k, v, integer_mask)

    with pytest.raises(ValueError, match="broadcastable"):
        naive_segment_attention(q, k, v, torch.ones(1, 1, 3, 3, dtype=torch.bool))


def test_additive_mask_is_accepted_by_the_custom_mask_entry():
    """The bridge to ColoAttention: a boolean segment mask becomes a CUSTOM float mask without all-masked rows."""
    boolean_mask = build_segment_mask(_eight_token_spec())
    additive_mask = invert_mask(boolean_mask.unsqueeze(0).unsqueeze(0).to(torch.bfloat16))

    assert torch.is_floating_point(additive_mask)
    assert additive_mask.shape == (1, 1, 8, 8)
    assert torch.equal(additive_mask[0, 0] != 0, ~boolean_mask)
    # ColoAttention.attention asserts that no query row is masked out entirely.
    assert not torch.all(additive_mask != 0, dim=-1).any()


def test_float64_softmax_keeps_full_precision():
    """A float64 reference must not be rounded through float32 on the way through the softmax."""
    q, k, v = (tensor.double() for tensor in _random_inputs().values())
    mask = build_segment_mask(_eight_token_spec())
    scores = torch.matmul(q, k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    expected = torch.matmul(torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1), v)

    actual = naive_segment_attention(q, k, v, mask)

    assert torch.equal(actual, expected), "a float64 call must run its softmax in float64 as well"


def test_fully_masked_query_row_is_rejected():
    q, k, v = (tensor.float() for tensor in _random_inputs().values())
    mask = build_segment_mask(_eight_token_spec()).clone()
    mask[0, :] = False

    with pytest.raises(ValueError, match="without a visible key"):
        naive_segment_attention(q, k, v, mask)


def test_three_dimensional_mask_is_broadcast_over_heads():
    q, k, v = (tensor.float() for tensor in _random_inputs().values())
    sequence_length = sum(EIGHT_TOKEN_LENGTHS)
    mask = build_segment_mask(_eight_token_spec()).expand(BATCH, sequence_length, sequence_length)

    assert torch.equal(
        naive_segment_attention(q, k, v, mask),
        naive_segment_attention(q, k, v, mask.unsqueeze(1)),
    )
