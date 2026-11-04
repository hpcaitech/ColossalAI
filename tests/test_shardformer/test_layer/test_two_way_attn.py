import pytest
import torch

from colossalai.shardformer.layer import (
    AttnMaskType,
    ColoAttention,
    SegmentMaskSpec,
    SegmentRule,
    TwoWayPackedMeta,
    build_segment_mask,
    build_two_way_kv_gather_index,
    naive_segment_attention,
    naive_two_way_flat_attention,
    npu_two_way_flat_attention,
    two_way_flat_attention,
    validate_two_way_varlen_limits,
)

HEADS = 2
HEAD_DIM = 16

# The hand-checked example of the design note: three autoregressive tokens followed by five diffusion tokens.
EIGHT_TOKEN_LENGTHS = ([3], [5])
EIGHT_TOKEN_MASK = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def _reference_mask(meta):
    """The dense mask of the whole packed sequence, built from per-segment rules."""
    ar_indices = [index for index, length in enumerate(meta.ar_lens) if length > 0]
    dm_indices = [index for index, length in enumerate(meta.dm_lens) if length > 0]
    segments = [meta.ar_lens[index] for index in ar_indices] + [meta.dm_lens[index] for index in dm_indices]
    rules = []
    for row, sample in enumerate(ar_indices):
        row_rules = [SegmentRule.NONE] * len(segments)
        row_rules[row] = SegmentRule.CAUSAL
        rules.append(row_rules)
    for row, sample in enumerate(dm_indices):
        row_rules = [SegmentRule.NONE] * len(segments)
        if sample in ar_indices:
            row_rules[ar_indices.index(sample)] = SegmentRule.FULL
        row_rules[len(ar_indices) + row] = SegmentRule.FULL
        rules.append(row_rules)
    return build_segment_mask(SegmentMaskSpec(segments, rules))


def _random_packed_tensors(meta, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape_ar = (meta.total_ar_tokens, HEADS, HEAD_DIM)
    shape_dm = (meta.total_dm_tokens, HEADS, HEAD_DIM)
    ar = tuple(torch.randn(*shape_ar, generator=generator, dtype=dtype) for _ in range(3))
    dm = tuple(torch.randn(*shape_dm, generator=generator, dtype=dtype) for _ in range(3))
    return ar, dm


def _reference_output(meta, q_ar, k_ar, v_ar, q_dm, k_dm, v_dm):
    """The result of one attention call over the whole packed sequence under the dense reference mask."""
    q = torch.cat((q_ar, q_dm), dim=0)
    k = torch.cat((k_ar, k_dm), dim=0)
    v = torch.cat((v_ar, v_dm), dim=0)
    mask = _reference_mask(meta)
    output = naive_segment_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), mask
    )
    return output[0].transpose(0, 1)


def test_eight_token_mask_is_the_hand_checked_pattern():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    expected = torch.tensor(EIGHT_TOKEN_MASK, dtype=torch.bool)

    assert torch.equal(_reference_mask(meta), expected)


def test_eight_token_naive_output_matches_the_segmented_reference():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=1)

    out_ar, out_dm = naive_two_way_flat_attention(*ar, *dm, meta)
    expected = _reference_output(meta, *ar, *dm)

    torch.testing.assert_close(torch.cat((out_ar, out_dm), dim=0), expected, rtol=1e-12, atol=1e-12)


def test_autoregressive_output_cannot_see_the_diffusion_block():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=2)
    q_ar, k_ar, v_ar = ar
    q_dm, k_dm, v_dm = dm

    baseline, _ = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)
    changed, _ = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm + 10.0, v_dm + 10.0, meta)

    assert torch.equal(baseline, changed), "changing DM keys/values must not move an AR output"


def test_diffusion_output_sees_the_autoregressive_block():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=3)
    q_ar, k_ar, v_ar = ar
    q_dm, k_dm, v_dm = dm

    _, baseline = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)
    _, changed = naive_two_way_flat_attention(q_ar, k_ar + 1.0, v_ar + 1.0, q_dm, k_dm, v_dm, meta)

    assert not torch.allclose(baseline, changed), "the DM output must depend on the AR keys/values"


def test_multiple_samples_match_the_segmented_reference_and_ignore_other_samples():
    meta = TwoWayPackedMeta([3, 2, 0, 5], [6, 0, 4, 1])
    ar, dm = _random_packed_tensors(meta, seed=4)

    out_ar, out_dm = naive_two_way_flat_attention(*ar, *dm, meta)
    expected = _reference_output(meta, *ar, *dm)

    assert out_ar.shape == (meta.total_ar_tokens, HEADS, HEAD_DIM)
    assert out_dm.shape == (meta.total_dm_tokens, HEADS, HEAD_DIM)
    torch.testing.assert_close(torch.cat((out_ar, out_dm), dim=0), expected, rtol=1e-12, atol=1e-12)


def test_segmented_reference_shares_one_key_head_per_query_group():
    spec = SegmentMaskSpec([3, 2], [[SegmentRule.CAUSAL, SegmentRule.NONE], [SegmentRule.FULL, SegmentRule.FULL]])
    generator = torch.Generator().manual_seed(20)
    q = torch.randn(1, 4, 5, HEAD_DIM, generator=generator, dtype=torch.float64)
    k = torch.randn(1, 2, 5, HEAD_DIM, generator=generator, dtype=torch.float64)
    v = torch.randn(1, 2, 5, HEAD_DIM, generator=generator, dtype=torch.float64)

    out = naive_segment_attention(q, k, v, spec)
    grouped = torch.cat(
        [
            naive_segment_attention(
                q[:, head : head + 1], k[:, head // 2 : head // 2 + 1], v[:, head // 2 : head // 2 + 1], spec
            )
            for head in range(4)
        ],
        dim=1,
    )

    torch.testing.assert_close(out, grouped, rtol=1e-12, atol=1e-12)


def test_segmented_reference_rejects_a_partial_query_group():
    q = torch.randn(1, 3, 4, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(1, 2, 4, HEAD_DIM, dtype=torch.float64)
    v = torch.randn(1, 2, 4, HEAD_DIM, dtype=torch.float64)

    with pytest.raises(ValueError, match="positive multiple"):
        naive_segment_attention(q, k, v)


def test_packed_grouped_query_attention_matches_the_expanded_reference():
    meta = TwoWayPackedMeta([3, 2, 0, 5], [6, 0, 4, 1])
    num_q_heads, num_kv_heads = 4, 2
    generator = torch.Generator().manual_seed(21)
    q_ar = torch.randn(meta.total_ar_tokens, num_q_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    q_dm = torch.randn(meta.total_dm_tokens, num_q_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    k_ar = torch.randn(meta.total_ar_tokens, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    k_dm = torch.randn(meta.total_dm_tokens, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    v_ar = torch.randn(meta.total_ar_tokens, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)
    v_dm = torch.randn(meta.total_dm_tokens, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float64)

    out_ar, out_dm = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)
    expected = _reference_output(
        meta,
        q_ar,
        k_ar.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
        v_ar.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
        q_dm,
        k_dm.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
        v_dm.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
    )

    torch.testing.assert_close(torch.cat((out_ar, out_dm), dim=0), expected, rtol=1e-12, atol=1e-12)


def test_packed_inputs_reject_a_partial_query_group():
    meta = TwoWayPackedMeta([3], [5])
    ar, dm = _random_packed_tensors(meta, seed=22)
    _, k_ar, v_ar = ar
    _, k_dm, v_dm = dm
    generator = torch.Generator().manual_seed(22)
    q_ar = torch.randn(meta.total_ar_tokens, 3, HEAD_DIM, generator=generator, dtype=torch.float64)
    q_dm = torch.randn(meta.total_dm_tokens, 3, HEAD_DIM, generator=generator, dtype=torch.float64)

    with pytest.raises(ValueError, match="multiple of the number of key/value heads"):
        naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)


def test_zero_length_samples_are_dropped_from_the_kernel_calls():
    meta = TwoWayPackedMeta([0, 3, 0], [5, 0, 2])
    ar, dm = _random_packed_tensors(meta, seed=5)

    out_ar, out_dm = naive_two_way_flat_attention(*ar, *dm, meta)

    assert out_ar.shape == (3, HEADS, HEAD_DIM)
    assert out_dm.shape == (7, HEADS, HEAD_DIM)
    torch.testing.assert_close(
        torch.cat((out_ar, out_dm), dim=0), _reference_output(meta, *ar, *dm), rtol=1e-12, atol=1e-12
    )


def test_empty_batch_returns_empty_outputs():
    meta = TwoWayPackedMeta([], [])
    q_ar, k_ar, v_ar = (torch.zeros(0, HEADS, HEAD_DIM) for _ in range(3))
    q_dm, k_dm, v_dm = (torch.zeros(0, HEADS, HEAD_DIM) for _ in range(3))

    out_ar, out_dm = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)

    assert out_ar.shape == (0, HEADS, HEAD_DIM)
    assert out_dm.shape == (0, HEADS, HEAD_DIM)


def test_naive_path_backpropagates_to_both_towers():
    meta = TwoWayPackedMeta([3, 2], [5, 4])
    ar, dm = _random_packed_tensors(meta, seed=6)
    inputs = [tensor.clone().requires_grad_(True) for tensor in (*ar, *dm)]

    out_ar, out_dm = naive_two_way_flat_attention(*inputs, meta)
    (out_ar.sum() + out_dm.sum()).backward()

    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_default_backend_is_the_naive_reference():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=7)

    default = two_way_flat_attention(*ar, *dm, meta)
    explicit = naive_two_way_flat_attention(*ar, *dm, meta)

    assert all(torch.equal(left, right) for left, right in zip(default, explicit))


def test_dispatcher_forwards_the_normalized_ar_keys(monkeypatch):
    meta = TwoWayPackedMeta([3, 2], [6, 4])
    ar, dm = _random_packed_tensors(meta, seed=9)
    k_ar_for_gen = ar[1] + 100.0

    default = two_way_flat_attention(*ar, *dm, meta, k_ar_for_gen=k_ar_for_gen)
    explicit = naive_two_way_flat_attention(*ar, *dm, meta, k_ar_for_gen=k_ar_for_gen)
    assert all(torch.equal(left, right) for left, right in zip(default, explicit))

    recorder = _record_attention(monkeypatch)
    two_way_flat_attention(*ar, *dm, meta, backend="npu", k_ar_for_gen=k_ar_for_gen)

    causal_call, full_call = recorder.calls
    gather_index = build_two_way_kv_gather_index(meta)
    assert torch.equal(causal_call["k"], ar[1])
    assert torch.equal(full_call["k"], torch.cat((k_ar_for_gen, dm[1]), dim=0).index_select(0, gather_index))


def test_unknown_backend_is_rejected():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=8)

    with pytest.raises(ValueError, match="Unsupported two-way attention backend"):
        two_way_flat_attention(*ar, *dm, meta, backend="cuda")


def test_inputs_must_match_the_metadata():
    meta = TwoWayPackedMeta(*EIGHT_TOKEN_LENGTHS)
    ar, dm = _random_packed_tensors(meta, seed=9)
    wrong_ar = (ar[0][:-1], ar[1][:-1], ar[2][:-1])

    with pytest.raises(ValueError, match="holds 2 tokens, but the metadata describes 3"):
        naive_two_way_flat_attention(*wrong_ar, *dm, meta)

    two_dimensional = ar[0][:, 0, :]
    with pytest.raises(ValueError, match=r"packed \[tokens, heads, head_dim\]"):
        naive_two_way_flat_attention(two_dimensional, *ar[1:], *dm, meta)

    wrong_heads = dm[0].new_zeros(dm[0].size(0), HEADS + 1, HEAD_DIM)
    with pytest.raises(ValueError, match="must agree on heads and head_dim"):
        naive_two_way_flat_attention(*ar, wrong_heads, *dm[1:], meta)


def test_metadata_round_trips_through_cumulative_lengths():
    meta = TwoWayPackedMeta.from_cu_seqlens([0, 3, 8], [0, 5, 5], [0, 8, 13])

    assert meta.ar_lens == (3, 5)
    assert meta.dm_lens == (5, 0)
    assert meta.cu_seqlens_ar == [0, 3, 8]
    assert meta.cu_seqlens_dm == [0, 5, 5]
    assert meta.cu_seqlens_kv == [0, 8, 13]
    assert (meta.max_ar, meta.max_dm, meta.max_kv) == (5, 5, 8)


@pytest.mark.parametrize(
    "ar_lens, dm_lens, error, message",
    [
        ([3], [5, 6], ValueError, "same number of samples"),
        ([-1], [5], ValueError, "non-negative"),
        ([3.5], [5], TypeError, "integers"),
    ],
)
def test_invalid_metadata_is_rejected(ar_lens, dm_lens, error, message):
    with pytest.raises(error, match=message):
        TwoWayPackedMeta(ar_lens, dm_lens)


def test_cumulative_lengths_must_agree_with_the_derived_kv_lengths():
    with pytest.raises(ValueError, match="ar_i \\+ dm_i"):
        TwoWayPackedMeta.from_cu_seqlens([0, 3, 8], [0, 5, 5], [0, 9, 13])


def test_cumulative_lengths_must_describe_the_same_samples():
    with pytest.raises(ValueError, match="describes 3 samples"):
        TwoWayPackedMeta.from_cu_seqlens([0, 3, 8], [0, 5, 5], [0, 8, 13, 20])


def test_gather_index_interleaves_each_sample_into_autoregressive_then_diffusion():
    meta = TwoWayPackedMeta([3, 2], [6, 4])

    index = build_two_way_kv_gather_index(meta)

    assert index.tolist() == [0, 1, 2, 5, 6, 7, 8, 9, 10, 3, 4, 11, 12, 13, 14]


def test_gather_index_omits_samples_without_diffusion_tokens():
    meta = TwoWayPackedMeta([3, 2], [0, 4])

    index = build_two_way_kv_gather_index(meta)

    assert index.tolist() == [3, 4, 5, 6, 7, 8]


def test_gather_index_is_empty_when_no_sample_has_diffusion_tokens():
    meta = TwoWayPackedMeta([3, 2], [0, 0])

    index = build_two_way_kv_gather_index(meta)

    assert index.numel() == 0


class _RecordingAttention:
    """A stand-in for the attention backend that records the two calls and returns the query."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, *args, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, "args": args, "kwargs": kwargs})
        return q


def _record_attention(monkeypatch):
    recorder = _RecordingAttention()
    monkeypatch.setattr(ColoAttention, "_dispatch_kernel", lambda *args, **kwargs: recorder)
    return recorder


def test_npu_path_issues_one_causal_and_one_full_packed_call(monkeypatch):
    meta = TwoWayPackedMeta([3, 2], [6, 4])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=10)
    q_ar, k_ar, v_ar = ar
    q_dm, k_dm, v_dm = dm
    # Give every token a unique key so the interleaved K/V order is readable from the recorded call.
    k_ar[:] = torch.arange(meta.total_ar_tokens, dtype=torch.float32).reshape(-1, 1, 1)
    k_dm[:] = torch.arange(meta.total_ar_tokens, meta.total_kv_tokens, dtype=torch.float32).reshape(-1, 1, 1)
    recorder = _record_attention(monkeypatch)

    out_ar, out_dm = npu_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)

    assert len(recorder.calls) == 2
    causal_call, full_call = recorder.calls

    assert causal_call["kwargs"]["is_causal"] is True
    assert causal_call["kwargs"]["attention_mask"] is None
    assert causal_call["kwargs"]["qkv_layout"] == "TND"
    assert causal_call["kwargs"]["cu_seqlens_q"] == [0, 3, 5]
    assert causal_call["kwargs"]["cu_seqlens_kv"] == [0, 3, 5]
    assert (causal_call["kwargs"]["max_seqlen_q"], causal_call["kwargs"]["max_seqlen_kv"]) == (3, 3)

    assert full_call["kwargs"]["is_causal"] is False
    assert full_call["kwargs"]["attention_mask"] is None
    assert full_call["kwargs"]["cu_seqlens_q"] == [0, 6, 10]
    assert full_call["kwargs"]["cu_seqlens_kv"] == [0, 9, 15]
    assert (full_call["kwargs"]["max_seqlen_q"], full_call["kwargs"]["max_seqlen_kv"]) == (6, 9)

    gather_index = build_two_way_kv_gather_index(meta)
    expected_keys = torch.cat((k_ar, k_dm), dim=0).index_select(0, gather_index)
    assert torch.equal(full_call["k"], expected_keys)
    assert torch.equal(full_call["v"], torch.cat((v_ar, v_dm), dim=0).index_select(0, gather_index))

    assert torch.equal(out_ar, q_ar)
    assert torch.equal(out_dm, q_dm)


def test_npu_path_dispatches_the_two_mask_types(monkeypatch):
    meta = TwoWayPackedMeta([3], [5])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=11)
    selected = []

    def record_mask_type(dtype, mask_type, size):
        selected.append((mask_type, size))
        return _RecordingAttention()

    monkeypatch.setattr(ColoAttention, "_dispatch_kernel", record_mask_type)

    npu_two_way_flat_attention(*ar, *dm, meta)

    assert selected[0][0] is AttnMaskType.CAUSAL
    # CUSTOM without a mask is the full-attention path, which the dispatch represents by ``None``.
    assert selected[1][0] is None


def test_npu_path_drops_zero_length_segments_before_the_kernel(monkeypatch):
    meta = TwoWayPackedMeta([0, 3], [5, 0])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=12)
    recorder = _record_attention(monkeypatch)

    npu_two_way_flat_attention(*ar, *dm, meta)

    assert len(recorder.calls) == 2
    causal_call, full_call = recorder.calls
    assert causal_call["kwargs"]["cu_seqlens_q"] == [0, 3]
    # The only DM sample has no AR tokens, so its K/V segment is just its five DM tokens.
    assert full_call["kwargs"]["cu_seqlens_q"] == [0, 5]
    assert full_call["kwargs"]["cu_seqlens_kv"] == [0, 5]
    assert torch.equal(full_call["k"], dm[1])


def test_npu_path_keys_the_full_call_with_the_normalized_ar_block(monkeypatch):
    meta = TwoWayPackedMeta([3, 2], [6, 4])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=25)
    q_ar, k_ar, v_ar = ar
    q_dm, k_dm, v_dm = dm
    k_ar_for_gen = k_ar + 100.0
    recorder = _record_attention(monkeypatch)

    npu_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta, k_ar_for_gen=k_ar_for_gen)

    causal_call, full_call = recorder.calls
    gather_index = build_two_way_kv_gather_index(meta)
    assert torch.equal(causal_call["k"], k_ar)
    assert torch.equal(full_call["k"], torch.cat((k_ar_for_gen, k_dm), dim=0).index_select(0, gather_index))
    assert torch.equal(full_call["v"], torch.cat((v_ar, v_dm), dim=0).index_select(0, gather_index))


def test_npu_path_explicit_gather_index_matches_the_slice_path(monkeypatch):
    meta = TwoWayPackedMeta([3, 2], [6, 4])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=26)
    gather_index = build_two_way_kv_gather_index(meta)
    recorder = _record_attention(monkeypatch)

    npu_two_way_flat_attention(*ar, *dm, meta, kv_gather_index=gather_index)

    _, full_call = recorder.calls
    assert torch.equal(full_call["k"], torch.cat((ar[1], dm[1]), dim=0).index_select(0, gather_index))
    assert torch.equal(full_call["v"], torch.cat((ar[2], dm[2]), dim=0).index_select(0, gather_index))


def test_naive_path_with_normalized_ar_keys_separates_the_two_k_blocks():
    meta = TwoWayPackedMeta([3], [5])
    ar, dm = _random_packed_tensors(meta, seed=27)
    q_ar, k_ar, v_ar = ar
    q_dm, k_dm, v_dm = dm
    k_ar_for_gen = k_ar + 100.0

    out_ar, out_dm = naive_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta, k_ar_for_gen=k_ar_for_gen)
    changed_ar, changed_dm = naive_two_way_flat_attention(
        q_ar, k_ar * 3.0, v_ar, q_dm, k_dm, v_dm, meta, k_ar_for_gen=k_ar_for_gen
    )

    assert not torch.allclose(out_ar, changed_ar), "the AR pass must keep the raw keys"
    assert torch.allclose(out_dm, changed_dm), "the DM pass must read k_ar_for_gen, not k_ar"


def test_normalized_ar_keys_must_match_the_ar_block():
    meta = TwoWayPackedMeta([3], [5])
    ar, dm = _random_packed_tensors(meta, seed=28)

    with pytest.raises(ValueError, match="shape of the AR key block"):
        naive_two_way_flat_attention(*ar, *dm, meta, k_ar_for_gen=ar[1][:-1])


def test_npu_path_keeps_gqa_heads_and_reads_only_each_samples_own_kv(monkeypatch):
    # Sample 0 has only AR tokens and sample 1 only DM tokens. Every call needs three K/V tokens, so a shared index of
    # the wrong contents could pass the length check while letting one sample read the other's K/V.
    meta = TwoWayPackedMeta([3, 0], [0, 3])
    num_q_heads, num_kv_heads = 4, 2
    generator = torch.Generator().manual_seed(23)
    q_ar = torch.randn(3, num_q_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    q_dm = torch.randn(3, num_q_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    k_ar = torch.randn(3, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    k_dm = torch.randn(3, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    v_ar = torch.randn(3, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    v_dm = torch.randn(3, num_kv_heads, HEAD_DIM, generator=generator, dtype=torch.float32)
    recorder = _record_attention(monkeypatch)

    out_ar, out_dm = npu_two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, meta)

    assert build_two_way_kv_gather_index(meta).tolist() == [3, 4, 5]
    causal_call, full_call = recorder.calls
    assert causal_call["q"].shape == (3, num_q_heads, HEAD_DIM)
    assert causal_call["k"].shape == (3, num_kv_heads, HEAD_DIM)
    assert torch.equal(causal_call["k"], k_ar)
    assert causal_call["kwargs"]["cu_seqlens_kv"] == [0, 3]
    assert full_call["q"].shape == (3, num_q_heads, HEAD_DIM)
    assert full_call["k"].shape == (3, num_kv_heads, HEAD_DIM)
    assert torch.equal(full_call["k"], k_dm)
    assert full_call["kwargs"]["cu_seqlens_kv"] == [0, 3]
    assert torch.equal(out_ar, q_ar) and torch.equal(out_dm, q_dm)


def test_npu_path_reports_the_longest_actual_kv_segment(monkeypatch):
    # The longest sample overall is the AR-only one; the DM call's K/V segments are shorter.
    meta = TwoWayPackedMeta([100, 1], [0, 5])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=24)
    recorder = _record_attention(monkeypatch)

    npu_two_way_flat_attention(*ar, *dm, meta)

    causal_call, full_call = recorder.calls
    assert causal_call["kwargs"]["max_seqlen_kv"] == 100
    assert (full_call["kwargs"]["max_seqlen_q"], full_call["kwargs"]["max_seqlen_kv"]) == (5, 6)


def test_npu_path_skips_a_call_when_one_tower_is_empty(monkeypatch):
    meta = TwoWayPackedMeta([3, 2], [0, 0])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=13)
    recorder = _record_attention(monkeypatch)

    out_ar, out_dm = npu_two_way_flat_attention(*ar, *dm, meta)

    assert len(recorder.calls) == 1
    assert torch.equal(out_ar, ar[0])
    assert out_dm.shape == (0, HEADS, HEAD_DIM)


def test_npu_path_rejects_a_gather_index_of_the_wrong_length(monkeypatch):
    meta = TwoWayPackedMeta([3], [5])
    ar, dm = _random_packed_tensors(meta, dtype=torch.float32, seed=14)
    _record_attention(monkeypatch)

    with pytest.raises(ValueError, match="kv_gather_index selects"):
        npu_two_way_flat_attention(*ar, *dm, meta, kv_gather_index=torch.tensor([0, 1], dtype=torch.long))


def test_varlen_limits_accept_a_reasonable_batch():
    validate_two_way_varlen_limits(TwoWayPackedMeta([3, 5], [5, 7]), num_heads=32, head_dim=128, num_kv_heads=8)


@pytest.mark.parametrize(
    "meta, num_heads, head_dim, num_kv_heads, message",
    [
        (TwoWayPackedMeta([3, 0], [5, 0]), 32, 128, None, "contribute no token"),
        (TwoWayPackedMeta([1] * 2049, [1] * 2049), 32, 128, None, "at most 2048 sequences"),
        (TwoWayPackedMeta([1], [1]), 32, 1024, None, "head dimension"),
        (TwoWayPackedMeta([1], [1]), 512, 128, None, "query heads"),
        (TwoWayPackedMeta([1], [1]), 32, 128, 5, "positive multiple"),
    ],
)
def test_varlen_limits_are_enforced_at_collate_time(meta, num_heads, head_dim, num_kv_heads, message):
    with pytest.raises(ValueError, match=message):
        validate_two_way_varlen_limits(meta, num_heads, head_dim, num_kv_heads)
