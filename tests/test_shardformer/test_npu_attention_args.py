import math
import sys
from unittest import mock

import pytest
import torch

from colossalai.accelerator import get_accelerator
from colossalai.kernel.extensions import ALL_EXTENSIONS, FlashAttentionNpuExtension
from colossalai.kernel.extensions.pybind.flash_attention.flash_attention_npu import (
    QKV_LAYOUTS,
    resolve_fusion_attention_args,
)
from colossalai.kernel.kernel_loader import FlashAttentionLoader, FlashAttentionWithCustomMaskLoader
from colossalai.shardformer.layer import AttnMaskType, ColoAttention, SegmentMaskSpec, SegmentRule, build_segment_mask
from colossalai.shardformer.layer.attn import invert_mask

BATCH = 2
HEADS = 4
SEQUENCE = 8
HEAD_DIM = 16
TOKENS = BATCH * SEQUENCE
DENSE_LAYOUTS = ("BNSD", "BSND")


def _tensors(qkv_layout):
    shape = {
        "BNSD": (BATCH, HEADS, SEQUENCE, HEAD_DIM),
        "BSND": (BATCH, SEQUENCE, HEADS, HEAD_DIM),
        "TND": (TOKENS, HEADS, HEAD_DIM),
    }[qkv_layout]
    return tuple(torch.randn(shape) for _ in range(3))


def _layout_kwargs(qkv_layout):
    if qkv_layout == "TND":
        cu_seqlens = torch.tensor([0, 7, TOKENS], dtype=torch.int32)
        return {"qkv_layout": "TND", "cu_seqlens_q": cu_seqlens, "cu_seqlens_kv": cu_seqlens}
    return {"qkv_layout": qkv_layout}


def _segment_mask():
    return build_segment_mask(
        SegmentMaskSpec([3, 5], [[SegmentRule.CAUSAL, SegmentRule.FULL], [SegmentRule.FULL, SegmentRule.FULL]])
    )


class _RecordingFusionAttention:
    """A stand-in for ``torch_npu`` that records how the kernel would have been called."""

    def __init__(self):
        self.calls = []

    def npu_fusion_attention(self, *arguments, **keywords):
        self.calls.append((arguments, keywords))
        return (arguments[0],)


@pytest.mark.parametrize("qkv_layout", QKV_LAYOUTS)
def test_head_count_and_scale_follow_the_tensor_layout(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, **_layout_kwargs(qkv_layout))

    assert arguments["head_num"] == HEADS
    assert arguments["input_layout"] == qkv_layout
    assert arguments["scale"] == pytest.approx(1 / math.sqrt(HEAD_DIM))
    assert arguments["keep_prob"] == 1.0


@pytest.mark.parametrize("qkv_layout", DENSE_LAYOUTS)
def test_missing_mask_is_passed_as_none_instead_of_crashing(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, qkv_layout=qkv_layout)

    assert arguments["atten_mask"] is None
    assert arguments["actual_seq_qlen"] is None
    assert arguments["actual_seq_kvlen"] is None


def test_packed_full_attention_never_receives_a_mask():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(q, k, v, **_layout_kwargs("TND"))

    assert arguments["atten_mask"] is None
    assert arguments["sparse_mode"] == 0


def test_dropout_is_turned_into_keep_probability():
    q, k, v = _tensors("BNSD")

    assert resolve_fusion_attention_args(q, k, v, dropout_p=0.1)["keep_prob"] == pytest.approx(0.9)


@pytest.mark.parametrize("qkv_layout", DENSE_LAYOUTS)
def test_dense_causal_call_passes_a_lower_triangular_mask(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, is_causal=True, qkv_layout=qkv_layout)

    atten_mask = arguments["atten_mask"]
    assert atten_mask.dtype is torch.bool
    assert atten_mask.shape == (SEQUENCE, SEQUENCE)
    assert torch.equal(atten_mask, ~torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).tril())
    assert arguments["sparse_mode"] == 3, "the mask is read with the right-down causal window"


def test_packed_causal_call_passes_a_mask_sized_by_the_longest_sequence():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(q, k, v, is_causal=True, **_layout_kwargs("TND"))

    # cu_seqlens [0, 7, 16] describes sequences of 7 and 9 tokens, so the SS mask is [9, 9].
    longest = 9
    atten_mask = arguments["atten_mask"]
    assert atten_mask.dtype is torch.bool
    assert atten_mask.shape == (longest, longest)
    assert torch.equal(atten_mask, ~torch.ones(longest, longest, dtype=torch.bool).tril())
    assert arguments["sparse_mode"] == 3, "the mask is read with the right-down causal window"


def test_packed_causal_call_accepts_a_host_side_maximum():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(
        q,
        k,
        v,
        is_causal=True,
        **_layout_kwargs("TND"),
        max_seqlen_q=12,
        max_seqlen_kv=12,
    )

    # The mask may overshoot the real maximum: sparse_mode=3 cuts each sequence out of its bottom-right corner.
    assert arguments["atten_mask"].shape == (12, 12)
    assert torch.equal(arguments["atten_mask"], ~torch.ones(12, 12, dtype=torch.bool).tril())


def test_packed_causal_mask_is_rectangular_when_query_and_key_lengths_differ():
    q = torch.randn(6, HEADS, HEAD_DIM)
    k = torch.randn(8, HEADS, HEAD_DIM)
    v = torch.randn(8, HEADS, HEAD_DIM)

    arguments = resolve_fusion_attention_args(
        q,
        k,
        v,
        is_causal=True,
        cu_seqlens_q=torch.tensor([0, 2, 6], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 8], dtype=torch.int32),
        qkv_layout="TND",
    )

    expected = ~torch.ones(4, 5, dtype=torch.bool).tril(diagonal=1)
    assert arguments["atten_mask"].shape == (4, 5)
    assert torch.equal(arguments["atten_mask"], expected)


def test_causal_mask_is_cached_per_shape():
    q, k, v = _tensors("BNSD")

    first = resolve_fusion_attention_args(q, k, v, is_causal=True)["atten_mask"]
    second = resolve_fusion_attention_args(q, k, v, is_causal=True)["atten_mask"]
    other = resolve_fusion_attention_args(q, k[:, :, :4], v[:, :, :4], is_causal=True)["atten_mask"]

    assert first is second, "the same (maxSq, maxSkv) must reuse the cached mask"
    assert other.shape == (SEQUENCE, 4)
    assert other.data_ptr() != first.data_ptr()


def test_floating_point_mask_is_read_as_non_zero_means_masked():
    q, k, v = _tensors("BNSD")
    additive_mask = invert_mask(_segment_mask().unsqueeze(0).unsqueeze(0).to(torch.bfloat16))

    aten_mask = resolve_fusion_attention_args(q, k, v, attention_mask=additive_mask)["atten_mask"]

    assert aten_mask.dtype is torch.bool
    assert torch.equal(aten_mask.squeeze(), ~_segment_mask())
    assert resolve_fusion_attention_args(q, k, v, attention_mask=additive_mask)["sparse_mode"] == 0


def test_boolean_mask_is_promoted_to_four_dimensions():
    q, k, v = _tensors("BNSD")
    two_dimensional = torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.bool)
    three_dimensional = two_dimensional.unsqueeze(0).expand(BATCH, SEQUENCE, SEQUENCE)
    four_dimensional = two_dimensional.unsqueeze(0).unsqueeze(0)

    expected_shape = {
        2: (1, 1, SEQUENCE, SEQUENCE),
        3: (BATCH, 1, SEQUENCE, SEQUENCE),
        4: (1, 1, SEQUENCE, SEQUENCE),
    }
    for mask in (two_dimensional, three_dimensional, four_dimensional):
        atten_mask = resolve_fusion_attention_args(q, k, v, attention_mask=mask)["atten_mask"]
        assert atten_mask.shape == expected_shape[mask.dim()]


def test_integer_mask_is_rejected():
    q, k, v = _tensors("BNSD")

    with pytest.raises(TypeError, match="boolean or floating point"):
        resolve_fusion_attention_args(q, k, v, attention_mask=torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.int32))


def test_padded_sequences_are_described_by_cumulative_lengths():
    q, k, v = _tensors("TND")
    cu_seqlens = torch.tensor([0, 3, 8, TOKENS], dtype=torch.int32)

    arguments = resolve_fusion_attention_args(q, k, v, qkv_layout="TND", cu_seqlens_q=cu_seqlens)

    assert arguments["actual_seq_qlen"] == [3, 8, TOKENS]
    assert arguments["actual_seq_kvlen"] == [3, 8, TOKENS]
    assert arguments["input_layout"] == "TND"


def test_packed_layout_rejects_a_dense_mask():
    q, k, v = _tensors("TND")

    with pytest.raises(NotImplementedError, match="does not accept a dense attention_mask"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            attention_mask=torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.bool),
            **_layout_kwargs("TND"),
        )


@pytest.mark.parametrize("qkv_layout", DENSE_LAYOUTS)
def test_dense_layout_rejects_cumulative_lengths(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    with pytest.raises(ValueError, match="cannot describe variable-length sequences"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            qkv_layout=qkv_layout,
        )


@pytest.mark.parametrize(
    "cu_seqlens, error, message",
    [
        (torch.tensor([3, 8, TOKENS], dtype=torch.int32), ValueError, "start at 0"),
        (torch.tensor([0, 8, 8, TOKENS], dtype=torch.int32), ValueError, "strictly increasing"),
        (torch.tensor([0, 5, 12], dtype=torch.int32), ValueError, "end at"),
        (torch.tensor([0.0, 8.0, float(TOKENS)]), TypeError, "integers"),
        (torch.tensor(TOKENS, dtype=torch.int32), ValueError, "1D prefix sum"),
    ],
)
def test_invalid_cumulative_lengths_are_rejected(cu_seqlens, error, message):
    q, k, v = _tensors("TND")

    with pytest.raises(error, match=message):
        resolve_fusion_attention_args(q, k, v, qkv_layout="TND", cu_seqlens_q=cu_seqlens)


def test_packed_lengths_must_agree_on_the_number_of_sequences():
    q, k, v = _tensors("TND")

    with pytest.raises(ValueError, match="same number of sequences"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            qkv_layout="TND",
            cu_seqlens_q=torch.tensor([0, 7, TOKENS], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 5, 9, TOKENS], dtype=torch.int32),
        )


def test_packed_layout_requires_three_dimensional_tensors():
    q, k, v = _tensors("BSND")

    with pytest.raises(ValueError, match="3D"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            qkv_layout="TND",
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        )


def test_unknown_layout_is_rejected():
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="Unsupported qkv_layout"):
        resolve_fusion_attention_args(q, k, v, qkv_layout="BSH")


def test_mismatched_head_counts_are_rejected():
    q, k, _ = _tensors("BNSD")
    v = torch.randn(BATCH, HEADS + 1, SEQUENCE, HEAD_DIM)

    with pytest.raises(ValueError, match="same number of heads"):
        resolve_fusion_attention_args(q, k, v)


def test_availability_follows_the_accelerator():
    is_available = FlashAttentionNpuExtension().is_available()

    if get_accelerator().name != "npu":
        assert is_available is False, "the NPU kernel must not be selected on a non-NPU machine"


def test_npu_extension_is_reachable_through_the_attention_loaders():
    assert FlashAttentionNpuExtension in ALL_EXTENSIONS
    assert FlashAttentionNpuExtension in FlashAttentionLoader.REGISTRY
    assert FlashAttentionNpuExtension in FlashAttentionWithCustomMaskLoader.REGISTRY


def test_loaded_kernel_advertises_the_window_it_builds(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", _RecordingFusionAttention())

    assert FlashAttentionNpuExtension().load().supports_internal_causal_window is True


def test_padded_batch_is_unpadded_before_the_kernel(monkeypatch):
    recorder = _RecordingFusionAttention()
    monkeypatch.setitem(sys.modules, "torch_npu", recorder)
    flash_attention = FlashAttentionNpuExtension().load()
    q, k, v = _tensors("BNSD")
    q_indices = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    output = flash_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        q_indices=q_indices,
        kv_indices=q_indices,
    )

    arguments, keywords = recorder.calls[0]
    packed_q, packed_k, packed_v, head_num, input_layout = arguments
    assert packed_q.shape == (q_indices.numel(), HEADS, HEAD_DIM)
    assert packed_k.shape == packed_q.shape and packed_v.shape == packed_q.shape
    assert head_num == HEADS
    assert input_layout == "TND"
    assert keywords["actual_seq_qlen"] == [3, 8]
    assert keywords["actual_seq_kvlen"] == [3, 8]
    assert "atten_mask" not in keywords, "the padding is gone, so the packed path needs no dense mask"
    assert output.shape == q.shape
    # The kernel output is written back to the un-padded positions and the padding positions stay zero.
    restored = output.transpose(1, 2).reshape(-1, HEADS, HEAD_DIM)
    assert torch.equal(restored[q_indices], packed_q)
    assert torch.count_nonzero(restored) == packed_q.numel()


def test_padded_causal_batch_receives_the_cached_mask(monkeypatch):
    recorder = _RecordingFusionAttention()
    monkeypatch.setitem(sys.modules, "torch_npu", recorder)
    flash_attention = FlashAttentionNpuExtension().load()
    q, k, v = _tensors("BNSD")
    q_indices = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    flash_attention(
        q,
        k,
        v,
        is_causal=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        q_indices=q_indices,
        kv_indices=q_indices,
        max_seqlen_q=5,
        max_seqlen_kv=5,
    )

    _, keywords = recorder.calls[0]
    assert keywords["sparse_mode"] == 3
    assert keywords["atten_mask"].shape == (5, 5)
    assert torch.equal(keywords["atten_mask"], ~torch.ones(5, 5, dtype=torch.bool).tril())


def test_empty_mask_sentinel_still_gets_the_cached_causal_mask():
    q, k, v = _tensors("BNSD")

    arguments = resolve_fusion_attention_args(q, k, v, attention_mask=torch.empty(0), is_causal=True)

    assert arguments["atten_mask"] is not None
    assert arguments["atten_mask"].shape == (SEQUENCE, SEQUENCE)
    assert arguments["sparse_mode"] == 3


def test_entry_forwards_the_layout_only_when_it_is_not_the_default():
    q, k, v = _tensors("BNSD")
    forwarded = []

    def fake_attention(*args, **kwargs):
        forwarded.append(kwargs)
        return args[0]

    with mock.patch.object(ColoAttention, "_dispatch_kernel", return_value=fake_attention):
        ColoAttention.attention(q, k, v)
        assert "qkv_layout" not in forwarded[-1], "the default layout must keep the kernel signature unchanged"

        ColoAttention.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), qkv_layout="BSND")
        assert forwarded[-1]["qkv_layout"] == "BSND"

        packed_q, packed_k, packed_v = _tensors("TND")
        ColoAttention.attention(packed_q, packed_k, packed_v, **_layout_kwargs("TND"))
        assert forwarded[-1]["qkv_layout"] == "TND"


@pytest.mark.parametrize("qkv_layout", ["BSH", "bnsd"])
def test_entry_rejects_unsupported_layouts(qkv_layout):
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="Unsupported qkv_layout"):
        ColoAttention.attention(q, k, v, qkv_layout=qkv_layout)


def test_entry_can_request_causal_attention_without_a_dense_mask():
    q, k, v = _tensors("BNSD")
    forwarded = []

    def fake_attention(*args, **kwargs):
        forwarded.append(kwargs)
        return args[0]

    with mock.patch.object(ColoAttention, "_dispatch_kernel", return_value=fake_attention) as dispatch:
        ColoAttention.attention(q, k, v, attention_mask_type=AttnMaskType.CAUSAL)

    assert dispatch.call_args.args[1] is AttnMaskType.CAUSAL
    assert forwarded[-1]["attention_mask"] is None
    assert forwarded[-1]["is_causal"] is True


def test_entry_rejects_padding_attention_without_a_mask():
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="AttnMaskType.CUSTOM or AttnMaskType.CAUSAL"):
        ColoAttention.attention(q, k, v, attention_mask_type=AttnMaskType.PADDED)


def test_oversized_mask_needs_a_backend_that_builds_the_window_itself():
    q, k, v = _tensors("BNSD")
    attention_mask = torch.zeros(1, 1, SEQUENCE, SEQUENCE)

    def backend_without_window_support(*args, **kwargs):
        return args[0]

    with mock.patch.object(ColoAttention, "_dispatch_kernel", return_value=backend_without_window_support):
        with mock.patch("colossalai.shardformer.layer.attn.MEMORY_BOUND", 1):
            with pytest.raises(NotImplementedError, match="MEMORY_BOUND"):
                ColoAttention.attention(q, k, v, attention_mask=attention_mask, attention_mask_type=AttnMaskType.CAUSAL)

            backend_without_window_support.supports_internal_causal_window = True
            assert (
                ColoAttention.attention(q, k, v, attention_mask=attention_mask, attention_mask_type=AttnMaskType.CAUSAL)
                is q
            )
