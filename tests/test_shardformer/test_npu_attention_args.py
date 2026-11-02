import math
import sys
from unittest import mock

import pytest
import torch

from colossalai.accelerator import get_accelerator
from colossalai.kernel.attention_layout import DENSE_QKV_LAYOUTS, QKV_LAYOUTS, attention_layout
from colossalai.kernel.extensions import ALL_EXTENSIONS, FlashAttentionNpuExtension
from colossalai.kernel.extensions.pybind.flash_attention.flash_attention_npu import (
    CAUSAL_MASK_SIZE,
    resolve_fusion_attention_args,
)
from colossalai.kernel.kernel_loader import FlashAttentionLoader, FlashAttentionWithCustomMaskLoader
from colossalai.shardformer.layer import AttnMaskType, ColoAttention

BATCH = 2
HEADS = 4
SEQUENCE = 8
HEAD_DIM = 16
HIDDEN = HEADS * HEAD_DIM
TOKENS = BATCH * SEQUENCE


def _tensors(qkv_layout):
    shape = {
        "BNSD": (BATCH, HEADS, SEQUENCE, HEAD_DIM),
        "BSND": (BATCH, SEQUENCE, HEADS, HEAD_DIM),
        "BSH": (BATCH, SEQUENCE, HIDDEN),
        "TND": (TOKENS, HEADS, HEAD_DIM),
    }[qkv_layout]
    return tuple(torch.randn(shape) for _ in range(3))


def _layout_kwargs(qkv_layout):
    if qkv_layout == "TND":
        cu_seqlens = torch.tensor([0, 7, TOKENS], dtype=torch.int32)
        return {"qkv_layout": "TND", "cu_seqlens_q": cu_seqlens, "cu_seqlens_kv": cu_seqlens}
    kwargs = {"qkv_layout": qkv_layout}
    if qkv_layout == "BSH":
        kwargs["head_num"] = HEADS
    return kwargs


def _assert_compressed_causal_mask(mask):
    assert mask.dtype is torch.bool
    assert mask.shape == (CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE)
    assert not mask[0, 0]
    assert mask[0, 1]
    assert not mask[-1, 0]
    assert not mask[-1, -1]
    row = CAUSAL_MASK_SIZE // 2
    assert not mask[row, : row + 1].any()
    assert mask[row, row + 1 :].all()


class _RecordingFusionAttention:
    """A stand-in for ``torch_npu`` that records how the kernel would have been called."""

    def __init__(self):
        self.calls = []

    def npu_fusion_attention(self, *arguments, **keywords):
        self.calls.append((arguments, keywords))
        return (arguments[0],)


def test_layout_registry_describes_the_supported_layouts():
    assert QKV_LAYOUTS == ("BNSD", "BSND", "BSH", "TND")
    assert DENSE_QKV_LAYOUTS == ("BNSD", "BSND", "BSH")
    assert attention_layout("TND").is_packed
    assert attention_layout("BSH").has_fused_heads
    assert not attention_layout("BSND").has_fused_heads


@pytest.mark.parametrize("qkv_layout", QKV_LAYOUTS)
def test_head_count_and_scale_follow_the_tensor_layout(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, **_layout_kwargs(qkv_layout))

    assert arguments["head_num"] == HEADS
    assert arguments["input_layout"] == qkv_layout
    assert arguments["scale"] == pytest.approx(1 / math.sqrt(HEAD_DIM))
    assert arguments["keep_prob"] == 1.0


@pytest.mark.parametrize("qkv_layout", DENSE_QKV_LAYOUTS)
def test_missing_mask_is_passed_as_none_instead_of_crashing(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, **_layout_kwargs(qkv_layout))

    assert arguments["atten_mask"] is None
    assert arguments["actual_seq_qlen"] is None
    assert arguments["actual_seq_kvlen"] is None


def test_packed_full_attention_never_receives_a_mask():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(q, k, v, **_layout_kwargs("TND"))

    assert arguments["atten_mask"] is None
    assert arguments["actual_seq_qlen"] == [7, TOKENS]
    assert arguments["actual_seq_kvlen"] == [7, TOKENS]
    assert arguments["sparse_mode"] == 0


def test_dropout_is_turned_into_keep_probability():
    q, k, v = _tensors("BNSD")

    assert resolve_fusion_attention_args(q, k, v, dropout_p=0.1)["keep_prob"] == pytest.approx(0.9)


@pytest.mark.parametrize("qkv_layout", DENSE_QKV_LAYOUTS)
def test_dense_causal_call_passes_the_compressed_mask(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    arguments = resolve_fusion_attention_args(q, k, v, is_causal=True, **_layout_kwargs(qkv_layout))

    _assert_compressed_causal_mask(arguments["atten_mask"])
    assert arguments["sparse_mode"] == 3, "the compressed mask is read with the right-down causal window"


def test_packed_causal_call_passes_the_compressed_mask():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(q, k, v, is_causal=True, **_layout_kwargs("TND"))

    _assert_compressed_causal_mask(arguments["atten_mask"])
    assert arguments["sparse_mode"] == 3


def test_causal_mask_is_shared_by_every_layout_and_sequence_length():
    masks = []
    for qkv_layout in QKV_LAYOUTS:
        q, k, v = _tensors(qkv_layout)
        masks.append(resolve_fusion_attention_args(q, k, v, is_causal=True, **_layout_kwargs(qkv_layout))["atten_mask"])
    short_q, short_k, short_v = _tensors("BNSD")
    masks.append(
        resolve_fusion_attention_args(short_q[:, :, :4], short_k[:, :, :4], short_v[:, :, :4], is_causal=True)[
            "atten_mask"
        ]
    )

    assert all(mask is masks[0] for mask in masks), "one compressed mask must serve every shape and layout"
    assert all(mask.data_ptr() == masks[0].data_ptr() for mask in masks)


def test_floating_point_mask_is_read_as_non_zero_means_masked():
    q, k, v = _tensors("BNSD")
    additive_mask = torch.zeros(SEQUENCE, SEQUENCE)
    additive_mask[0, 1] = torch.finfo(additive_mask.dtype).min

    arguments = resolve_fusion_attention_args(q, k, v, attention_mask=additive_mask)

    assert arguments["atten_mask"].dtype is torch.bool
    assert arguments["atten_mask"][0, 0, 0, 1]
    assert not arguments["atten_mask"][0, 0, 0, 0]
    assert arguments["sparse_mode"] == 0


@pytest.mark.parametrize("qkv_layout", DENSE_QKV_LAYOUTS)
def test_boolean_mask_is_promoted_to_four_dimensions(qkv_layout):
    q, k, v = _tensors(qkv_layout)
    two_dimensional = torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.bool)
    three_dimensional = two_dimensional.unsqueeze(0).expand(BATCH, SEQUENCE, SEQUENCE)
    four_dimensional = two_dimensional.unsqueeze(0).unsqueeze(0)

    expected_shape = {
        2: (1, 1, SEQUENCE, SEQUENCE),
        3: (BATCH, 1, SEQUENCE, SEQUENCE),
        4: (1, 1, SEQUENCE, SEQUENCE),
    }
    for mask in (two_dimensional, three_dimensional, four_dimensional):
        atten_mask = resolve_fusion_attention_args(q, k, v, attention_mask=mask, **_layout_kwargs(qkv_layout))[
            "atten_mask"
        ]
        assert atten_mask.shape == expected_shape[mask.dim()]


def test_uint8_mask_is_read_like_a_boolean_mask():
    q, k, v = _tensors("BNSD")
    mask = torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.uint8)
    mask[0, 0] = 1

    atten_mask = resolve_fusion_attention_args(q, k, v, attention_mask=mask)["atten_mask"]

    assert atten_mask.dtype is torch.bool
    assert atten_mask[0, 0, 0, 0]


def test_mask_shape_must_match_the_sequences():
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="does not match q"):
        resolve_fusion_attention_args(q, k, v, attention_mask=torch.zeros(SEQUENCE, SEQUENCE + 1, dtype=torch.bool))


def test_mask_batch_and_head_dimensions_are_validated():
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="does not match q"):
        resolve_fusion_attention_args(
            q, k, v, attention_mask=torch.zeros(BATCH + 1, 1, SEQUENCE, SEQUENCE, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="does not match q"):
        resolve_fusion_attention_args(
            q, k, v, attention_mask=torch.zeros(1, HEADS + 1, SEQUENCE, SEQUENCE, dtype=torch.bool)
        )


def test_integer_mask_is_rejected():
    q, k, v = _tensors("BNSD")

    with pytest.raises(TypeError, match="boolean, uint8 or floating point"):
        resolve_fusion_attention_args(q, k, v, attention_mask=torch.zeros(SEQUENCE, SEQUENCE, dtype=torch.int32))


def test_padded_sequences_are_described_by_cumulative_lengths():
    q, k, v = _tensors("TND")
    cu_seqlens = torch.tensor([0, 3, 8, TOKENS], dtype=torch.int32)

    arguments = resolve_fusion_attention_args(q, k, v, qkv_layout="TND", cu_seqlens_q=cu_seqlens)

    assert arguments["actual_seq_qlen"] == [3, 8, TOKENS]
    assert arguments["actual_seq_kvlen"] == [3, 8, TOKENS]
    assert arguments["input_layout"] == "TND"


def test_zero_length_query_sequence_is_allowed():
    q, k, v = _tensors("TND")
    cu_seqlens = torch.tensor([0, 3, 3, TOKENS], dtype=torch.int32)

    arguments = resolve_fusion_attention_args(
        q, k, v, is_causal=True, qkv_layout="TND", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens
    )

    assert arguments["actual_seq_qlen"] == [3, 3, TOKENS]
    assert arguments["sparse_mode"] == 3


def test_query_without_keys_is_rejected():
    q, k, v = _tensors("TND")

    with pytest.raises(ValueError, match="no key/value tokens"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            qkv_layout="TND",
            cu_seqlens_q=torch.tensor([0, 3, 8, TOKENS], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 3, 3, TOKENS], dtype=torch.int32),
        )


def test_causal_packed_call_rejects_a_query_longer_than_its_keys():
    q, k, v = _tensors("TND")

    with pytest.raises(ValueError, match="no longer than its key/value sequence"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            is_causal=True,
            qkv_layout="TND",
            cu_seqlens_q=torch.tensor([0, 5, TOKENS], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 3, TOKENS], dtype=torch.int32),
        )


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


def test_packed_layout_reads_an_empty_sentinel_mask_as_none():
    q, k, v = _tensors("TND")

    full = resolve_fusion_attention_args(q, k, v, attention_mask=torch.empty(0), **_layout_kwargs("TND"))
    causal = resolve_fusion_attention_args(
        q, k, v, attention_mask=torch.empty(0), is_causal=True, **_layout_kwargs("TND")
    )

    assert full["atten_mask"] is None
    assert full["sparse_mode"] == 0
    _assert_compressed_causal_mask(causal["atten_mask"])


@pytest.mark.parametrize("qkv_layout", DENSE_QKV_LAYOUTS)
def test_dense_layout_rejects_cumulative_lengths(qkv_layout):
    q, k, v = _tensors(qkv_layout)

    with pytest.raises(ValueError, match="cannot describe variable-length sequences"):
        resolve_fusion_attention_args(
            q,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            **_layout_kwargs(qkv_layout),
        )


@pytest.mark.parametrize(
    "cu_seqlens, error, message",
    [
        (torch.tensor([3, 8, TOKENS], dtype=torch.int32), ValueError, "start at 0"),
        (torch.tensor([0, 8, 7, TOKENS], dtype=torch.int32), ValueError, "non-decreasing"),
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


@pytest.mark.parametrize("qkv_layout", ["SBH", "bnsd"])
def test_unknown_layout_is_rejected(qkv_layout):
    q, k, v = _tensors("BNSD")

    with pytest.raises(ValueError, match="Unsupported qkv_layout"):
        resolve_fusion_attention_args(q, k, v, qkv_layout=qkv_layout)


def test_bsh_requires_the_head_count():
    q, k, v = _tensors("BSH")

    with pytest.raises(ValueError, match="head_num is required"):
        resolve_fusion_attention_args(q, k, v, qkv_layout="BSH")


def test_bsh_head_count_must_divide_the_hidden_size():
    q, k, v = _tensors("BSH")

    with pytest.raises(ValueError, match="divisible by head_num"):
        resolve_fusion_attention_args(q, k, v, qkv_layout="BSH", head_num=3)


def test_bsh_fused_head_dimension_must_match():
    q, _, _ = _tensors("BSH")
    k = torch.randn(BATCH, SEQUENCE, 2 * HIDDEN)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="same fused head dimension"):
        resolve_fusion_attention_args(q, k, v, qkv_layout="BSH", head_num=HEADS)


@pytest.mark.parametrize("qkv_layout", ["BNSD", "TND"])
def test_mismatched_head_counts_are_rejected(qkv_layout):
    q, _, _ = _tensors(qkv_layout)
    if qkv_layout == "TND":
        k = torch.randn(TOKENS, HEADS + 1, HEAD_DIM)
    else:
        k = torch.randn(BATCH, HEADS + 1, SEQUENCE, HEAD_DIM)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="same number of heads"):
        resolve_fusion_attention_args(q, k, v, **_layout_kwargs(qkv_layout))


def test_max_sequence_length_is_checked_but_no_longer_sizes_the_mask():
    q, k, v = _tensors("TND")

    arguments = resolve_fusion_attention_args(
        q, k, v, is_causal=True, max_seqlen_q=9, max_seqlen_kv=9, **_layout_kwargs("TND")
    )
    _assert_compressed_causal_mask(arguments["atten_mask"])

    with pytest.raises(ValueError, match="smaller than the longest sequence"):
        resolve_fusion_attention_args(q, k, v, is_causal=True, max_seqlen_q=4, **_layout_kwargs("TND"))


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


@pytest.mark.parametrize("qkv_layout", ["BNSD", "BSH"])
def test_padded_batch_is_unpadded_before_the_kernel(monkeypatch, qkv_layout):
    recorder = _RecordingFusionAttention()
    monkeypatch.setitem(sys.modules, "torch_npu", recorder)
    flash_attention = FlashAttentionNpuExtension().load()
    q, k, v = _tensors(qkv_layout)
    q_indices = torch.tensor([0, 1, 2, 8, 9, 10, 11, 12])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    kwargs = {"head_num": HEADS} if qkv_layout == "BSH" else {}

    output = flash_attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        q_indices=q_indices,
        kv_indices=q_indices,
        qkv_layout=qkv_layout,
        **kwargs,
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
    restored = output.transpose(1, 2) if qkv_layout == "BNSD" else output
    restored = restored.reshape(-1, HEADS, HEAD_DIM)
    assert torch.equal(restored[q_indices], packed_q)
    assert torch.count_nonzero(restored) == packed_q.numel()


def test_padded_causal_batch_receives_the_compressed_mask(monkeypatch):
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
    _assert_compressed_causal_mask(keywords["atten_mask"])


def test_entry_forwards_a_non_default_layout():
    forwarded = []

    def fake_attention(*args, **kwargs):
        forwarded.append(kwargs)
        return args[0]

    with mock.patch.object(ColoAttention, "_dispatch_kernel", return_value=fake_attention):
        q, k, v = _tensors("BNSD")
        ColoAttention.attention(q, k, v)
        assert "qkv_layout" not in forwarded[-1], "the default layout must keep the kernel signature unchanged"

        ColoAttention.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), qkv_layout="BSND")
        assert forwarded[-1]["qkv_layout"] == "BSND"

        packed_q, packed_k, packed_v = _tensors("TND")
        ColoAttention.attention(packed_q, packed_k, packed_v, **_layout_kwargs("TND"))
        assert forwarded[-1]["qkv_layout"] == "TND"


@pytest.mark.parametrize("qkv_layout", ["SBH", "bnsd"])
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
