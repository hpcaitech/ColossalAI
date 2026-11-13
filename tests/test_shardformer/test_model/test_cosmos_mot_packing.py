import pytest
import torch

from colossalai.shardformer.layer import two_way_flat_attention
from colossalai.shardformer.modeling.cosmos_mot import (
    CosmosTwoWayLayout,
    all_reduce_supervised_tokens,
    layout_from_split_info,
    layout_from_split_lengths,
    pack_cosmos3_samples,
    token_mean_loss_scale,
)

HEADS = 2
HEAD_DIM = 8


def test_layout_splits_a_pack_into_the_kernel_blocks():
    layout = layout_from_split_lengths([3, 5, 2, 4], ["causal", "full", "causal", "full"], [8, 6])

    assert layout.meta.ar_lens == (3, 2)
    assert layout.meta.dm_lens == (5, 4)
    assert (layout.real_tokens, layout.pad_tokens, layout.sequence_length) == (14, 0, 14)

    packed = torch.arange(layout.sequence_length)
    ar_block, dm_block = layout.split_tensor(packed)

    assert ar_block.tolist() == [0, 1, 2, 8, 9]
    assert dm_block.tolist() == [3, 4, 5, 6, 7, 10, 11, 12, 13]
    assert layout.merge_outputs(ar_block, dm_block).tolist() == packed.tolist()


def test_layout_accepts_a_sample_with_only_one_tower():
    # Sample 0 has both towers, sample 1 is text only and sample 2 is generation only.
    layout = layout_from_split_lengths([3, 5, 2, 4], ["causal", "full", "causal", "full"], [8, 2, 4])

    assert layout.meta.ar_lens == (3, 2, 0)
    assert layout.meta.dm_lens == (5, 0, 4)


def test_layout_keeps_zero_length_splits_out_of_the_blocks():
    layout = layout_from_split_lengths([0, 5, 3, 0], ["causal", "full", "causal", "full"], [5, 3])

    assert layout.meta.ar_lens == (0, 3)
    assert layout.meta.dm_lens == (5, 0)

    packed = torch.arange(8)
    ar_block, dm_block = layout.split_tensor(packed)
    assert ar_block.tolist() == [5, 6, 7]
    assert dm_block.tolist() == [0, 1, 2, 3, 4]


def test_layout_skips_and_zero_fills_the_pad_tail():
    layout = layout_from_split_lengths([3, 5], ["causal", "full"], [8], pad_tokens=4)

    assert (layout.real_tokens, layout.sequence_length) == (8, 12)
    packed = torch.arange(12)
    ar_block, dm_block = layout.split_tensor(packed)
    assert ar_block.tolist() == [0, 1, 2]
    assert dm_block.tolist() == [3, 4, 5, 6, 7]
    assert layout.merge_outputs(ar_block, dm_block).tolist() == list(range(8)) + [0, 0, 0, 0]


def test_split_and_merge_backpropagate_into_pack_order():
    layout = layout_from_split_lengths([3, 5, 2, 4], ["causal", "full", "causal", "full"], [8, 6], pad_tokens=2)
    packed = torch.randn(layout.sequence_length, HEADS, HEAD_DIM, requires_grad=True)

    ar_block, dm_block = layout.split_tensor(packed)
    layout.merge_outputs(ar_block * 2.0, dm_block * 3.0).sum().backward()

    grad = packed.grad
    assert grad is not None
    assert torch.equal(grad[0:3], torch.full((3, HEADS, HEAD_DIM), 2.0))
    assert torch.equal(grad[3:8], torch.full((5, HEADS, HEAD_DIM), 3.0))
    assert torch.equal(grad[8:10], torch.full((2, HEADS, HEAD_DIM), 2.0))
    assert torch.equal(grad[10:14], torch.full((4, HEADS, HEAD_DIM), 3.0))
    assert torch.equal(grad[14:16], torch.zeros(2, HEADS, HEAD_DIM))


def test_layout_meta_drives_the_two_way_kernel():
    layout = layout_from_split_lengths([3, 5, 2, 4], ["causal", "full", "causal", "full"], [8, 6])
    generator = torch.Generator().manual_seed(0)
    packed = tuple(
        torch.randn(layout.sequence_length, HEADS, HEAD_DIM, generator=generator, dtype=torch.float64) for _ in range(3)
    )

    q_ar, q_dm = layout.split_tensor(packed[0])
    k_ar, k_dm = layout.split_tensor(packed[1])
    v_ar, v_dm = layout.split_tensor(packed[2])
    out_ar, out_dm = two_way_flat_attention(q_ar, k_ar, v_ar, q_dm, k_dm, v_dm, layout.meta)
    merged = layout.merge_outputs(out_ar, out_dm)

    assert merged.shape == packed[0].shape
    assert torch.isfinite(merged).all()


def test_layout_from_split_info_reads_attributes_and_mappings():
    class SplitInfo:
        split_lens = [3, 5]
        attn_modes = ["causal", "full"]
        sample_lens = [8]

    from_attributes = layout_from_split_info(SplitInfo())
    from_mapping = layout_from_split_info({"split_lens": [3, 5], "attn_modes": ["causal", "full"], "sample_lens": [8]})

    assert from_attributes.meta == from_mapping.meta
    assert from_attributes.meta == layout_from_split_lengths([3, 5], ["causal", "full"], [8]).meta


def test_layout_from_split_info_accepts_an_explicit_sample_lens_override():
    class SplitInfo:
        split_lens = [3, 5]
        attn_modes = ["causal", "full"]

    layout = layout_from_split_info(SplitInfo(), sample_lens=[8])

    assert layout.meta.ar_lens == (3,)
    assert layout.meta.dm_lens == (5,)


def test_layout_from_split_info_rejects_a_partial_object():
    with pytest.raises(TypeError, match="does not carry"):
        layout_from_split_info({"split_lens": [3, 5], "attn_modes": ["causal", "full"]})


@pytest.mark.parametrize(
    "split_lens, attn_modes, sample_lens, message",
    [
        ([3, 5], ["causal"], [8], "same number of splits"),
        ([3, 5], ["causal", "full"], [9], "sum to 8"),
        ([3, 5], ["causal", "causal"], [8], "more than one 'causal' split"),
        ([3, 5], ["full", "causal"], [8], "before its causal split"),
        ([3, 5], ["causal", "other"], [8], "Unsupported attention mode"),
        ([3, 5], ["causal", "full"], [8, 0], "Sample 1 contributes no token"),
        ([-1, 5], ["causal", "full"], [4], "split length must be non-negative"),
        ([3, 5], ["causal", "full"], [-1], "sample length must be non-negative"),
        ([3, 5], ["causal", "full"], [4, 4], "splits add up to 8"),
        ([], [], [], "at least one sample"),
    ],
)
def test_invalid_split_descriptions_are_rejected(split_lens, attn_modes, sample_lens, message):
    with pytest.raises(ValueError, match=message):
        layout_from_split_lengths(split_lens, attn_modes, sample_lens)


def test_layout_rejects_a_negative_pad_length():
    with pytest.raises(ValueError, match="pad_tokens must be non-negative"):
        layout_from_split_lengths([3, 5], ["causal", "full"], [8], pad_tokens=-1)


def test_split_tensor_rejects_the_wrong_number_of_rows():
    layout = layout_from_split_lengths([3, 5], ["causal", "full"], [8], pad_tokens=2)

    with pytest.raises(ValueError, match="packed holds 9 rows"):
        layout.split_tensor(torch.zeros(9, HEADS, HEAD_DIM))


def test_merge_outputs_rejects_blocks_of_the_wrong_length():
    layout = layout_from_split_lengths([3, 5], ["causal", "full"], [8])

    with pytest.raises(ValueError, match="out_ar holds 2 tokens"):
        layout.merge_outputs(torch.zeros(2, HEADS, HEAD_DIM), torch.zeros(5, HEADS, HEAD_DIM))
    with pytest.raises(ValueError, match="out_dm holds 4 tokens"):
        layout.merge_outputs(torch.zeros(3, HEADS, HEAD_DIM), torch.zeros(4, HEADS, HEAD_DIM))
    with pytest.raises(ValueError, match="agree on the trailing dimensions"):
        layout.merge_outputs(torch.zeros(3, HEADS, HEAD_DIM), torch.zeros(5, HEADS, HEAD_DIM + 1))


def test_layout_is_frozen():
    layout = layout_from_split_lengths([3, 5], ["causal", "full"], [8])

    assert isinstance(layout, CosmosTwoWayLayout)
    with pytest.raises(AttributeError):
        layout.pad_tokens = 4


def _sample(und_len, dm_len, seed, sample_id=None):
    """A packer-shaped sample with one AR index field and the diffusion fields the transformer consumes."""
    generator = torch.Generator().manual_seed(seed)
    sequence_length = und_len + dm_len
    sample = dict(
        und_len=und_len,
        sequence_length=sequence_length,
        input_ids=torch.randint(0, 100, (und_len,), generator=generator),
        text_indexes=torch.arange(und_len),
        position_ids=torch.arange(3 * sequence_length, dtype=torch.long).reshape(3, sequence_length),
        vision_sequence_indexes=torch.arange(und_len, sequence_length),
        vision_mse_loss_indexes=torch.arange(und_len, sequence_length),
        vision_timesteps=torch.full((dm_len,), float(seed)),
        vision_tokens=[torch.randn(1, 2, 2, 2, 2, generator=generator)],
        vision_token_shapes=[(1, 2, 2)],
        vision_noisy_frame_indexes=[torch.arange(1)],
        vision_domain_ids=[torch.tensor(0)],
    )
    if sample_id is not None:
        sample["sample_id"] = sample_id
    return sample


def test_pack_cosmos3_samples_rewrites_every_global_index():
    batch = pack_cosmos3_samples([_sample(2, 3, seed=1, sample_id="a"), _sample(1, 4, seed=2, sample_id="b")])

    assert batch.meta.ar_lens == (2, 1)
    assert batch.meta.dm_lens == (3, 4)
    assert batch.sample_ids == ("a", "b")
    assert batch.model_inputs["und_len"] == 3
    assert batch.model_inputs["sequence_length"] == 10

    # AR block first: sample b's text indexes start after sample a's two AR tokens.
    assert batch.model_inputs["text_indexes"].tolist() == [0, 1, 2]
    # DM block second: sample a's DM starts at 3, sample b's at 6. Local DM index 2 in a -> 3 + (2 - 2) = 3;
    # b's local DM index 1 -> 6 + (1 - 1) = 6.
    assert batch.model_inputs["vision_sequence_indexes"].tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert batch.model_inputs["vision_mse_loss_indexes"].tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert batch.model_inputs["position_ids"].shape == (3, 10)
    assert batch.model_inputs["vision_timesteps"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert len(batch.model_inputs["vision_tokens"]) == 2
    assert batch.model_inputs["vision_domain_ids"][0].item() == 0


def test_pack_cosmos3_samples_counts_supervised_tokens_and_padding():
    batch = pack_cosmos3_samples(
        [_sample(2, 3, seed=3), _sample(1, 4, seed=4)],
        pad_tokens=2,
    )

    assert batch.layout.pad_tokens == 2
    assert batch.layout.sequence_length == 12
    assert batch.model_inputs["sequence_length"] == 10
    assert batch.supervised_tokens["vision_mse_loss_indexes"] == (3, 4)
    assert batch.supervised_tokens["total"] == (3, 4)
    assert batch.total_supervised_tokens == 7


def test_pack_cosmos3_samples_keeps_each_sample_position_ids():
    first = _sample(2, 3, seed=5)
    second = _sample(1, 4, seed=6)
    batch = pack_cosmos3_samples([first, second])

    packed = batch.model_inputs["position_ids"]
    assert torch.equal(packed[:, :2], first["position_ids"][:, :2])
    assert torch.equal(packed[:, 3:6], first["position_ids"][:, 2:])
    assert torch.equal(packed[:, 2:3], second["position_ids"][:, :1])
    assert torch.equal(packed[:, 6:], second["position_ids"][:, 1:])


def test_pack_cosmos3_samples_rejects_unknown_and_mismatched_fields():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 sample field"):
        pack_cosmos3_samples([_sample(2, 3, seed=7) | {"mystery": torch.zeros(1)}])

    missing_field = _sample(1, 4, seed=9)
    del missing_field["text_indexes"]
    with pytest.raises(ValueError, match="same packer fields"):
        pack_cosmos3_samples([_sample(2, 3, seed=8), missing_field])


def test_pack_cosmos3_samples_rejects_an_invalid_length():
    with pytest.raises(ValueError, match="und_len must be in"):
        pack_cosmos3_samples([_sample(2, 3, seed=10) | {"und_len": 6}])


def test_token_mean_loss_scale_weights_ranks_by_supervised_tokens():
    assert token_mean_loss_scale(10, 40, 2) == 0.5
    assert token_mean_loss_scale(30, 40, 2) == 1.5
    assert token_mean_loss_scale(0, 40, 2) == 0.0

    with pytest.raises(ValueError, match="no supervised token"):
        token_mean_loss_scale(0, 0, 2)
    with pytest.raises(ValueError, match="positive"):
        token_mean_loss_scale(1, 2, 0)


def test_all_reduce_supervised_tokens_uses_an_int64_collective(monkeypatch):
    """HCCL cannot all-reduce float64 and a CPU tensor has no HCCL backend: the temporary must be int64 and live
    on the requested device."""
    import torch.distributed as dist

    seen = []

    def fake_all_reduce(tensor, group=None):
        seen.append((tensor.dtype, tensor.device.type, int(tensor.item())))
        tensor.add_(41)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    total = all_reduce_supervised_tokens(7, device=torch.device("cpu"))

    assert total == 48
    assert seen == [(torch.int64, "cpu", 7)]
