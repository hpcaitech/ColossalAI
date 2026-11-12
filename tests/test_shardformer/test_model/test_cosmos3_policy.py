"""Tests for the Cosmos 3 ShardFormer policy and its two-way attention processor.

The real model is a ``diffusers`` class. These tests use stand-ins that carry the same class names and the same
attribute contract as the diffusers modules, so they exercise exactly what the policy and the processor touch -- the
string-keyed module-policy entries, ``set_processor``, and the per-pathway projection/norm/RoPE prologue -- without
making the ColossalAI test suite depend on a diffusers release that ships Cosmos 3.

The processor is checked against an independent per-sample implementation built on
:func:`torch.nn.functional.scaled_dot_product_attention`; the policy is driven through
:class:`~colossalai.shardformer.ShardFormer`, i.e. the same rewrite pass ``HybridParallelPlugin`` runs.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row, TwoWayPackedMeta
from colossalai.shardformer.modeling.cosmos3 import (
    ColossalTwoWayAttnProcessor,
    ColossalTwoWayCPAttnProcessor,
    Cosmos3ContextParallel,
    clear_cosmos3_packed_meta,
    set_cosmos3_packed_meta,
)
from colossalai.shardformer.modeling.cosmos_mot import layout_from_split_lengths
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription
from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

HIDDEN = 16
HEAD_DIM = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2


class _Halve(torch.nn.Module):
    """Stand-in for ``k_norm_und_for_gen``: a per-head map the generation pathway applies to the AR keys."""

    def forward(self, x):
        return x * 0.5


class Cosmos3PackedMoTAttention(torch.nn.Module):
    """Stand-in with the attribute contract of the diffusers attention module, under the diffusers class name.

    The policy keys its module policy on that name, so keeping it here is what lets the test drive
    :class:`~colossalai.shardformer.ShardFormer` over a stand-in.
    """

    def __init__(self, use_und_k_norm_for_gen: bool = False, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def linear(in_features: int, out_features: int) -> torch.nn.Linear:
            layer = torch.nn.Linear(in_features, out_features)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(out_features, in_features, generator=generator))
                layer.bias.copy_(torch.randn(out_features, generator=generator))
            return layer

        self.hidden_size = HIDDEN
        self.head_dim = HEAD_DIM
        self.num_attention_heads = NUM_HEADS
        self.num_key_value_heads = NUM_KV_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS

        self.to_q = linear(HIDDEN, NUM_HEADS * HEAD_DIM)
        self.to_k = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.to_v = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.to_out = linear(NUM_HEADS * HEAD_DIM, HIDDEN)
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        self.k_norm_und_for_gen = _Halve() if use_und_k_norm_for_gen else None

        self.add_q_proj = linear(HIDDEN, NUM_HEADS * HEAD_DIM)
        self.add_k_proj = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.add_v_proj = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.to_add_out = linear(NUM_HEADS * HEAD_DIM, HIDDEN)
        self.norm_added_q = torch.nn.Identity()
        self.norm_added_k = torch.nn.Identity()

        self.processor = None
        self.set_processor(ColossalTwoWayAttnProcessor())

    def set_processor(self, processor) -> None:
        # Same contract as `AttentionModuleMixin.set_processor` for a processor that is not an `nn.Module`.
        self.processor = processor

    def forward(self, und_seq, gen_seq, rotary_emb):
        return self.processor(self, und_seq, gen_seq, rotary_emb)


class Cosmos3OmniTransformer(torch.nn.Module):
    """Stand-in for the diffusers transformer: a config plus a stack of attention modules."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.time_embedder = torch.nn.Identity()
        self.time_proj = torch.nn.Identity()
        self.config = SimpleNamespace(
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            hidden_size=HIDDEN,
        )
        self.layers = torch.nn.ModuleList([Cosmos3PackedMoTAttention(seed=index) for index in range(num_layers)])

    def forward(self, und_seq, gen_seq, rotary_emb):
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        return und_seq, gen_seq


def _rotary(length: int, seed: int, dtype: torch.dtype = torch.float64):
    """Cos/sin tables shaped like the ones the transformer hands the processor, ``[tokens, head_dim]``."""
    angles = torch.randn(length, HEAD_DIM, generator=torch.Generator().manual_seed(seed), dtype=dtype)
    return angles.cos(), angles.sin()


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _sdpa(q, k, v, causal: bool):
    """Attention over one sample's ``[tokens, heads, head_dim]`` tensors, expanding the KV heads of a GQA group."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        is_causal=causal,
    )
    return out.squeeze(0).transpose(0, 1)


def _reference_pathways(attn, und_seq, gen_seq, rotary_emb, ar_lens, dm_lens):
    """Per-sample reference: AR causal inside its own sample, DM full over its own ``[AR_i; DM_i]``."""
    q_und = attn.norm_q(attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim))
    k_und = attn.norm_k(attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim))
    v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
    k_und_for_gen = attn.k_norm_und_for_gen(k_und) if attn.k_norm_und_for_gen is not None else k_und
    q_gen = attn.norm_added_q(attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim))
    k_gen = attn.norm_added_k(attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim))
    v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

    cos_und, sin_und, cos_gen, sin_gen = (tensor.unsqueeze(1) for tensor in rotary_emb)
    q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
    k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
    k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
    q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
    k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

    ar_outputs, dm_outputs = [], []
    ar_offset = dm_offset = 0
    for ar_len, dm_len in zip(ar_lens, dm_lens):
        if ar_len > 0:
            ar_outputs.append(
                _sdpa(
                    q_und[ar_offset : ar_offset + ar_len],
                    k_und[ar_offset : ar_offset + ar_len],
                    v_und[ar_offset : ar_offset + ar_len],
                    causal=True,
                )
            )
        if dm_len > 0:
            keys = torch.cat(
                (
                    k_und_for_gen[ar_offset : ar_offset + ar_len],
                    k_gen[dm_offset : dm_offset + dm_len],
                )
            )
            values = torch.cat(
                (
                    v_und[ar_offset : ar_offset + ar_len],
                    v_gen[dm_offset : dm_offset + dm_len],
                )
            )
            dm_outputs.append(_sdpa(q_gen[dm_offset : dm_offset + dm_len], keys, values, causal=False))
        ar_offset += ar_len
        dm_offset += dm_len

    out_ar = torch.cat(ar_outputs, dim=0)
    out_dm = torch.cat(dm_outputs, dim=0)
    return attn.to_out(out_ar.flatten(-2, -1)), attn.to_add_out(out_dm.flatten(-2, -1))


def _inputs(ar_lens, dm_lens, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total_ar, total_dm = sum(ar_lens), sum(dm_lens)
    und_seq = torch.randn(total_ar, HIDDEN, generator=generator, dtype=torch.float64)
    gen_seq = torch.randn(total_dm, HIDDEN, generator=generator, dtype=torch.float64)
    rotary_emb = _rotary(total_ar, seed + 1) + _rotary(total_dm, seed + 2)
    return und_seq, gen_seq, rotary_emb


@pytest.mark.parametrize(
    "ar_lens, dm_lens",
    [
        ([5], [3]),  # the batch-size-one pack the SFT scripts start from
        ([4, 3], [2, 5]),  # two samples, different lengths
        ([0, 3], [4, 4]),  # a generation-only sample in front of a mixed one
    ],
)
@pytest.mark.parametrize("use_und_k_norm_for_gen", [False, True])
def test_processor_matches_the_per_sample_reference(ar_lens, dm_lens, use_und_k_norm_for_gen):
    attn = Cosmos3PackedMoTAttention(use_und_k_norm_for_gen=use_und_k_norm_for_gen).double()
    und_seq, gen_seq, rotary_emb = _inputs(ar_lens, dm_lens)

    processor = ColossalTwoWayAttnProcessor(backend="naive", packed_meta=TwoWayPackedMeta(ar_lens, dm_lens))
    got = processor(attn, und_seq, gen_seq, rotary_emb)
    expected = _reference_pathways(attn, und_seq, gen_seq, rotary_emb, ar_lens, dm_lens)

    torch.testing.assert_close(got[0], expected[0], atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(got[1], expected[1], atol=1e-8, rtol=1e-6)


def test_processor_without_metadata_treats_the_pack_as_one_sample():
    attn = Cosmos3PackedMoTAttention().double()
    und_seq, gen_seq, rotary_emb = _inputs([5], [3])

    got = ColossalTwoWayAttnProcessor()(attn, und_seq, gen_seq, rotary_emb)
    expected = _reference_pathways(attn, und_seq, gen_seq, rotary_emb, [5], [3])

    torch.testing.assert_close(got[0], expected[0], atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(got[1], expected[1], atol=1e-8, rtol=1e-6)


def test_processor_keeps_packed_samples_independent():
    ar_lens, dm_lens = [4, 3], [2, 5]
    attn = Cosmos3PackedMoTAttention().double()
    und_seq, gen_seq, rotary_emb = _inputs(ar_lens, dm_lens)
    processor = ColossalTwoWayAttnProcessor(packed_meta=TwoWayPackedMeta(ar_lens, dm_lens))
    baseline = processor(attn, und_seq, gen_seq, rotary_emb)

    # Move only sample 0's AR tokens: sample 0 may change, sample 1 may not see it in either pathway.
    perturbed = und_seq.clone()
    perturbed[: ar_lens[0]] += 1.0
    changed = processor(attn, perturbed, gen_seq, rotary_emb)

    torch.testing.assert_close(changed[0][ar_lens[0] :], baseline[0][ar_lens[0] :], atol=1e-12, rtol=0)
    torch.testing.assert_close(changed[1][dm_lens[0] :], baseline[1][dm_lens[0] :], atol=1e-12, rtol=0)
    assert not torch.allclose(changed[0][: ar_lens[0]], baseline[0][: ar_lens[0]])


def test_processor_accepts_a_pack_layout_as_metadata():
    layout = layout_from_split_lengths([4, 2, 3, 5], ["causal", "full", "causal", "full"], [6, 8])

    processor = ColossalTwoWayAttnProcessor()
    processor.set_packed_meta(layout)
    assert processor.packed_meta is layout.meta

    processor.set_packed_meta(None)
    assert processor.packed_meta is None

    with pytest.raises(TypeError, match="TwoWayPackedMeta"):
        processor.set_packed_meta([4, 3])


def test_processor_rejects_metadata_from_another_pack():
    attn = Cosmos3PackedMoTAttention().double()
    processor = ColossalTwoWayAttnProcessor(packed_meta=TwoWayPackedMeta([4, 3], [2, 5]))
    und_seq, gen_seq, rotary_emb = _inputs([2, 3], [2, 5])

    with pytest.raises(ValueError, match="same pack"):
        processor(attn, und_seq, gen_seq, rotary_emb)


def test_context_parallel_processor_rejects_metadata_from_another_pack_before_exchanging():
    attn = Cosmos3PackedMoTAttention().double()
    processor = ColossalTwoWayCPAttnProcessor(packed_meta=TwoWayPackedMeta([2, 1], [1, 2]))
    und_seq, gen_seq, rotary_emb = _inputs([4], [3])
    processor.set_sequence_lengths(und_seq.shape[0], gen_seq.shape[0])

    with pytest.raises(ValueError, match="CP shard seam carries 4 AR and 3 DM tokens"):
        processor(attn, und_seq, gen_seq, rotary_emb)


def test_context_parallel_processor_exchange_is_a_noop_at_cp_size_one():
    processor = ColossalTwoWayCPAttnProcessor(process_group=None)
    x = torch.randn(3, NUM_HEADS, HEAD_DIM)

    assert processor._exchange(x) is x
    assert processor._exchange_back(x) is x


def test_context_parallel_processor_handles_an_empty_pathway():
    processor = ColossalTwoWayCPAttnProcessor(process_group=None)
    empty = torch.empty(0, NUM_HEADS, HEAD_DIM)

    assert processor._exchange(empty).shape == empty.shape
    assert processor._exchange_back(empty).shape == empty.shape


def test_context_parallel_helper_delegates_metadata_and_carries_options():
    helper = Cosmos3ContextParallel(process_group=None, scale=0.25, dropout_p=0.0)
    meta = TwoWayPackedMeta([4, 3], [2, 5])

    assert helper.packed_meta is None
    helper.set_packed_meta(meta)
    assert helper.packed_meta is meta
    assert helper.processor.scale == 0.25
    assert helper.processor.dropout_p == 0.0

    helper.clear_packed_meta()
    assert helper.packed_meta is None


def test_processor_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        ColossalTwoWayAttnProcessor(backend="flash")


class _ProbePolicy(Cosmos3OmniPolicy):
    """Policy that also rewrites the attention module, to prove the processor install happens after the rewrites."""

    def module_policy(self):
        policy = super().module_policy()
        policy["Cosmos3PackedMoTAttention"] = ModulePolicyDescription(attribute_replacement={"processor": None})
        return policy


def test_policy_installs_the_processor_after_the_module_rewrites():
    model = Cosmos3OmniTransformer()
    policy = _ProbePolicy()

    ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model, policy=policy)

    processors = [
        module.processor for module in model.modules() if module.__class__.__name__ == "Cosmos3PackedMoTAttention"
    ]
    assert len(processors) == 2
    assert all(processor is policy.processor for processor in processors)


def test_policy_postprocess_rejects_a_model_without_cosmos3_attention():
    with pytest.raises(RuntimeError, match="found no Cosmos3PackedMoTAttention"):
        ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(
            torch.nn.Linear(2, 2), policy=Cosmos3OmniPolicy()
        )


def _stand_in_shard_config(**overrides):
    """A ``ShardConfig``-shaped object for the checks that must not start a process group."""
    values = dict(
        enable_tensor_parallelism=False,
        tensor_parallel_size=1,
        enable_sequence_parallelism=False,
        fp8_communication=False,
        pipeline_stage_manager=None,
        gradient_checkpoint_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _policy_with(shard_config):
    policy = Cosmos3OmniPolicy()
    policy.set_model(Cosmos3OmniTransformer())
    policy.set_shard_config(shard_config)
    return policy


def test_policy_rejects_pipeline_parallelism():
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        _policy_with(_stand_in_shard_config(pipeline_stage_manager=object()))


def test_policy_rejects_sequence_parallelism_that_is_not_ulysses():
    # Only the all-to-all (Ulysses) flavour keeps whole per-sample blocks on the attention side; the split-gather
    # and ring flavours cut inside a block, which the two-way kernel cannot represent.
    with pytest.raises(NotImplementedError, match="all_to_all"):
        _policy_with(
            _stand_in_shard_config(
                enable_sequence_parallelism=True,
                sequence_parallelism_mode="split_gather",
                sequence_parallel_size=2,
            )
        )


def test_policy_rejects_context_parallelism_mixed_with_tensor_parallelism():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        _policy_with(
            _stand_in_shard_config(
                enable_sequence_parallelism=True,
                sequence_parallelism_mode="all_to_all",
                sequence_parallel_size=2,
                enable_tensor_parallelism=True,
                tensor_parallel_size=2,
            )
        )


def test_policy_rejects_a_context_parallel_group_that_cannot_split_the_heads():
    with pytest.raises(ValueError, match="divisible by the context parallel size"):
        _policy_with(
            _stand_in_shard_config(
                enable_sequence_parallelism=True,
                sequence_parallelism_mode="all_to_all",
                sequence_parallel_size=3,
            )
        )


def test_context_parallel_policy_rewrites_no_module():
    # Ulysses CP only swaps the processor and wires the model's seams, both in postprocess().
    assert (
        _policy_with(
            _stand_in_shard_config(
                enable_sequence_parallelism=True,
                sequence_parallelism_mode="all_to_all",
                sequence_parallel_size=2,
            )
        ).module_policy()
        == {}
    )


def test_policy_rejects_fp8_context_parallelism():
    with pytest.raises(NotImplementedError, match="fp8 communication"):
        _policy_with(
            _stand_in_shard_config(
                enable_sequence_parallelism=True,
                sequence_parallelism_mode="all_to_all",
                sequence_parallel_size=2,
                fp8_communication=True,
            )
        )


def test_policy_rejects_shardformer_gradient_checkpoint_config():
    marker = object()
    with pytest.raises(NotImplementedError, match="gradient_checkpoint_config"):
        _policy_with(_stand_in_shard_config(gradient_checkpoint_config=marker))


def test_policy_rejects_shardformer_optimizations_it_does_not_implement():
    with pytest.raises(NotImplementedError, match="enable_flash_attention"):
        _policy_with(_stand_in_shard_config(enable_flash_attention=True))


def test_zero_shaped_policy_rewrites_no_module():
    # What a `HybridParallelPlugin(tp_size=1, zero_stage=2)` run asks for: the processor install, nothing else.
    assert _policy_with(_stand_in_shard_config()).module_policy() == {}


def test_tensor_parallel_policy_splits_head_counts_and_projections():
    module_policy = _policy_with(
        _stand_in_shard_config(enable_tensor_parallelism=True, tensor_parallel_size=2)
    ).module_policy()

    attention = module_policy["Cosmos3PackedMoTAttention"]
    assert attention.attribute_replacement == {
        "num_attention_heads": NUM_HEADS // 2,
        "num_key_value_heads": NUM_KV_HEADS // 2,
    }
    assert {description.suffix: description.target_module for description in attention.sub_module_replacement} == {
        "to_q": Linear1D_Col,
        "to_k": Linear1D_Col,
        "to_v": Linear1D_Col,
        "add_q_proj": Linear1D_Col,
        "add_k_proj": Linear1D_Col,
        "add_v_proj": Linear1D_Col,
        "to_out": Linear1D_Row,
        "to_add_out": Linear1D_Row,
    }

    mlp = module_policy["Cosmos3VLTextMLP"]
    assert {
        description.suffix: (description.target_module, description.ignore_if_not_exist)
        for description in mlp.sub_module_replacement
    } == {
        "gate_proj": (Linear1D_Col, True),
        "up_proj": (Linear1D_Col, False),
        "down_proj": (Linear1D_Row, False),
    }


def test_tensor_parallel_policy_checks_head_divisibility():
    with pytest.raises(ValueError, match="attention heads"):
        _policy_with(_stand_in_shard_config(enable_tensor_parallelism=True, tensor_parallel_size=3))

    with pytest.raises(ValueError, match="key/value heads"):
        _policy_with(_stand_in_shard_config(enable_tensor_parallelism=True, tensor_parallel_size=4))


def test_policy_passes_pack_metadata_to_the_processor():
    policy = Cosmos3OmniPolicy()
    meta = TwoWayPackedMeta([4, 3], [2, 5])

    policy.set_packed_meta(meta)
    assert policy.packed_meta is meta
    assert policy.processor.packed_meta is meta


def test_set_packed_meta_helper_reaches_a_wrapped_model():
    class _Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    model = _Wrapper(Cosmos3OmniTransformer(num_layers=3))
    meta = TwoWayPackedMeta([4, 3], [2, 5])

    assert set_cosmos3_packed_meta(model, meta) == 3
    for module in model.modules():
        if module.__class__.__name__ == "Cosmos3PackedMoTAttention":
            assert module.processor.packed_meta is meta

    assert set_cosmos3_packed_meta(model, None) == 3
    assert set_cosmos3_packed_meta(torch.nn.Linear(2, 2), meta) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 真模型：跑 diffusers 的 Cosmos3OmniTransformer 本体（没装 diffusers 就跳过）
# ─────────────────────────────────────────────────────────────────────────────
LATENT_CHANNEL = 4
LATENT_PATCH = 2
PATCH_LATENT_DIM = LATENT_CHANNEL * LATENT_PATCH**2
ACTION_DIM = 8
LATENT_HW = 4


def _tiny_cosmos3(use_und_k_norm_for_gen: bool = False):
    """A 2-layer float32 Cosmos 3 with tiny dimensions, built from the diffusers class itself."""
    diffusers = pytest.importorskip("diffusers")
    transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
    if transformer_cls is None:
        pytest.skip("this diffusers release does not ship Cosmos3OmniTransformer")

    torch.manual_seed(0)
    return transformer_cls(
        hidden_size=32,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=2,
        latent_channel=LATENT_CHANNEL,
        latent_patch_size=LATENT_PATCH,
        patch_latent_dim=PATCH_LATENT_DIM,
        action_dim=ACTION_DIM,
        action_gen=True,
        num_embodiment_domains=2,
        vocab_size=64,
        hidden_act="silu",
        qk_norm_for_text=not use_und_k_norm_for_gen,
        use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        rope_axes_dim=[4, 2, 2],
    ).eval()


def _test_device():
    """Run the real-model checks on the accelerator when one is present.

    diffusers' ``RMSNorm`` calls the Ascend fusion op whenever ``torch_npu`` is installed, and that op has no CPU
    kernel, so on an NPU host these checks have to run on the device even though the model is tiny. A CUDA host must
    be detected here too: otherwise a GPU job would silently exercise the CPU fallback and prove nothing about the
    device path.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    try:
        # Imported for its side effect: `torch.npu` only exists once torch_npu has been imported.
        importlib.import_module("torch_npu")
    except ImportError:
        return torch.device("cpu")
    if torch.npu.is_available():
        torch.npu.set_device(0)
        return torch.device("npu", 0)
    return torch.device("cpu")


def _to_device(pack, device):
    moved = {}
    for key, value in pack.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            moved[key] = [item.to(device) for item in value]
        else:
            moved[key] = value
    return moved


def _single_sample_pack(n_text: int, frames: int, seed: int, domain_id: int = 0, n_act: int = 2):
    """A synthetic ``[text | vision patches | action]`` sample with unit-spaced temporal coordinates."""
    generator = torch.Generator().manual_seed(seed)
    h_patches = LATENT_HW // LATENT_PATCH
    w_patches = LATENT_HW // LATENT_PATCH
    n_vis = frames * h_patches * w_patches
    sequence_length = n_text + n_vis + n_act
    vision_indexes = torch.arange(n_text, n_text + n_vis)
    action_indexes = torch.arange(n_text + n_vis, sequence_length)

    text_positions = torch.arange(n_text).expand(3, -1)
    frame_positions = torch.arange(frames).repeat_interleave(h_patches * w_patches) + n_text
    row_positions = torch.arange(h_patches).repeat_interleave(w_patches).repeat(frames)
    column_positions = torch.arange(w_patches).repeat(h_patches * frames)
    vision_positions = torch.stack((frame_positions, row_positions, column_positions))

    action_time_positions = torch.arange(n_act) + n_text + frames
    action_spatial_positions = torch.zeros(n_act, dtype=torch.long)
    action_positions = torch.stack((action_time_positions, action_spatial_positions, action_spatial_positions))
    position_ids = torch.cat((text_positions, vision_positions, action_positions), dim=1)

    return dict(
        input_ids=torch.randint(0, 64, (n_text,), generator=generator),
        text_indexes=torch.arange(n_text),
        position_ids=position_ids,
        und_len=n_text,
        sequence_length=sequence_length,
        vision_tokens=[torch.randn(1, LATENT_CHANNEL, frames, LATENT_HW, LATENT_HW, generator=generator)],
        vision_token_shapes=[(frames, h_patches, w_patches)],
        vision_sequence_indexes=vision_indexes,
        vision_mse_loss_indexes=vision_indexes,
        vision_timesteps=torch.full((n_vis,), 500.0),
        vision_noisy_frame_indexes=[torch.arange(frames)],
        action_tokens=[torch.randn(n_act, ACTION_DIM, generator=generator)],
        action_token_shapes=[(n_act, 1, 1)],
        action_sequence_indexes=action_indexes,
        action_mse_loss_indexes=action_indexes,
        action_timesteps=torch.full((n_act,), 500.0),
        action_noisy_frame_indexes=[torch.arange(n_act)],
        action_domain_ids=[torch.tensor(domain_id)],
    )


def test_single_sample_pack_uses_three_axis_mrope_positions():
    n_text, frames, n_act = 3, 2, 2
    pack = _single_sample_pack(n_text=n_text, frames=frames, seed=1, n_act=n_act)
    h_patches, w_patches = LATENT_HW // LATENT_PATCH, LATENT_HW // LATENT_PATCH

    text_positions = torch.arange(n_text).expand(3, -1)
    torch.testing.assert_close(pack["position_ids"][:, :n_text], text_positions)

    vision_positions = pack["position_ids"][:, pack["vision_sequence_indexes"]]
    patches_per_frame = h_patches * w_patches
    expected_frame_positions = torch.arange(frames).repeat_interleave(patches_per_frame) + n_text
    expected_row_positions = torch.arange(h_patches).repeat_interleave(w_patches).repeat(frames)
    expected_column_positions = torch.arange(w_patches).repeat(h_patches * frames)
    torch.testing.assert_close(
        vision_positions,
        torch.stack(
            (
                expected_frame_positions,
                expected_row_positions,
                expected_column_positions,
            )
        ),
    )

    action_positions = pack["position_ids"][:, pack["action_sequence_indexes"]]
    expected_action_times = torch.arange(n_act) + n_text + frames
    expected_action_spatial_positions = torch.zeros(n_act, dtype=torch.long)
    torch.testing.assert_close(
        action_positions,
        torch.stack(
            (
                expected_action_times,
                expected_action_spatial_positions,
                expected_action_spatial_positions,
            )
        ),
    )


def _merge_single_sample_packs(packs):
    """The documented multi-sample layout: one global AR block, then one global DM block.

    This is the packing contract ``TwoWayPackedMeta`` describes, written out independently of the training script so
    the test checks the contract rather than a copy of the script.
    """
    ar_lens = [int(pack["und_len"]) for pack in packs]
    dm_lens = [int(pack["sequence_length"]) - int(pack["und_len"]) for pack in packs]
    total_ar, total_dm = sum(ar_lens), sum(dm_lens)
    position_ids = torch.zeros(3, total_ar + total_dm, dtype=torch.long)

    merged = {key: [] for key in ("input_ids", "text_indexes", "vision_timesteps", "action_timesteps")}
    merged_indexes = {
        key: []
        for key in (
            "vision_sequence_indexes",
            "vision_mse_loss_indexes",
            "action_sequence_indexes",
            "action_mse_loss_indexes",
        )
    }
    per_item = {
        key: []
        for key in (
            "vision_tokens",
            "vision_token_shapes",
            "vision_noisy_frame_indexes",
            "action_tokens",
            "action_token_shapes",
            "action_noisy_frame_indexes",
            "action_domain_ids",
        )
    }

    ar_offset, dm_offset = 0, total_ar
    for pack, n_text, n_dm in zip(packs, ar_lens, dm_lens):
        merged["text_indexes"].append(torch.arange(ar_offset, ar_offset + n_text))
        position_ids[:, ar_offset : ar_offset + n_text] = pack["position_ids"][:, :n_text]
        position_ids[:, dm_offset : dm_offset + n_dm] = pack["position_ids"][:, n_text:]

        shift = dm_offset - n_text
        merged_indexes["vision_sequence_indexes"].append(pack["vision_sequence_indexes"] + shift)
        merged_indexes["vision_mse_loss_indexes"].append(pack["vision_mse_loss_indexes"] + shift)
        merged_indexes["action_sequence_indexes"].append(pack["action_sequence_indexes"] + shift)
        merged_indexes["action_mse_loss_indexes"].append(pack["action_mse_loss_indexes"] + shift)

        merged["input_ids"].append(pack["input_ids"])
        merged["vision_timesteps"].append(pack["vision_timesteps"])
        merged["action_timesteps"].append(pack["action_timesteps"])
        for key in per_item:
            per_item[key].extend(pack[key])

        ar_offset += n_text
        dm_offset += n_dm

    packed = {key: torch.cat(value) for key, value in {**merged, **merged_indexes}.items()}
    packed.update(per_item)
    packed.update(position_ids=position_ids, und_len=total_ar, sequence_length=total_ar + total_dm)
    return packed


@pytest.mark.parametrize("use_und_k_norm_for_gen", [False, True])
def test_real_model_gives_the_same_result_with_the_policy_processor(
    use_und_k_norm_for_gen,
):
    device = _test_device()
    model = _tiny_cosmos3(use_und_k_norm_for_gen).to(device)
    pack = _to_device(_single_sample_pack(n_text=3, frames=2, seed=1), device)

    with torch.no_grad():
        stock = model(**pack)
        ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model, policy=Cosmos3OmniPolicy())
        ours = model(**pack)

    torch.testing.assert_close(ours.sample[0], stock.sample[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ours.action[0], stock.action[0], atol=1e-4, rtol=1e-4)


def test_real_model_multi_sample_pack_matches_two_single_sample_runs():
    device = _test_device()
    model = _tiny_cosmos3().to(device)
    pack_a = _to_device(_single_sample_pack(n_text=3, frames=2, seed=1), device)
    pack_b = _to_device(_single_sample_pack(n_text=4, frames=3, seed=2, domain_id=1), device)

    with torch.no_grad():
        # Reference: the stock processor, one sample per forward.
        alone_a = model(**pack_a)
        alone_b = model(**pack_b)

        policy = Cosmos3OmniPolicy()
        ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model, policy=policy)
        dm_lens = [int(pack["sequence_length"]) - int(pack["und_len"]) for pack in (pack_a, pack_b)]
        policy.set_packed_meta(TwoWayPackedMeta([pack_a["und_len"], pack_b["und_len"]], dm_lens))
        packed = model(**_to_device(_merge_single_sample_packs([pack_a, pack_b]), device))

    assert len(packed.sample) == 2 and len(packed.action) == 2
    torch.testing.assert_close(packed.sample[0], alone_a.sample[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(packed.sample[1], alone_b.sample[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(packed.action[0], alone_a.action[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(packed.action[1], alone_b.action[0], atol=1e-4, rtol=1e-4)


def test_packed_metadata_helper_is_strict_and_can_clear():
    model = Cosmos3OmniTransformer(num_layers=3)
    meta = TwoWayPackedMeta([4, 3], [2, 5])

    assert set_cosmos3_packed_meta(model, meta, strict=True) == 3
    for layer in model.layers:
        assert layer.processor.packed_meta is meta

    assert clear_cosmos3_packed_meta(model, strict=True) == 3
    for layer in model.layers:
        assert layer.processor.packed_meta is None

    with pytest.raises(RuntimeError, match="No Cosmos 3 attention processor"):
        set_cosmos3_packed_meta(torch.nn.Linear(2, 2), meta, strict=True)


def test_same_totals_with_different_boundaries_produce_different_attention():
    attn = Cosmos3PackedMoTAttention().double()
    und_seq, gen_seq, rotary_emb = _inputs([4, 3], [2, 5])
    split_across_two_samples = ColossalTwoWayAttnProcessor(packed_meta=TwoWayPackedMeta([4, 3], [2, 5]))
    one_sample = ColossalTwoWayAttnProcessor(packed_meta=TwoWayPackedMeta([7], [7]))

    split_output = split_across_two_samples(attn, und_seq, gen_seq, rotary_emb)
    one_output = one_sample(attn, und_seq, gen_seq, rotary_emb)

    # The processor cannot infer the boundary from the token totals, so both are accepted. The training entry point
    # must bind the metadata that belongs to the pack; this assertion documents why a total-only check is not enough.
    assert not torch.allclose(split_output[0], one_output[0])
    assert not torch.allclose(split_output[1], one_output[1])


class _CheckpointedCosmos3OmniTransformer(Cosmos3OmniTransformer):
    """Stand-in that wraps every decoder layer in non-reentrant activation checkpointing."""

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__(num_layers=num_layers)
        self.gradient_checkpointing = True

    def forward(self, und_seq, gen_seq, rotary_emb):
        for layer in self.layers:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                und_seq, gen_seq = torch.utils.checkpoint.checkpoint(
                    layer, und_seq, gen_seq, rotary_emb, use_reentrant=False
                )
            else:
                und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        return und_seq, gen_seq


def test_activation_checkpointing_recomputes_with_the_same_pack_metadata():
    import copy

    torch.manual_seed(0)
    reference_model = _CheckpointedCosmos3OmniTransformer(num_layers=2).double()
    reference_model.gradient_checkpointing = False
    checkpointed_model = copy.deepcopy(reference_model)
    checkpointed_model.gradient_checkpointing = True
    meta = TwoWayPackedMeta([4, 3], [2, 5])
    set_cosmos3_packed_meta(reference_model, meta, strict=True)
    set_cosmos3_packed_meta(checkpointed_model, meta, strict=True)

    und_seq, gen_seq, rotary_emb = _inputs([4, 3], [2, 5], seed=40)
    reference_inputs = [
        und_seq.clone().requires_grad_(True),
        gen_seq.clone().requires_grad_(True),
        rotary_emb,
    ]
    checkpointed_inputs = [
        und_seq.clone().requires_grad_(True),
        gen_seq.clone().requires_grad_(True),
        rotary_emb,
    ]

    reference_und, reference_gen = reference_model(*reference_inputs)
    (reference_und.square().sum() + reference_gen.square().sum()).backward()
    checkpointed_und, checkpointed_gen = checkpointed_model(*checkpointed_inputs)
    (checkpointed_und.square().sum() + checkpointed_gen.square().sum()).backward()

    torch.testing.assert_close(checkpointed_und, reference_und, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(checkpointed_gen, reference_gen, rtol=1e-12, atol=1e-12)
    for (name, reference), (checkpointed_name, actual) in zip(
        reference_model.named_parameters(), checkpointed_model.named_parameters()
    ):
        assert name == checkpointed_name
        assert reference.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-10, atol=1e-12)


def test_processor_accepts_the_cuda_backend_label(monkeypatch):
    monkeypatch.setattr(
        "colossalai.shardformer.modeling.cosmos3._load_cuda_varlen_attention",
        lambda: None,
    )
    processor = ColossalTwoWayAttnProcessor(backend="cuda")
    assert processor.backend == "cuda"

    with pytest.raises(ValueError, match="backend"):
        ColossalTwoWayAttnProcessor(backend="sdpa")


def _packed_real_model_reference(model, packs, device):
    """Run each sample through the stock processor and keep the per-sample outputs."""
    references = []
    with torch.no_grad():
        for pack in packs:
            references.append(model(**_to_device(pack, device)))
    return references


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused CUDA backend needs a CUDA device")
def test_real_model_packed_forward_with_the_cuda_backend_matches_stock_runs():
    device = torch.device("cuda", torch.cuda.current_device())
    model = _tiny_cosmos3().to(device=device, dtype=torch.bfloat16)
    pack_a = _to_device(_single_sample_pack(n_text=3, frames=2, seed=11), device)
    pack_b = _to_device(_single_sample_pack(n_text=4, frames=3, seed=12, domain_id=1), device)
    references = _packed_real_model_reference(model, [pack_a, pack_b], device)

    policy = Cosmos3OmniPolicy(backend="cuda")
    ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model, policy=policy)
    dm_lens = [int(pack["sequence_length"]) - int(pack["und_len"]) for pack in (pack_a, pack_b)]
    policy.set_packed_meta(TwoWayPackedMeta([pack_a["und_len"], pack_b["und_len"]], dm_lens))
    with torch.no_grad():
        packed = model(**_to_device(_merge_single_sample_packs([pack_a, pack_b]), device))

    for index, reference in enumerate(references):
        torch.testing.assert_close(packed.sample[index], reference.sample[0], atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(packed.action[index], reference.action[0], atol=5e-3, rtol=5e-3)
