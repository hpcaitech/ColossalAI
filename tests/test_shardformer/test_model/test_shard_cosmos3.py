"""Single-process integration test for the Cosmos 3 Booster/plugin wiring.

The real run uses two to eight accelerators. This test runs the same `HybridParallelPlugin + Cosmos3OmniPolicy`
configuration on one CPU rank with Gloo, so the parts that do not depend on a GPU -- policy installation, packed
metadata binding, backward through the plugin and an optimizer step -- are exercised in ordinary CI. It does not
replace the two/eight-device numerical and checkpoint checks.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator, set_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer.layer import TwoWayPackedMeta
from colossalai.shardformer.modeling.cosmos3 import (
    ColossalTwoWayAttnProcessor,
    ColossalTwoWayCPAttnProcessor,
    clear_cosmos3_packed_meta,
    set_cosmos3_packed_meta,
)
from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

HIDDEN = 16
HEAD_DIM = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2


class Cosmos3PackedMoTAttention(torch.nn.Module):
    """Minimal stand-in with the attribute contract the Cosmos 3 policy and processor use."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)

        def linear(in_features, out_features):
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
        self.k_norm_und_for_gen = None
        self.add_q_proj = linear(HIDDEN, NUM_HEADS * HEAD_DIM)
        self.add_k_proj = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.add_v_proj = linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM)
        self.to_add_out = linear(NUM_HEADS * HEAD_DIM, HIDDEN)
        self.norm_added_q = torch.nn.Identity()
        self.norm_added_k = torch.nn.Identity()
        self.processor = None
        self.set_processor(ColossalTwoWayAttnProcessor())

    def set_processor(self, processor) -> None:
        self.processor = processor

    def forward(self, und_seq, gen_seq, rotary_emb):
        return self.processor(self, und_seq, gen_seq, rotary_emb)


class Cosmos3OmniTransformer(torch.nn.Module):
    _cp_shard_fn = None
    _cp_gather_fn = None

    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            hidden_size=HIDDEN,
        )
        self.layers = torch.nn.ModuleList([Cosmos3PackedMoTAttention(seed=index) for index in range(num_layers)])
        # Exercise the production policy's timestep pair through the real Booster, too.
        self.time_proj = torch.nn.Identity()
        self.time_embedder = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, und_seq, gen_seq, rotary_emb):
        features = self.time_proj(torch.ones_like(gen_seq[:1]))
        gen_seq = gen_seq + self.time_embedder(features)
        if self._cp_shard_fn is not None:
            und_seq, gen_seq, rotary_emb = self._cp_shard_fn(und_seq, gen_seq, rotary_emb)
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq, rotary_emb)
        if self._cp_gather_fn is not None:
            und_seq, gen_seq = self._cp_gather_fn(und_seq, gen_seq)
        return und_seq, gen_seq


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    und_seq = torch.randn(7, HIDDEN, generator=generator, requires_grad=True)
    gen_seq = torch.randn(7, HIDDEN, generator=generator, requires_grad=True)
    angles_und = torch.randn(7, HEAD_DIM, generator=generator)
    angles_gen = torch.randn(7, HEAD_DIM, generator=generator)
    rotary = (angles_und.cos(), angles_und.sin(), angles_gen.cos(), angles_gen.sin())
    return und_seq, gen_seq, rotary


def _processors(model):
    return [module.processor for module in model.modules() if module.__class__.__name__ == "Cosmos3PackedMoTAttention"]


def test_hybrid_plugin_installs_the_processor_updates_parameters_and_reloads(tmp_path):
    previous_accelerator = get_accelerator()
    set_accelerator("cpu")
    rendezvous = Path(tmp_path, "rendezvous")
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        policy = Cosmos3OmniPolicy()
        plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            sp_size=1,
            precision="fp32",
            zero_stage=0,
            custom_policy=policy,
            overlap_communication=False,
            max_norm=0.0,
        )
        model = Cosmos3OmniTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        booster = Booster(plugin=plugin)
        model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)

        meta = TwoWayPackedMeta([4, 3], [2, 5])
        assert set_cosmos3_packed_meta(model, meta, strict=True) == 2

        und_seq, gen_seq, rotary = _inputs()
        outputs = model(und_seq, gen_seq, rotary)
        loss = outputs[0].square().mean() + outputs[1].square().mean()
        booster.backward(loss, optimizer)
        optimizer.step()

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in trainable)

        # The processor is a plain object, not a state_dict entry: reloading model weights into a freshly boosted
        # model must still install it, or packed metadata would be silently ignored after a resume.
        save_dir = Path(tmp_path, "model")
        booster.save_model(model, str(save_dir), shard=True, use_safetensors=True)

        reloaded = Cosmos3OmniTransformer()
        reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
        reloaded, reloaded_optimizer, _, _, _ = booster.boost(model=reloaded, optimizer=reloaded_optimizer)
        booster.load_model(reloaded, str(save_dir))
        assert set_cosmos3_packed_meta(reloaded, meta, strict=True) == 2

        with torch.no_grad():
            reloaded_outputs = reloaded(*(tensor.detach() for tensor in (und_seq, gen_seq)), rotary)
            original_outputs = model(*(tensor.detach() for tensor in (und_seq, gen_seq)), rotary)
        torch.testing.assert_close(reloaded_outputs[0], original_outputs[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reloaded_outputs[1], original_outputs[1], rtol=1e-5, atol=1e-6)

        clear_cosmos3_packed_meta(model, strict=True)
        clear_cosmos3_packed_meta(reloaded, strict=True)
        assert all(processor.packed_meta is None for processor in _processors(model))
        assert all(processor.packed_meta is None for processor in _processors(reloaded))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        set_accelerator(previous_accelerator)


def test_context_parallel_policy_preserves_preset_pack_metadata(tmp_path):
    previous_accelerator = get_accelerator()
    set_accelerator("cpu")
    rendezvous = Path(tmp_path, "rendezvous")
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        meta = TwoWayPackedMeta([4, 3], [2, 5])
        policy = Cosmos3OmniPolicy()
        policy.set_packed_meta(meta)
        plugin = HybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            sp_size=1,
            enable_sequence_parallelism=True,
            # Leave the mode unset: the plugin resolves it to all_to_all and has to pass that resolved value into
            # ShardConfig, otherwise the policy would see split_gather and reject the run.
            precision="fp32",
            zero_stage=0,
            custom_policy=policy,
            overlap_communication=False,
            max_norm=0.0,
        )
        model = Cosmos3OmniTransformer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        booster = Booster(plugin=plugin)
        model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)

        processors = _processors(model)
        assert len(processors) == 2
        assert all(isinstance(processor, ColossalTwoWayCPAttnProcessor) for processor in processors)
        assert all(processor.packed_meta is meta for processor in processors)

        und_seq, gen_seq, rotary = _inputs()
        outputs = model(und_seq, gen_seq, rotary)
        loss = outputs[0].square().mean() + outputs[1].square().mean()
        booster.backward(loss, optimizer)
        optimizer.step()
        assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        set_accelerator(previous_accelerator)


def test_policy_rejects_pipeline_and_non_ulysses_sequence_parallelism():
    from colossalai.shardformer import ShardConfig

    policy = Cosmos3OmniPolicy()
    policy.set_model(Cosmos3OmniTransformer())
    shard_config = ShardConfig(enable_tensor_parallelism=False)
    policy.set_shard_config(shard_config)
    assert policy.config_sanity_check is not None

    # The policy's explicit boundary is what keeps a stage split from silently cutting the two-way block.
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        policy.shard_config.pipeline_stage_manager = SimpleNamespace()
        policy.config_sanity_check()

    policy.shard_config.pipeline_stage_manager = None
    policy.shard_config.enable_sequence_parallelism = True
    policy.shard_config.sequence_parallelism_mode = "split_gather"
    with pytest.raises(NotImplementedError, match="all_to_all"):
        policy.config_sanity_check()

    policy.shard_config.sequence_parallelism_mode = "all_to_all"
    policy.config_sanity_check()
