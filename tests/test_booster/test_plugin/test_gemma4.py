"""Real 1/2-device plugin steps; CPU tests live in tests/test_model_zoo/test_gemma4.py."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn
from tests.kit.model_zoo.transformers.gemma4 import gemma4_data, tiny_gemma4


def _snapshot(model, plugin_name):
    if plugin_name == "gemini":
        state = model.state_dict(only_rank_0=False)
    elif plugin_name == "fsdp":
        with FSDP.state_dict_type(
            model.unwrap(), StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            state = model.unwrap().state_dict()
    else:
        state = model.unwrap().state_dict()
    assert state, "No model state collected"
    return {n: p.detach().cpu().clone() for n, p in state.items()}


def _run(rank, world_size, port, variant, plugin_name, checkpoint, lazy):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", seed=42)
    accelerator = get_accelerator()
    try:
        if lazy:
            with LazyInitContext(default_device=accelerator.get_current_device()):
                model = tiny_gemma4(variant)
        else:
            model = tiny_gemma4(variant)
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if plugin_name.startswith("ddp"):
            plugin = TorchDDPPlugin()
        elif plugin_name.startswith("zero"):
            plugin = LowLevelZeroPlugin(
                stage=int(plugin_name[-1]), precision="bf16", cpu_offload=True, overlap_communication=False, max_norm=0
            )
        elif plugin_name == "gemini":
            plugin = GeminiPlugin(
                precision="bf16",
                placement_policy="static",
                initial_scale=1,
                min_chunk_size_m=1,
                enable_all_optimization=False,
            )
        else:
            plugin = TorchFSDPPlugin(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
                )
            )
        if lazy and plugin_name != "gemini":
            LazyInitContext.materialize(model)
        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
        optimizer = (
            HybridAdam(model.parameters(), lr=1e-3)
            if plugin_name == "gemini"
            else torch.optim.AdamW(model.parameters(), lr=1e-3)
        )
        booster = Booster(plugin=plugin, mixed_precision="bf16" if plugin_name == "ddp_bf16" else None)
        model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
        batch = {k: v.to(accelerator.get_current_device()) for k, v in gemma4_data().items()}
        before = _snapshot(model, plugin_name)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            booster.backward(loss, optimizer)
            optimizer.step()
        after = _snapshot(model, plugin_name)
        assert before.keys() == after.keys()
        assert all(torch.isfinite(p).all() for p in after.values())
        changed = [n for n, p in after.items() if not torch.equal(before[n], p)]
        assert changed, "No parameter changed"
        assert any("embed_tokens_per_layer" in n for n in changed), "No PLE update"
    finally:
        accelerator.empty_cache()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("variant", ["e2b", "e4b"])
@pytest.mark.parametrize("plugin", ["ddp", "ddp_bf16", "zero1", "zero2", "gemini", "fsdp"])
@pytest.mark.parametrize("checkpoint,lazy", [(False, False), (True, True)])
def test_gemma4_training(world_size, variant, plugin, checkpoint, lazy):
    # A missing allocation is not a successful or skipped model test.
    assert get_accelerator().device_count() >= world_size, f"Requires {world_size} allocated devices"
    spawn(_run, world_size, variant=variant, plugin_name=plugin, checkpoint=checkpoint, lazy=lazy)
