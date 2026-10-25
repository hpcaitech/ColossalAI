"""Small real-plugin checkpoint regression, requiring two guarded CUDA devices."""

import copy
import tempfile
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai._compat import is_transformers_v5
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchFSDPPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.safetensors import HAS_TENSORNVME


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right.to(left.device))
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), (left.keys(), right.keys())
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _run(rank, world_size, port, plugin_name):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", backend="nccl")
    torch.cuda.set_device(rank)
    torch.manual_seed(1234)
    if plugin_name == "fsdp":
        plugin = TorchFSDPPlugin()
    elif plugin_name == "zero":
        plugin = LowLevelZeroPlugin(stage=2, precision="bf16", initial_scale=1)
    elif plugin_name == "gemini":
        plugin = GeminiPlugin(precision="bf16", placement_policy="static", initial_scale=1)
    else:
        plugin = HybridParallelPlugin(tp_size=1, pp_size=1, zero_stage=1, precision="bf16", initial_scale=1)
    booster = Booster(plugin=plugin)
    if plugin_name == "hybrid":
        from transformers import LlamaConfig, LlamaModel

        model = LlamaModel(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                attn_implementation="eager",
            )
        ).cuda()
    else:
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).cuda()
    optimizer = (
        HybridAdam(model.parameters(), lr=0.001) if plugin_name == "gemini" else torch.optim.Adam(model.parameters())
    )
    model, optimizer, *_ = booster.boost(model, optimizer)
    if plugin_name == "hybrid":
        output = model(input_ids=torch.randint(0, 64, (2, 8), device="cuda")).last_hidden_state
    else:
        dtype = torch.float32 if plugin_name == "fsdp" else torch.bfloat16
        output = model(torch.rand(2, 8, device="cuda", dtype=dtype))
    booster.backward(output.float().square().mean(), optimizer)
    optimizer.step()
    optimizer.zero_grad()
    inner = optimizer.unwrap() if hasattr(optimizer, "unwrap") else optimizer.optim

    def snapshot():
        # Gemini's public checkpoint state adds PyTorch Adam defaults and
        # gathers partitioned states; local HybridAdam internals differ.
        state = optimizer.state_dict(only_rank_0=False) if plugin_name == "gemini" else inner.state_dict()
        return copy.deepcopy(state)

    with tempfile.TemporaryDirectory() if rank == 0 else nullcontext(None) as directory:
        directories = [directory]
        dist.broadcast_object_list(directories, src=0)
        for use_async in (False, True):
            for shard in (False, True):
                for low_mem in (True, False):
                    # Updating this changes schema size between saves and also
                    # verifies that parameter-group options survive each loader.
                    inner.param_groups[0]["checkpoint_note"] = "data" * (100 if low_mem else 1)
                    expected = snapshot()
                    assert expected["state"], "optimizer state must be nonempty"
                    path = directories[0] + f"/{use_async}_{shard}_{low_mem}"
                    if use_async and not shard:
                        path += ".safetensors"
                    booster.save_optimizer(optimizer, path, shard=shard, use_async=use_async, size_per_shard=1)
                    booster.checkpoint_io._sync_d2h()
                    booster.checkpoint_io._sync_io()
                    dist.barrier()
                    for state in inner.state.values():
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                value.fill_(-12)
                            elif type(value) is int:
                                state[key] = -12
                    booster.load_optimizer(optimizer, path, low_cpu_mem_mode=low_mem)
                    _equal(expected, snapshot())
                    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not HAS_TENSORNVME or torch.cuda.device_count() < 2, reason="Requires two guarded CUDA devices")
@pytest.mark.parametrize("plugin_name", ["fsdp", "zero", "gemini", "hybrid"])
@rerun_if_address_is_in_use()
def test_optimizer_format_plugins(plugin_name):
    spawn(_run, 2, plugin_name=plugin_name)


def _run_moe_migration(rank, world_size, port, config):
    from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
    from colossalai.checkpoint_io.optimizer_migration import migrate_optimizer_checkpoint
    from colossalai.checkpoint_io.utils import load_param_groups
    from colossalai.testing.random import seed_all
    from tests.test_shardformer.test_model.test_moe_optimizer_checkpoint import _state_snapshot
    from tests.test_shardformer.test_model.test_moe_optimizer_coverage import NUM_TOKENS, _build

    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", backend="nccl")
    torch.cuda.set_device(rank)
    model_name, ep_size = config
    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=1,
        ep_size=ep_size,
        zero_stage=1,
        precision="bf16",
        initial_scale=1,
        overlap_communication=False,
        find_unused_parameters=True,
    )
    booster = Booster(plugin=plugin)
    seed_all(1234)
    model = _build(model_name).to(torch.bfloat16).cuda()
    hidden = model.config.hidden_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, optimizer, *_ = booster.boost(model, optimizer)
    seed_all(4321 + dist.get_rank(plugin.dp_group))
    inputs = torch.rand(2, NUM_TOKENS // 2, hidden, device="cuda", dtype=torch.bfloat16)

    def step():
        loss = model(inputs_embeds=inputs).last_hidden_state.float().square().mean()
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()

    step()
    expected = _state_snapshot(optimizer)
    assert expected, "the optimizer must have real state before migration"
    with tempfile.TemporaryDirectory() if rank == 0 else nullcontext(None) as directory:
        directories = [directory]
        dist.broadcast_object_list(directories, src=0)
        root = directories[0]
        booster.save_optimizer(optimizer, root + "/old", shard=True)
        dist.barrier()
        if rank == 0:
            import json
            from pathlib import Path

            old = Path(root) / "old"
            index_path = next(old.glob("*.index.json"))
            index = json.loads(index_path.read_text())
            # The state shards already use the legacy PyTorch layout. Recreate
            # the old groups companion so both parts exercise real migration.
            groups = load_param_groups(old / index["metadata"]["param_groups"])
            torch.save(groups, old / "legacy_groups.bin")
            index["metadata"]["param_groups"] = "legacy_groups.bin"
            index["metadata"].pop("param_groups_format", None)
            index_path.write_text(json.dumps(index))
            migrate_optimizer_checkpoint(old, Path(root) / "new", trust_legacy_pickle=True)
        dist.barrier()
        inner = optimizer.unwrap() if hasattr(optimizer, "unwrap") else optimizer.optim
        for state in inner.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.fill_(-12)
        booster.load_optimizer(optimizer, root + "/new")
        _equal(expected, _state_snapshot(optimizer))
        dist.barrier()
        # Exercise a real resumed training step after the exact-state check.
        step()
    dist.destroy_process_group()


@pytest.mark.skipif(not is_transformers_v5(), reason="Fused MoE experts only exist in transformers v5")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two guarded CUDA devices")
@pytest.mark.parametrize("config", [("mixtral", 2), ("deepseek_v3", 2), ("mixtral_odd", 1)])
@rerun_if_address_is_in_use()
def test_moe_migrated_optimizer(config):
    spawn(_run_moe_migration, 2, config=config)
