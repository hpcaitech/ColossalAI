"""Ouro's repeated shared parameters through real 1/2-GPU Booster plugins."""

import json
from dataclasses import asdict
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from transformers import __version__ as transformers_version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.lazy import LazyInitContext, LazyTensor
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import judge
from tests.kit.numerics.judge import require_gradients
from tests.kit.numerics.scenarios import data_parallel, independent_impl, resume
from tests.kit.ouro import make_ouro, ouro_data, ouro_entry

RECIPES = (
    "ddp_fp32",
    "ddp_bf16",
    "zero1_fp32",
    "zero1_bf16",
    "zero2_fp32",
    "zero2_bf16",
    "zero2_bf16_offload",
    "gemini_bf16",
    "gemini_sharded",
    "gemini_offload",
    "fsdp_fp32",
    "fsdp_bf16",
)


def build_plugin(name):
    """Keep plugin arithmetic explicit; the shared tool independently audits it."""
    if name.startswith("ddp"):
        return TorchDDPPlugin()
    if name.startswith("zero"):
        return LowLevelZeroPlugin(
            stage=int(name[4]), precision=name.split("_")[1], cpu_offload=name.endswith("offload")
        )
    if name.startswith("gemini"):
        chunks = (
            {dist.get_world_size(): {"chunk_size": 32768, "keep_gathered": False}}
            if name in ("gemini_sharded", "gemini_offload")
            else None
        )
        return GeminiPlugin(
            precision="bf16",
            enable_gradient_accumulation=True,
            chunk_config_dict=chunks,
            offload_optim_frac=1.0 if name == "gemini_offload" else 0.0,
            offload_param_frac=1.0 if name == "gemini_offload" else 0.0,
        )
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return TorchFSDPPlugin(mixed_precision=mp if name.endswith("bf16") else None)


def mixed_precision(name):
    return "bf16" if name == "ddp_bf16" else None


def _save(report, path):
    path.write_text(json.dumps({"passed": report.passed, "entries": [asdict(e) for e in report.entries]}, indent=2))
    assert report.passed, report.format()


def _alignment(rank, world_size, port, key, recipe, directory):
    from pathlib import Path

    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1")
    try:
        entry = ouro_entry(key, loops=4, checkpoint=True)
        traces = independent_impl.training_traces(entry, build_plugin(recipe), mixed_precision(recipe), steps=3)
        for role, trace in zip(("test", "ref", "truth"), traces):
            trace.save(Path(directory) / f"{role}-{rank}.safetensors")
        test, ref, truth = traces
        for name in ("model.early_exit_gate.weight", "model.early_exit_gate.bias"):
            assert torch.isfinite(test[f"grad/param/{name}"]).all()
            assert test[f"grad/param/{name}"].abs().sum() > 0
        report = require_gradients(judge(*traces), test=test, ref=ref, truth=truth)
        _save(report, Path(directory) / f"alignment-{rank}.json")
    finally:
        dist.destroy_process_group()
        get_accelerator().empty_cache()


def _blank(rank, world_size, port, key, directory):
    from pathlib import Path

    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1")
    try:
        report = independent_impl.blank_control(ouro_entry(key, loops=4, checkpoint=True), steps=3)
        _save(report, Path(directory) / "blank.json")
    finally:
        dist.destroy_process_group()
        get_accelerator().empty_cache()


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
@rerun_if_address_is_in_use()
def test_ouro_reference_blank_control(key, tmp_path):
    assert torch.cuda.device_count() >= 1
    spawn(_blank, 1, key=key, directory=str(tmp_path))


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
@pytest.mark.parametrize("recipe", RECIPES)
@rerun_if_address_is_in_use()
def test_ouro_alignment(key, recipe, tmp_path):
    assert torch.cuda.device_count() >= 2, "Allocate two GPUs for Ouro L2"
    spawn(_alignment, 2, key=key, recipe=recipe, directory=str(tmp_path))


@pytest.mark.parametrize(
    "key,recipe",
    [
        ("ouro14", "ddp_bf16"),
        ("ouro14", "zero1_bf16"),
        ("ouro14", "zero2_bf16"),
        ("ouro14", "zero2_bf16_offload"),
        ("ouro14", "gemini_bf16"),
        ("ouro14", "gemini_sharded"),
        ("ouro14", "gemini_offload"),
        ("ouro14", "fsdp_bf16"),
        ("ouro26", "zero2_bf16"),
        ("ouro26", "gemini_sharded"),
    ],
)
@rerun_if_address_is_in_use()
def test_ouro_accumulation_and_resume(key, recipe, tmp_path):
    assert torch.cuda.device_count() >= 2, "Allocate two GPUs for Ouro L2"
    entry = ouro_entry(key, loops=4, checkpoint=True)
    plugin = partial(build_plugin, recipe)
    report = data_parallel.check(
        entry,
        plugin,
        mixed_precision(recipe),
        world_size=2,
        micro_batches=4,
        token_weighted=True,
        directory=str(tmp_path / "dp"),
    )
    _save(report, tmp_path / "dp.json")
    report = resume.check(entry, plugin, mixed_precision(recipe), world_size=2, directory=str(tmp_path / "resume"))
    _save(report, tmp_path / "resume.json")


def _lazy_step(rank, world_size, port, key, recipe):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1", seed=42)
    try:
        with LazyInitContext(default_device=get_accelerator().get_current_device()):
            model = make_ouro(key, loops=2, checkpoint=True, apply_policy=False)
        assert all(isinstance(p, LazyTensor) for p in model.parameters())
        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
        assert not any(isinstance(p, LazyTensor) for p in model.parameters())
        native = model
        plugin = build_plugin(recipe)
        optimizer = (
            HybridAdam(model.parameters(), lr=1e-3)
            if recipe.startswith("gemini")
            else torch.optim.AdamW(model.parameters(), lr=1e-3)
        )
        booster = Booster(plugin=plugin, mixed_precision=mixed_precision(recipe))
        model, optimizer, *_ = booster.boost(model, optimizer)
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ouro_data().items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        booster.backward(loss, optimizer)
        assert all(layer._ouro_calls > 2 for layer in native.model.layers), "No checkpoint recomputation"
        optimizer.step()
        optimizer.zero_grad()
        second = model(**batch).loss
        assert torch.isfinite(second) and not torch.equal(loss.detach(), second.detach()), "No training effect"
    finally:
        dist.destroy_process_group()
        get_accelerator().empty_cache()


@pytest.mark.parametrize("key", ["ouro14", "ouro26"])
@pytest.mark.parametrize(
    "recipe", ["ddp_bf16", "zero1_bf16", "zero2_bf16_offload", "gemini_sharded", "gemini_offload", "fsdp_bf16"]
)
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_ouro_lazy_checkpoint_step(key, recipe, world_size):
    assert torch.cuda.device_count() >= world_size
    spawn(_lazy_step, world_size, key=key, recipe=recipe)


pytestmark = pytest.mark.skipif(transformers_version != "5.17.0", reason="Ouro adapter requires Transformers 5.17.0")
