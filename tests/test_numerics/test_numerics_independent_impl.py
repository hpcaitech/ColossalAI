import copy
from functools import partial

import pytest
import torch
from torch import nn

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo.registry import ModelAttribute
from tests.kit.numerics import NumericsSpec, Routers, Trace
from tests.kit.numerics.capture.recipes import Recipe, UnsupportedRecipe, recipe_for
from tests.kit.numerics.capture.training import prepare, run_reference
from tests.kit.numerics.scenarios.independent_impl import (
    _worse_of,
    blank_control,
    check_fwd_bwd,
    check_training,
    dtype_mismatches,
)

STEPS = 3
NO_ACCELERATOR = get_accelerator().name == "cpu"


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        # must be switched off by the training run: runs are compared in train mode
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return {"y": self.fc2(self.dropout(torch.relu(self.fc1(x))))}


class RenamedMLP(nn.Module):
    """An independent reference implementation of TinyMLP whose parameters have other names."""

    def __init__(self, drop_column: bool = False):
        super().__init__()
        self.inp = nn.Linear(64, 128)
        self.out = nn.Linear(128, 32)
        self.drop_column = drop_column

    def forward(self, x):
        h = torch.relu(self.inp(x))
        if self.drop_column:
            # a typical sharding bug: one column of the reduction dimension is dropped
            h = torch.cat([h[:, :63], torch.zeros_like(h[:, 63:64]), h[:, 64:]], dim=1)
        return {"y": self.out(h)}


PARAMS = [f"fc{i}.{p}" for i in (1, 2) for p in ("weight", "bias")]
TRACE_NAMES = (
    {"out/y", "grad/x"}
    | {f"grad/param/{p}" for p in PARAMS}
    | {f"param/{p}" for p in PARAMS}
    | {f"loss/{step}" for step in range(STEPS)}
)


def _rename(name):
    return name.replace("inp.", "fc1.").replace("out.", "fc2.")


def _data():
    return {"x": torch.randn(16, 64)}


def _entry(spec=NumericsSpec()):
    return (TinyMLP, _data, lambda out: out, None, ModelAttribute(numerics=spec))


BUGGY = NumericsSpec(reference_fn=lambda: RenamedMLP(drop_column=True), param_map=_rename)


def _launch(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)


def _check_aligned(rank, world_size, port, plugin_fn, mixed_precision):
    _launch(rank, world_size, port)
    for spec in (NumericsSpec(), NumericsSpec(reference_fn=RenamedMLP, param_map=_rename)):
        report = check_training(_entry(spec), plugin_fn(), mixed_precision, steps=STEPS)
        assert {e.name for e in report.entries} == TRACE_NAMES
        report.assert_pass()
        # a single device changes no arithmetic, so the plugin reproduces the reference exactly
        assert all(e.ratio == pytest.approx(1.0) for e in report.entries), report.format()


def _check_bug_detected(rank, world_size, port, plugin_fn, mixed_precision):
    _launch(rank, world_size, port)
    report = check_training(_entry(BUGGY), plugin_fn(), mixed_precision, steps=STEPS)
    failed = {e.name for e in report.failures}
    # the tensors a dropped column moves far beyond bf16 noise (ratio > 15); weights after one AdamW step are not
    # among them: the first update is about lr * sign(grad), insensitive to the size of a gradient error
    expected = {"out/y", "grad/param/fc2.weight", "loss/0", "loss/1", "loss/2"}
    assert expected <= failed, report.format()


def _check_blank_control(rank, world_size, port):
    _launch(rank, world_size, port)
    blank_control(_entry(), steps=STEPS)
    with pytest.raises(AssertionError, match="blank control"):
        blank_control(_entry(BUGGY))


@rerun_if_address_is_in_use()
def test_ddp_fp32_aligned():
    spawn(_check_aligned, 1, plugin_fn=TorchDDPPlugin, mixed_precision=None)


@rerun_if_address_is_in_use()
def test_ddp_fp32_detects_bug():
    spawn(_check_bug_detected, 1, plugin_fn=TorchDDPPlugin, mixed_precision=None)


@rerun_if_address_is_in_use()
def test_blank_control():
    spawn(_check_blank_control, 1)


@pytest.mark.skipif(NO_ACCELERATOR, reason="LowLevelZero requires a CUDA or NPU device")
@rerun_if_address_is_in_use()
def test_low_level_zero_bf16_aligned():
    spawn(_check_aligned, 1, plugin_fn=partial(LowLevelZeroPlugin, stage=2, precision="bf16"), mixed_precision=None)


@pytest.mark.skipif(NO_ACCELERATOR, reason="LowLevelZero requires a CUDA or NPU device")
@rerun_if_address_is_in_use()
def test_low_level_zero_bf16_detects_bug():
    spawn(
        _check_bug_detected, 1, plugin_fn=partial(LowLevelZeroPlugin, stage=1, precision="bf16"), mixed_precision=None
    )


@pytest.mark.skipif(get_accelerator().name != "cuda", reason="torch AMP fp16 with GradScaler requires CUDA")
@rerun_if_address_is_in_use()
def test_ddp_fp16_aligned():
    spawn(_check_aligned, 1, plugin_fn=TorchDDPPlugin, mixed_precision="fp16")


def test_dtype_audit_tells_recipes_apart():
    s = prepare(_entry())
    cast = run_reference(s, Recipe(torch.bfloat16))
    autocast = run_reference(s, Recipe(torch.bfloat16, autocast=True))
    assert dtype_mismatches(cast, cast) == []
    mismatches = dtype_mismatches(autocast, cast)
    # autocast keeps the weights in float32; casting the whole model does not
    assert "param/fc1.weight: test float32 / ref bfloat16" in mismatches
    assert any(m.startswith("module/fc1:") for m in mismatches)


def test_nondeterministic_reference_uses_worse_run():
    truth = Trace({"a": torch.zeros(4), "b": torch.zeros(4)}, {})
    first = Trace({"a": torch.full((4,), 0.1), "b": torch.full((4,), 0.1)}, {})
    second = Trace({"a": torch.full((4,), 0.3), "b": torch.full((4,), 0.1)}, {})
    ref = _worse_of(first, second, truth)
    assert ref.meta["nondeterministic"] == ["a"]
    assert torch.equal(ref["a"], second["a"]) and torch.equal(ref["b"], first["b"])


REJECTED = [
    (TorchDDPPlugin, "bf16", "BF16MixedPrecision"),
    (partial(LowLevelZeroPlugin, precision="fp16"), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="bf16", max_norm=1.0), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="bf16", cpu_offload=True), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="bf16"), "fp16", "controls precision itself"),
    (object, None, "not supported yet"),
]


def _check_rejected(rank, world_size, port):
    _launch(rank, world_size, port)
    for plugin_fn, mixed_precision, match in REJECTED:
        with pytest.raises(UnsupportedRecipe, match=match):
            recipe_for(plugin_fn(), mixed_precision)


@rerun_if_address_is_in_use()
def test_unverified_recipes_rejected():
    spawn(_check_rejected, 1)


def test_entry_without_spec_rejected():
    with pytest.raises(ValueError, match="no NumericsSpec"):
        prepare((TinyMLP, _data, lambda out: out, None, None))


class _WithUnusedBranch(TinyMLP):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)  # never called


def _check_missing_gradients(rank, world_size, port):
    _launch(rank, world_size, port)
    # DDP must be told about the unused branch, or it waits for its gradient
    plugin_fn = partial(TorchDDPPlugin, find_unused_parameters=True)
    entry = (_WithUnusedBranch, _data, lambda out: out, None, ModelAttribute(numerics=NumericsSpec()))
    report = check_training(entry, plugin_fn())
    failed = {e.name: e.reason for e in report.failures}
    assert failed == {f"grad/param/head.{p}": "no gradient in test, ref, truth" for p in ("weight", "bias")}, failed
    spec = NumericsSpec(unused_params=("head.weight", "head.bias"))
    check_training((_WithUnusedBranch, _data, lambda out: out, None, ModelAttribute(numerics=spec)), plugin_fn())


@rerun_if_address_is_in_use()
def test_missing_gradients_detected():
    spawn(_check_missing_gradients, 1)


class _DropColumn(nn.Linear):
    def forward(self, input):
        return super().forward(torch.cat([input[:, :63], torch.zeros_like(input[:, 63:])], dim=1))


def test_check_fwd_bwd():
    torch.manual_seed(0)
    ref_fn = nn.Linear(64, 32)
    inputs = {"input": torch.randn(16, 64, dtype=torch.float64)}
    report = check_fwd_bwd(copy.deepcopy(ref_fn), ref_fn, inputs, torch.bfloat16)
    assert {e.name for e in report.entries} == {"out", "grad/input", "grad/param/weight", "grad/param/bias"}
    report.assert_pass()

    # the fused path assertion: forward of the module under test must run, an unrelated function must not
    called = check_fwd_bwd(copy.deepcopy(ref_fn), ref_fn, inputs, torch.bfloat16, expect_calls=[(nn.Linear, "forward")])
    assert [(e.name, e.passed) for e in called.entries if e.name.startswith("<called>")] == [
        ("<called>/Linear.forward", True)
    ]
    missed = check_fwd_bwd(copy.deepcopy(ref_fn), ref_fn, inputs, torch.bfloat16, expect_calls=[(torch, "layer_norm")])
    assert [(e.name, e.reason) for e in missed.failures] == [("<called>/torch.layer_norm", "never called")]
    assert nn.Linear.forward.__name__ == "forward" and torch.layer_norm.__name__ == "layer_norm"  # restored

    buggy = _DropColumn(64, 32)
    buggy.load_state_dict(ref_fn.state_dict())
    failed = {e.name for e in check_fwd_bwd(buggy, ref_fn, inputs, torch.bfloat16).failures}
    assert {"out", "grad/input", "grad/param/weight"} <= failed


def _tiny_llama():
    import transformers

    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        attn_implementation="eager",
    )
    return transformers.LlamaModel(config)


def _tokens():
    return {"input_ids": torch.randint(0, 128, (2, 8))}


def _hidden(output):
    return {"hidden": output.last_hidden_state}


def _check_shardformer(rank, world_size, port, precision):
    _launch(rank, world_size, port)
    entry = (_tiny_llama, _tokens, _hidden, None, ModelAttribute(numerics=NumericsSpec()))
    plugin = HybridParallelPlugin(tp_size=1, pp_size=1, precision=precision)
    check_training(entry, plugin, steps=STEPS).assert_pass()
    for kwargs, match in [
        (dict(zero_stage=1), "zero_stage"),
        (dict(max_norm=1.0), "max_norm"),
        (dict(precision="fp16"), "not verified"),
    ]:
        with pytest.raises(UnsupportedRecipe, match=match):
            recipe_for(HybridParallelPlugin(**{"tp_size": 1, "pp_size": 1, "precision": "bf16", **kwargs}))


@pytest.mark.skipif(NO_ACCELERATOR, reason="HybridParallelPlugin requires a CUDA or NPU device")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@rerun_if_address_is_in_use()
def test_shardformer_aligned(precision):
    spawn(_check_shardformer, 1, precision=precision)


class _TinyMoE(nn.Module):
    """Top-2 of 4 experts, for routing diagnostics through a plugin."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 4, bias=False)
        self.experts = nn.ModuleList(nn.Linear(64, 32) for _ in range(4))

    def forward(self, x):
        weights, chosen = self.gate(x).softmax(-1).topk(2, dim=-1)
        out = 0
        for k in range(2):
            for e, expert in enumerate(self.experts):
                out = out + (chosen[:, k : k + 1] == e) * weights[:, k : k + 1] * expert(x)
        return {"y": out}


def _check_routing_in_training(rank, world_size, port):
    _launch(rank, world_size, port)
    spec = NumericsSpec(routers=Routers(select=lambda name, module: name == "gate", top_k=2))
    entry = (_TinyMoE, _data, lambda out: out, None, ModelAttribute(numerics=spec))
    report = check_training(entry, TorchDDPPlugin(), steps=STEPS)
    report.assert_pass()
    # the first forward only: 16 tokens through one router, and the plugin reproduces the reference's routing
    assert report.routing["routings"] == 16
    assert report.routing["test"] == report.routing["ref"]


@rerun_if_address_is_in_use()
def test_routing_counted_in_training():
    spawn(_check_routing_in_training, 1)
