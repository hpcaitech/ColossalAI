import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster.mixed_precision import BF16MixedPrecision
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import NumericsSpec, Routers, Trace
from tests.kit.numerics.capture.recipes import Recipe, UnsupportedRecipe, recipe_for
from tests.kit.numerics.capture.training import prepare, run_reference
from tests.kit.numerics.scenarios.independent_impl import (
    _worse_of,
    blank_control,
    check_fwd_bwd,
    check_optimizer,
    check_training,
    dtype_mismatches,
    training_traces,
)

STEPS = 3
NO_ACCELERATOR = get_accelerator().name == "cpu"


def _attribute(spec):
    # all the tool reads from an entry's ModelAttribute; importing tests.kit.model_zoo would need timm, torchvision and
    # torchaudio, which the CPU CI job does not install
    return SimpleNamespace(numerics=spec)


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
    return (TinyMLP, _data, lambda out: out, None, _attribute(spec))


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


@pytest.mark.skipif(NO_ACCELERATOR, reason="BF16Module needs the accelerator's BF16 autocast")
@pytest.mark.parametrize("mixed_precision", ["bf16", BF16MixedPrecision()])
@rerun_if_address_is_in_use()
def test_ddp_bf16_aligned(mixed_precision):
    spawn(_check_aligned, 1, plugin_fn=TorchDDPPlugin, mixed_precision=mixed_precision)


@pytest.mark.skipif(NO_ACCELERATOR, reason="BF16Module needs the accelerator's BF16 autocast")
@rerun_if_address_is_in_use()
def test_ddp_bf16_detects_bug():
    spawn(_check_bug_detected, 1, plugin_fn=TorchDDPPlugin, mixed_precision="bf16")


def test_dtype_audit_tells_recipes_apart():
    s = prepare(_entry())
    device = get_accelerator().get_current_device().type
    cast = run_reference(s, Recipe(torch.bfloat16))
    autocast = run_reference(s, Recipe(torch.bfloat16, autocast=True))
    assert dtype_mismatches(cast, cast) == []
    mismatches = dtype_mismatches(autocast, cast)
    # autocast keeps the weights in float32; casting the whole model does not
    assert f"param/fc1.weight: test float32@{device} / ref bfloat16@{device}" in mismatches
    assert any(m.startswith("module/fc1:") for m in mismatches)
    master = run_reference(s, Recipe(torch.bfloat16, master_weights=True))
    assert f"optim/params: test float32@{device} / ref bfloat16@{device}" in dtype_mismatches(master, cast)


@pytest.mark.skipif(NO_ACCELERATOR, reason="needs an accelerator besides the CPU")
def test_device_audit_tells_offloading_apart():
    s = prepare(_entry())
    device = get_accelerator().get_current_device().type
    on_device = run_reference(s, Recipe(torch.bfloat16, master_weights=True))
    offloaded = run_reference(s, Recipe(torch.bfloat16, master_weights=True, optimizer_device="cpu"))
    # master weights and Adam's moments live on the CPU with offloading, in the same dtype
    assert set(dtype_mismatches(offloaded, on_device)) == {
        f"optim/{key}: test float32@cpu / ref float32@{device}" for key in ("params", "exp_avg", "exp_avg_sq")
    }


def test_nondeterministic_reference_uses_worse_run():
    truth = Trace({"a": torch.zeros(4), "b": torch.zeros(4)}, {})
    first = Trace({"a": torch.full((4,), 0.1), "b": torch.full((4,), 0.1)}, {})
    second = Trace({"a": torch.full((4,), 0.3), "b": torch.full((4,), 0.1)}, {})
    ref = _worse_of(first, second, truth)
    assert ref.meta["nondeterministic"] == ["a"]
    assert torch.equal(ref["a"], second["a"]) and torch.equal(ref["b"], first["b"])


REJECTED = [
    (partial(LowLevelZeroPlugin, precision="fp16"), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="bf16", max_norm=1.0), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="fp32", cpu_offload=True), None, "not verified"),
    (partial(LowLevelZeroPlugin, precision="bf16"), "fp16", "controls precision itself"),
    (object, None, "not supported yet"),
]


def _check_rejected(rank, world_size, port):
    _launch(rank, world_size, port)
    for plugin_fn, mixed_precision, match in REJECTED:
        with pytest.raises(UnsupportedRecipe, match=match):
            recipe_for(plugin_fn(), mixed_precision)
    if get_accelerator().name == "cpu":
        for mixed_precision in ("bf16", BF16MixedPrecision()):
            with pytest.raises(UnsupportedRecipe, match="CPUAccelerator.autocast"):
                recipe_for(TorchDDPPlugin(), mixed_precision)


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
    entry = (_WithUnusedBranch, _data, lambda out: out, None, _attribute(NumericsSpec()))
    report = check_training(entry, plugin_fn())
    failed = {e.name: e.reason for e in report.failures}
    assert failed == {f"grad/param/head.{p}": "no gradient in test, ref, truth" for p in ("weight", "bias")}, failed
    spec = NumericsSpec(unused_params=("head.weight", "head.bias"))
    check_training((_WithUnusedBranch, _data, lambda out: out, None, _attribute(spec)), plugin_fn())


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
    entry = (_tiny_llama, _tokens, _hidden, None, _attribute(NumericsSpec()))
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
    entry = (_TinyMoE, _data, lambda out: out, None, _attribute(spec))
    report = check_training(entry, TorchDDPPlugin(), steps=STEPS)
    report.assert_pass()
    # the first forward only: 16 tokens through one router, and the plugin reproduces the reference's routing
    assert report.routing["routings"] == 16
    assert report.routing["test"] == report.routing["ref"]


@rerun_if_address_is_in_use()
def test_routing_counted_in_training():
    spawn(_check_routing_in_training, 1)


def _plugin(name):
    # plugins are built inside the spawned process: Gemini and FSDP need the process group at construction
    from torch.distributed.fsdp import MixedPrecision

    from colossalai.booster.plugin import GeminiPlugin, TorchFSDPPlugin

    bf16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return {
        "gemini_bf16": lambda: GeminiPlugin(precision="bf16"),
        "gemini_fp16": lambda: GeminiPlugin(precision="fp16"),
        "fsdp_fp32": lambda: TorchFSDPPlugin(),
        "fsdp_bf16": lambda: TorchFSDPPlugin(mixed_precision=bf16),
        "zero2_bf16_offload": lambda: LowLevelZeroPlugin(stage=2, precision="bf16", cpu_offload=True),
    }[name]()


def _check_plugin(rank, world_size, port, name):
    _launch(rank, world_size, port)
    entry = (_tiny_llama, _tokens, _hidden, None, _attribute(NumericsSpec()))
    report = check_training(entry, _plugin(name), steps=STEPS)
    report.assert_pass()
    # on one device the plugin reproduces its recipe exactly; a ratio above 1 means the recipe misses a detail (as
    # where the master weights start). Not Gemini in fp16: its fused HybridAdam and torch's AdamW, the reference,
    # differ in the last float32 bit of a few elements, which fp16's finer rounding sometimes shows
    if name != "gemini_fp16":
        assert all(e.ratio == pytest.approx(1.0) for e in report.entries), report.format()
    # gradients come back for every parameter, also where the plugin keeps them in its own buffers
    params = {n for n, _ in _tiny_llama().named_parameters()}
    assert {e.name for e in report.entries if e.name.startswith("grad/param/")} == {f"grad/param/{n}" for n in params}


@pytest.mark.skipif(NO_ACCELERATOR, reason="Gemini, FSDP and LowLevelZero require a CUDA or NPU device")
@pytest.mark.parametrize("name", ["gemini_bf16", "gemini_fp16", "fsdp_fp32", "fsdp_bf16", "zero2_bf16_offload"])
@rerun_if_address_is_in_use()
def test_plugin_aligned(name):
    spawn(_check_plugin, 1, name=name)


def _unverified_plugins(rank, world_size, port):
    _launch(rank, world_size, port)
    from torch.distributed.fsdp import CPUOffload, MixedPrecision

    from colossalai.booster.plugin import GeminiPlugin, TorchFSDPPlugin

    for plugin, match in [
        (GeminiPlugin(precision="bf16", placement_policy="auto"), "offloading"),
        (GeminiPlugin(precision="bf16", max_norm=1.0), "max_norm"),
        (TorchFSDPPlugin(cpu_offload=CPUOffload(offload_params=True)), "cpu_offload"),
        (TorchFSDPPlugin(mixed_precision=MixedPrecision(param_dtype=torch.bfloat16)), "buffer_dtype"),
        (LowLevelZeroPlugin(stage=2, precision="bf16", cpu_offload=True, master_weights=False), "master_weights"),
    ]:
        with pytest.raises(UnsupportedRecipe, match=match):
            recipe_for(plugin)


@pytest.mark.skipif(NO_ACCELERATOR, reason="Gemini and FSDP require a CUDA device")
@rerun_if_address_is_in_use()
def test_unverified_plugins_refused():
    spawn(_unverified_plugins, 1)


class _Tower(nn.Module):
    """A frozen branch next to a trained head, as a vision tower in a text-only training path."""

    def __init__(self):
        super().__init__()
        self.tower = nn.Linear(64, 64)
        self.head = nn.Linear(64, 32)

    def forward(self, x):
        return {"y": self.head(torch.relu(self.tower(x)))}


def _freeze_tower(model):
    model.tower.requires_grad_(False)


def _check_prepare_fn(rank, world_size, port):
    _launch(rank, world_size, port)
    entry = (_Tower, _data, lambda out: out, None, _attribute(NumericsSpec(prepare_fn=_freeze_tower)))
    s = prepare(entry)
    assert not s.model.tower.weight.requires_grad and s.model.head.weight.requires_grad
    # kept on the CPU between runs: each run copies it to the device, so only one model is there at a time
    assert {p.device.type for p in s.model.parameters()} == {"cpu"}
    report = check_training(entry, TorchDDPPlugin(), steps=STEPS)
    report.assert_pass()
    # frozen parameters are neither required to have a gradient nor updated
    names = {e.name for e in report.entries}
    assert "grad/param/head.weight" in names and "grad/param/tower.weight" not in names
    tower = s.model.tower.weight.detach().cpu()
    assert torch.equal(run_reference(s, recipe_for(TorchDDPPlugin()))["param/tower.weight"], tower)


@rerun_if_address_is_in_use()
def test_prepare_fn_applies_to_every_run():
    spawn(_check_prepare_fn, 1)


class _PlainAdam(torch.optim.Optimizer):
    """Adam written out from the paper: an independent reference for torch.optim.Adam."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.bias_correction = bias_correction

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                state["m"] = beta1 * state["m"] + (1 - beta1) * p.grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * p.grad**2
                m, v = state["m"], state["v"]
                if self.bias_correction:
                    m, v = m / (1 - beta1 ** state["t"]), v / (1 - beta2 ** state["t"])
                p -= group["lr"] * m / (v.sqrt() + group["eps"])


def test_optimizer_update_rule():
    # float32 and lr 1e-2: in bfloat16 an update of 1e-3 is below the resolution of weights near 1 and rounds away
    params = TinyMLP()
    report = check_optimizer(lambda p: torch.optim.Adam(p, lr=1e-2), _PlainAdam, params, torch.float32, steps=3)
    report.assert_pass()
    assert {e.name for e in report.entries} == {f"param/{t}/{p}" for t in range(3) for p in PARAMS}
    # an update without bias correction is off by far more than rounding
    buggy = check_optimizer(partial(_PlainAdam, bias_correction=False), _PlainAdam, params, torch.float32, steps=3)
    assert {e.name for e in buggy.failures} >= {f"param/0/{p}" for p in PARAMS}


def _check_cache(rank, world_size, port, directory):
    import os

    _launch(rank, world_size, port)
    plugin = TorchDDPPlugin()
    first = training_traces(_entry(), plugin, steps=STEPS, cache_dir=directory)
    files = sorted(os.listdir(os.path.join(directory, "custom")))
    assert [f.split("-")[0] for f in files] == ["ref", "truth"]
    second = training_traces(_entry(), plugin, steps=STEPS, cache_dir=directory)
    for a, b in zip(first[1:], second[1:]):
        assert "cached" in b.meta and set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert "cached" not in second[0].meta  # the model under test always runs
    # other weights (another seed) are another key, not a stale hit
    training_traces(_entry(), plugin, steps=STEPS, seed=1, cache_dir=directory)
    assert len(os.listdir(os.path.join(directory, "custom"))) == 4


@rerun_if_address_is_in_use()
def test_reference_runs_are_cached(tmp_path):
    spawn(_check_cache, 1, directory=str(tmp_path))


def _matrices_only(params):
    # a typical parameter-group mistake: the biases are left out
    return torch.optim.AdamW([p for p in params if p.dim() > 1], lr=1e-3)


def _check_optimizer_coverage(rank, world_size, port):
    _launch(rank, world_size, port)
    report = check_training(_entry(), TorchDDPPlugin(), steps=STEPS, optimizer_fn=_matrices_only)
    # both sides leave the biases untouched, so every comparison passes; only the contract catches it
    failed = {e.name: e.reason for e in report.failures}
    assert failed == {f"optim/param/fc{i}.bias": "not held by the optimizer in test, ref, truth" for i in (1, 2)}


@rerun_if_address_is_in_use()
def test_parameters_missing_from_optimizer_detected():
    spawn(_check_optimizer_coverage, 1)


class _TiedLM(nn.Module):
    """The input embedding and the output projection share one weight, as in many language models."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.mix = nn.Linear(16, 16)
        self.head = nn.Linear(16, 32, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids):
        return {"logits": self.head(torch.tanh(self.mix(self.embed(input_ids))))}


def _ids():
    return {"input_ids": torch.randint(0, 32, (4, 8))}


def _untie(model):
    model.head.weight = nn.Parameter(model.head.weight.detach().clone())


TIED = (("embed.weight", "head.weight"),)
TIE_KEY = "tied/embed.weight,head.weight"


def _tied_entry(spec):
    return (_TiedLM, _ids, lambda out: out, None, _attribute(spec))


def _check_tied(rank, world_size, port, name):
    _launch(rank, world_size, port)
    plugin_fn = TorchDDPPlugin if name is None else partial(_plugin, name)
    report = check_training(_tied_entry(NumericsSpec(tied_params=TIED)), plugin_fn(), steps=STEPS)
    report.assert_pass()
    assert all(e.ratio == pytest.approx(1.0) for e in report.entries), report.format()
    # one Parameter, compared once under its first name
    names = {e.name for e in report.entries}
    assert "grad/param/embed.weight" in names and "grad/param/head.weight" not in names
    if name is not None:
        return
    undeclared = check_training(_tied_entry(NumericsSpec()), plugin_fn(), steps=STEPS)
    reason = "one Parameter but not declared in NumericsSpec.tied_params in test, ref, truth"
    assert [(e.name, e.reason) for e in undeclared.failures] == [(TIE_KEY, reason)]
    broken = check_training(_tied_entry(NumericsSpec(tied_params=TIED, prepare_fn=_untie)), plugin_fn(), steps=STEPS)
    assert [(e.name, e.reason) for e in broken.failures] == [
        (TIE_KEY, "declared tied but not one Parameter in test, ref, truth")
    ]


@rerun_if_address_is_in_use()
def test_tied_parameters_checked():
    spawn(_check_tied, 1, name=None)


@pytest.mark.skipif(NO_ACCELERATOR, reason="Gemini, FSDP and LowLevelZero require a CUDA or NPU device")
@pytest.mark.parametrize("name", ["gemini_bf16", "fsdp_bf16", "zero2_bf16_offload"])
@rerun_if_address_is_in_use()
def test_tied_parameters_through_plugins(name):
    spawn(_check_tied, 1, name=name)


# calls of the decoder layers in every run; with gradient checkpointing each layer runs again in backward. Counted
# before the forward: a non-reentrant recomputation stops once it has what backward needs, before forward hooks run
_LAYER_CALLS = []


def _count_layer_calls(model):
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda *args: _LAYER_CALLS.append(1))


def _checkpointed(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _count_layer_calls(model)


def _check_gradient_checkpointing(rank, world_size, port, name):
    _launch(rank, world_size, port)
    plugin_fn = {
        "ddp_fp32": TorchDDPPlugin,
        "shardformer_bf16": partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="bf16"),
    }.get(name, partial(_plugin, name))
    counts = []
    for prepare_fn in (_count_layer_calls, _checkpointed):
        _LAYER_CALLS.clear()
        entry = (_tiny_llama, _tokens, _hidden, None, _attribute(NumericsSpec(prepare_fn=prepare_fn)))
        check_training(entry, plugin_fn(), steps=STEPS).assert_pass()
        counts.append(len(_LAYER_CALLS))
    # every run, the plugin's included, recomputed each layer once in backward
    assert counts[1] == 2 * counts[0], counts


@rerun_if_address_is_in_use()
def test_gradient_checkpointing():
    spawn(_check_gradient_checkpointing, 1, name="ddp_fp32")


@pytest.mark.skipif(NO_ACCELERATOR, reason="the plugins require a CUDA or NPU device")
@pytest.mark.parametrize("name", ["zero2_bf16_offload", "gemini_bf16", "fsdp_bf16", "shardformer_bf16"])
@rerun_if_address_is_in_use()
def test_gradient_checkpointing_through_plugins(name):
    spawn(_check_gradient_checkpointing, 1, name=name)
