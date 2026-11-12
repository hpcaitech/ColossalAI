"""CPU contracts for real HybridParallelModule casts, independent of accelerator-only boost tests."""

import ast
from pathlib import Path

import pytest
import torch

from colossalai.accelerator import CpuAccelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy


class _NoOpPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self):
        return {}

    def postprocess(self):
        return self.model


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1003, 0.2007, 0.3009, 0.4001]))
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.1, 0.01, 0.001]))
        self.register_buffer("indexes", torch.arange(4))
        self.seen = None

    def forward(self, hidden, nested, *, timesteps):
        # Record actual arguments at the leaf forward, including recursively nested positional inputs.
        self.seen = (
            hidden.detach().clone(),
            nested["positions"][0].clone(),
            timesteps.clone(),
            nested["indexes"].clone(),
            nested["label"],
        )
        phase = nested["positions"][0].float() * self.inv_freq.float()
        return (hidden.to(self.weight.dtype) * self.weight).float().sum() + phase.cos().sum()


def _wrap(monkeypatch, **kwargs):
    monkeypatch.setattr("colossalai.booster.plugin.hybrid_parallel_plugin.get_accelerator", lambda: CpuAccelerator())
    model = _Recorder()
    original_weight = model.weight.detach().clone()
    original_freq = model.inv_freq.clone()
    wrapped = HybridParallelModule(
        module=model,
        precision=kwargs.pop("precision", "bf16"),
        shard_config=ShardConfig(enable_tensor_parallelism=False),
        dp_group=None,
        tp_group=None,
        sp_group=None,
        use_ddp=False,
        ddp_config={},
        custom_policy=_NoOpPolicy(),
        **kwargs,
    )
    return wrapped, original_weight, original_freq


def _check_forward(wrapped, original_weight, original_freq, cast_inputs, cast_buffers, dtype):
    hidden = torch.full((4,), 0.12345)
    positions = torch.tensor([15163.8, 8193.25, 12345.6, 0.12345])
    timesteps = torch.tensor([123.456])
    indexes = torch.arange(4)
    output = wrapped(hidden, {"positions": [positions], "indexes": indexes, "label": "pack-a"}, timesteps=timesteps)
    seen_hidden, seen_positions, seen_timesteps, seen_indexes, label = wrapped.unwrap().seen
    input_dtype = dtype if cast_inputs else torch.float32
    buffer_dtype = dtype if cast_buffers else torch.float32
    torch.testing.assert_close(seen_hidden, hidden.to(input_dtype), rtol=0, atol=0)
    torch.testing.assert_close(seen_positions, positions.to(input_dtype), rtol=0, atol=0)
    torch.testing.assert_close(seen_timesteps, timesteps.to(input_dtype), rtol=0, atol=0)
    torch.testing.assert_close(wrapped.unwrap().weight, original_weight.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(wrapped.unwrap().inv_freq, original_freq.to(buffer_dtype), rtol=0, atol=0)
    assert seen_indexes.dtype == torch.int64 and label == "pack-a"
    torch.testing.assert_close(seen_indexes, indexes)
    torch.testing.assert_close(wrapped.unwrap().indexes, indexes)
    expected_phase = positions.to(input_dtype).float() * original_freq.to(buffer_dtype).float()
    expected = (hidden.to(input_dtype).to(dtype) * original_weight.to(dtype)).float().sum() + expected_phase.cos().sum()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.backward()
    torch.testing.assert_close(wrapped.unwrap().weight.grad, hidden.to(input_dtype).to(dtype), rtol=0, atol=0)
    # The caller's input tensors are untouched in every mode.
    assert positions.dtype == torch.float32
    assert positions[0].item() == torch.tensor(15163.8).item()


@pytest.mark.parametrize("cast_inputs", [False, True])
@pytest.mark.parametrize("cast_buffers", [False, True])
def test_independent_input_and_buffer_casts_reach_real_forward(monkeypatch, cast_inputs, cast_buffers):
    wrapped, weight, freq = _wrap(monkeypatch, cast_inputs=cast_inputs, cast_buffers=cast_buffers)
    _check_forward(wrapped, weight, freq, cast_inputs, cast_buffers, torch.bfloat16)


def test_default_casts_remain_backward_compatible(monkeypatch):
    wrapped, weight, freq = _wrap(monkeypatch)
    _check_forward(wrapped, weight, freq, True, True, torch.bfloat16)


@pytest.mark.parametrize("cast_inputs,cast_buffers", [(True, True), (False, False)])
def test_fp32_precision_preserves_values_regardless_of_switches(monkeypatch, cast_inputs, cast_buffers):
    wrapped, weight, freq = _wrap(monkeypatch, precision="fp32", cast_inputs=cast_inputs, cast_buffers=cast_buffers)
    _check_forward(wrapped, weight, freq, cast_inputs, cast_buffers, torch.float32)


@pytest.mark.parametrize(
    "filename,class_name",
    [
        ("hybrid_parallel_plugin.py", "HybridParallelPlugin"),
        ("moe_hybrid_parallel_plugin.py", "MoeHybridParallelPlugin"),
    ],
)
def test_plugin_flags_reach_hybrid_module_constructor(filename, class_name):
    # MoE constructor requires accelerator process groups. Check the real configure call and defaults without
    # claiming an EP runtime validation or replacing the module constructor with an implementation mirror.
    import colossalai.booster.plugin.hybrid_parallel_plugin as implementation

    tree = ast.parse((Path(implementation.__file__).parent / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    defaults = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults) :], init.args.defaults))
    for flag in ("cast_inputs", "cast_buffers"):
        assert isinstance(defaults[flag], ast.Constant) and defaults[flag].value is True
        assignments = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == flag for target in node.targets)
        ]
        assert any(isinstance(node.value, ast.Name) and node.value.id == flag for node in assignments)
    configure = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "configure")
    calls = [
        node
        for node in ast.walk(configure)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HybridParallelModule"
    ]
    assert calls
    for call in calls:
        kwargs = {item.arg: item.value for item in call.keywords}
        for flag in ("cast_inputs", "cast_buffers"):
            assert isinstance(kwargs[flag], ast.Attribute)
            assert isinstance(kwargs[flag].value, ast.Name) and kwargs[flag].value.id == "self"
            assert kwargs[flag].attr == flag
