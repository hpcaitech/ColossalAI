from functools import partial

import pytest
import torch
from torch import nn

from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.model_zoo.registry import ModelAttribute
from tests.kit.numerics import NumericsSpec, judge_relative_bound, relative_bound
from tests.kit.numerics.scenarios import data_parallel

GRADS = {f"grad/param/fc{i}.{p}" for i in (1, 2) for p in ("weight", "bias")}


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return {"y": self.fc2(torch.relu(self.fc1(x)))}


def _data():
    return {"x": torch.randn(16, 64)}


def _identity(output):
    return output


# module-level parts only: the entry is sent to spawned processes
ENTRY = (TinyMLP, _data, _identity, None, ModelAttribute(numerics=NumericsSpec()))


class UnsyncedDDPPlugin(TorchDDPPlugin):
    """DDP with one parameter left out of gradient synchronization: a typical data-parallel bug."""

    def configure(self, model, *args, **kwargs):
        model._ddp_params_and_buffers_to_ignore = ["fc1.weight"]
        return super().configure(model, *args, **kwargs)


@rerun_if_address_is_in_use()
def test_ddp_fp32_equivalent():
    report = data_parallel.check(ENTRY, TorchDDPPlugin, world_size=2, micro_batches=4)
    assert {e.name for e in report.entries} == GRADS
    report.assert_pass()


@rerun_if_address_is_in_use()
def test_unsynchronized_parameter_detected():
    report = data_parallel.check(ENTRY, UnsyncedDDPPlugin, world_size=2)
    assert {e.name for e in report.failures} == {"grad/param/fc1.weight"}, report.format()
    # far beyond the bound, not a borderline call
    assert report.failures[0].ratio > 3, report.format()


@pytest.mark.skipif(
    get_accelerator().name == "cpu" or get_accelerator().device_count() < 2,
    reason="LowLevelZero requires two CUDA or NPU devices",
)
@rerun_if_address_is_in_use()
def test_low_level_zero_bf16_equivalent():
    data_parallel.check(ENTRY, partial(LowLevelZeroPlugin, stage=2, precision="bf16"), world_size=2).assert_pass()


def test_relative_bound_uses_megatron_constants():
    assert relative_bound(torch.bfloat16) == pytest.approx(4 * 1.03**33 * 2**-7)
    assert relative_bound(torch.bfloat16) == pytest.approx(0.0829, abs=1e-4)


def test_judge_relative_bound():
    g = torch.Generator().manual_seed(0)
    ref = {"a": torch.randn(256, generator=g), "b": torch.randn(256, generator=g)}
    close = {"a": ref["a"] * (1 + 1e-3), "b": ref["b"]}
    assert judge_relative_bound(close, ref, torch.bfloat16).passed
    # a wrong averaging factor
    report = judge_relative_bound({"a": ref["a"] * 2, "b": ref["b"]}, ref, torch.bfloat16)
    assert [e.name for e in report.failures] == ["a"] and report.failures[0].ratio > 10
    # tiny on both sides passes per tensor, but a comparison of nothing but zeros fails
    zeros = {"a": torch.zeros(4), "b": torch.full((4,), 1e-6)}
    report = judge_relative_bound(zeros, {"a": torch.zeros(4), "b": torch.zeros(4)}, torch.bfloat16)
    assert [e.name for e in report.failures] == ["<all>"]
