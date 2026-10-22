"""CPU scheduler backward contracts; no P2P transport is exercised here."""

import pytest
import torch

from colossalai.pipeline.schedule.interleaved_pp import InterleavedSchedule
from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule


class NativeOptimizer:
    def __init__(self):
        self.calls = 0

    def backward(self, loss):
        self.calls += 1
        loss.backward()

    def backward_by_grad(self, tensor, grad):
        self.calls += 1
        torch.autograd.backward(tensor, grad)


@pytest.mark.parametrize("schedule", [OneForwardOneBackwardSchedule, InterleavedSchedule])
def test_scheduler_nested_backward(schedule):
    inputs = {"hidden": torch.ones(2, requires_grad=True), "router": (torch.ones(2, requires_grad=True),)}
    outputs = {"hidden": inputs["hidden"] * 2, "router": (inputs["router"][0] * 3,)}
    upstream = {"hidden": torch.ones(2), "router": (torch.ones(2),)}
    optimizer = NativeOptimizer()
    gradients = schedule.backward_step(None, optimizer, inputs, outputs, upstream)
    assert optimizer.calls == 1
    torch.testing.assert_close(gradients["hidden"], torch.full((2,), 2.0))
    torch.testing.assert_close(gradients["router"][0], torch.full((2,), 3.0))


@pytest.mark.parametrize("schedule", [OneForwardOneBackwardSchedule, InterleavedSchedule])
def test_scheduler_no_backward_tensor(schedule):
    optimizer = NativeOptimizer()
    result = schedule.backward_step(None, optimizer, {"metadata": "x"}, {"metadata": "y"}, {})
    assert result is None
    assert optimizer.calls == 0


@pytest.mark.parametrize("schedule", [OneForwardOneBackwardSchedule, InterleavedSchedule])
def test_scheduler_structure_error_before_optimizer(schedule):
    optimizer = NativeOptimizer()
    inputs = {"hidden": torch.ones(2, requires_grad=True)}
    outputs = {"hidden": (inputs["hidden"] * 2,)}
    with pytest.raises(ValueError, match="structures must match"):
        schedule.backward_step(None, optimizer, inputs, outputs, {"hidden": torch.ones(2)})
    assert optimizer.calls == 0
