import io

import pytest
import torch
from transformers.optimization import Adafactor as ReferenceAdafactor

from colossalai.nn.optimizer import Adafactor, DistributedAdaFactor


def make_optimizer(optimizer_type, parameters, **kwargs):
    optimizer = optimizer_type(parameters, **kwargs)
    if isinstance(optimizer, DistributedAdaFactor):
        optimizer.setup_distributed(use_zero=False)
    return optimizer


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
@pytest.mark.parametrize("shape", [(6,), (2, 3)])
@pytest.mark.parametrize("beta1", [None, 0.8])
def test_manual_learning_rate_matches_reference(optimizer_type, shape, beta1):
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 6).reshape(shape))
    expected = torch.nn.Parameter(parameter.detach().clone())
    options = dict(lr=0.05, relative_step=False, scale_parameter=False, beta1=beta1, weight_decay=0.02)
    optimizer = make_optimizer(optimizer_type, (p for p in [parameter]), **options)
    reference = ReferenceAdafactor([expected], **options)
    for step in range(5):
        gradient = (torch.arange(6).reshape(shape) + 1).float() * (step + 1)
        parameter.grad = gradient.clone()
        expected.grad = gradient.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(parameter, expected)
        assert optimizer.param_groups[0]["lr"] == 0.05


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_manual_rate_rejects_relative_step_conflict(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="Cannot combine manual"):
        optimizer_type([parameter], lr=0.1, relative_step=True)


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_default_relative_rate_still_steps(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = make_optimizer(optimizer_type, [parameter], scale_parameter=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.99))
    assert optimizer.param_groups[0]["lr"] is None


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_parameter_group_learning_rates_remain_independent(optimizer_type):
    parameters = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    optimizer = make_optimizer(
        optimizer_type,
        [{"params": [parameters[0]], "lr": 0.1}, {"params": [parameters[1]], "lr": 0.02}],
        lr=0.4,
        relative_step=False,
        scale_parameter=False,
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter, expected in zip(parameters, [0.9, 0.98]):
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_zero_manual_rate_leaves_parameters_unchanged(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = make_optimizer(optimizer_type, [parameter], lr=0.0, relative_step=False, scale_parameter=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_manual_rate_scheduler_and_checkpoint_resume(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = make_optimizer(optimizer_type, [parameter], lr=0.1, relative_step=False, scale_parameter=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    checkpoint = io.BytesIO()
    torch.save(
        {"parameter": parameter.detach(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    resumed = torch.nn.Parameter(state["parameter"])
    resumed_optimizer = make_optimizer(optimizer_type, [resumed], lr=0.8, relative_step=False, scale_parameter=False)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.5)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    for _ in range(3):
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
        for current, current_optimizer, current_scheduler in (
            (parameter, optimizer, scheduler),
            (resumed, resumed_optimizer, resumed_scheduler),
        ):
            current.grad = torch.ones_like(current)
            current_optimizer.step()
            current_scheduler.step()
        torch.testing.assert_close(parameter, resumed, rtol=0, atol=0)
