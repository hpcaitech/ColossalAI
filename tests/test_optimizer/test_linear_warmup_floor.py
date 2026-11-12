import io

import pytest
import torch

from colossalai.nn.lr_scheduler import LinearWarmupLR


def make_optimization(warmup_steps):
    parameters = [torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [{"params": [parameters[0]], "lr": 0.3}, {"params": [parameters[1]], "lr": 0.12}],
        momentum=0.8,
    )
    scheduler = LinearWarmupLR(optimizer, total_steps=4, warmup_steps=warmup_steps)
    return parameters, optimizer, scheduler


@pytest.mark.parametrize(
    "warmup_steps,factors",
    [
        (0, [1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0]),
        (1, [0.5, 1.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0]),
        (3, [0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0]),
    ],
)
@pytest.mark.parametrize("explicit_epoch", [False, True])
def test_linear_schedule_stays_at_zero_after_decay(warmup_steps, factors, explicit_epoch):
    parameters, optimizer, scheduler = make_optimization(warmup_steps)
    reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    reference_optimizer = torch.optim.SGD(
        [{"params": [reference[0]], "lr": 0.3}, {"params": [reference[1]], "lr": 0.12}],
        momentum=0.8,
    )
    for step, factor in enumerate(factors):
        rates = [0.3 * factor, 0.12 * factor]
        assert scheduler.get_last_lr() == pytest.approx(rates)
        for group, rate in zip(reference_optimizer.param_groups, rates):
            group["lr"] = rate
        for index, (parameter, expected) in enumerate(zip(parameters, reference)):
            parameter.grad = torch.full_like(parameter, float(index + 1))
            expected.grad = parameter.grad.clone()
        optimizer.step()
        reference_optimizer.step()
        for parameter, expected in zip(parameters, reference):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        scheduler.step(step + 1) if explicit_epoch else scheduler.step()


@pytest.mark.parametrize("warmup_steps", [0, 2])
@pytest.mark.parametrize("epoch", [5, 100])
def test_jumping_past_the_schedule_does_not_reverse_updates(warmup_steps, epoch):
    parameters, optimizer, scheduler = make_optimization(warmup_steps)
    optimizer.step()
    scheduler.step(epoch)
    assert scheduler.get_last_lr() == [0.0, 0.0]
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter, expected in zip(parameters, before):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("warmup_steps", [0, 2])
@pytest.mark.parametrize("save_after_steps", [2, 5])
def test_checkpoint_continuation_keeps_the_completed_schedule_at_zero(warmup_steps, save_after_steps):
    parameters, optimizer, scheduler = make_optimization(warmup_steps)
    for _ in range(save_after_steps):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    checkpoint = io.BytesIO()
    torch.save(
        {
            "parameters": [parameter.detach() for parameter in parameters],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint,
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed, resumed_optimizer, resumed_scheduler = make_optimization(warmup_steps)
    with torch.no_grad():
        for parameter, value in zip(resumed, saved["parameters"]):
            parameter.copy_(value)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_scheduler.load_state_dict(saved["scheduler"])
    for _ in range(4):
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
        assert min(resumed_scheduler.get_last_lr()) >= 0.0
        for current, current_optimizer, current_scheduler in (
            (parameters, optimizer, scheduler),
            (resumed, resumed_optimizer, resumed_scheduler),
        ):
            for parameter in current:
                parameter.grad = torch.ones_like(parameter)
            current_optimizer.step()
            current_scheduler.step()
        for parameter, expected in zip(resumed, parameters):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("warmup_steps", [0, 1, 3])
def test_schedule_reaches_zero_at_the_existing_endpoint(warmup_steps):
    _, optimizer, scheduler = make_optimization(warmup_steps)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == [0.0, 0.0]
