import io
import math

import pytest
import torch

from colossalai.nn.lr_scheduler.delayed import DelayerScheduler, WarmupDelayerScheduler, WarmupScheduler


def make_schedule(wrapper, decay, phase_length):
    parameters = [torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float64)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [{"params": [parameters[0]], "lr": 0.8}, {"params": [parameters[1]], "lr": 0.2}], momentum=0.9
    )
    if decay == "cosine":
        after = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0.05)
    elif decay == "exponential":
        after = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    else:
        after = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    if wrapper == "delay":
        scheduler = DelayerScheduler(optimizer, phase_length, after)
    elif wrapper == "warmup":
        scheduler = WarmupScheduler(optimizer, phase_length, after)
    else:
        scheduler = WarmupDelayerScheduler(optimizer, phase_length, phase_length, after)
    return parameters, optimizer, scheduler


def expected_lrs(wrapper, decay, phase_length, epoch):
    warmup = phase_length if wrapper != "delay" else 0
    delay = phase_length if wrapper != "warmup" else 0
    if epoch < warmup:
        return [base * (epoch + 1) / warmup for base in (0.8, 0.2)]
    if epoch < warmup + delay:
        return [0.8, 0.2]
    decay_epoch = epoch - warmup - delay
    if decay == "cosine":
        return [0.05 + (base - 0.05) * (1 + math.cos(math.pi * decay_epoch / 6)) / 2 for base in (0.8, 0.2)]
    exponent = decay_epoch if decay == "exponential" else decay_epoch // 2
    return [base * 0.5**exponent for base in (0.8, 0.2)]


def train_step(parameters, optimizer):
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()


@pytest.mark.parametrize("wrapper", ["delay", "warmup", "warmup_delay"])
@pytest.mark.parametrize("decay", ["cosine", "exponential", "step"])
@pytest.mark.parametrize("phase_length", [0, 2])
def test_handoff_matches_decay_epoch_zero(wrapper, decay, phase_length):
    parameters, optimizer, scheduler = make_schedule(wrapper, decay, phase_length)
    reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    reference_optimizer = torch.optim.SGD(
        [{"params": [reference[0]], "lr": 0.8}, {"params": [reference[1]], "lr": 0.2}], momentum=0.9
    )
    prefix = phase_length * (2 if wrapper == "warmup_delay" else 1)
    for epoch in range(prefix + 7):
        expected = expected_lrs(wrapper, decay, phase_length, epoch)
        assert scheduler.get_last_lr() == pytest.approx(expected)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected)
        for group, lr in zip(reference_optimizer.param_groups, expected):
            group["lr"] = lr
        train_step(parameters, optimizer)
        train_step(reference, reference_optimizer)
        for parameter, expected_parameter in zip(parameters, reference):
            torch.testing.assert_close(parameter, expected_parameter)
        scheduler.step()


@pytest.mark.parametrize("wrapper", ["delay", "warmup", "warmup_delay"])
@pytest.mark.parametrize("checkpoint_phase", ["before_handoff", "after_handoff"])
def test_handoff_checkpoint_continuation(wrapper, checkpoint_phase):
    parameters, optimizer, scheduler = make_schedule(wrapper, "cosine", 2)
    prefix = 4 if wrapper == "warmup_delay" else 2
    checkpoint_epoch = prefix - 1 if checkpoint_phase == "before_handoff" else prefix + 2
    for _ in range(checkpoint_epoch):
        train_step(parameters, optimizer)
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
    state = torch.load(checkpoint, weights_only=True)
    resumed, resumed_optimizer, resumed_scheduler = make_schedule(wrapper, "cosine", 2)
    with torch.no_grad():
        for parameter, saved in zip(resumed, state["parameters"]):
            parameter.copy_(saved)
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    for _ in range(5):
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
        train_step(parameters, optimizer)
        train_step(resumed, resumed_optimizer)
        scheduler.step()
        resumed_scheduler.step()
        for parameter, resumed_parameter in zip(parameters, resumed):
            torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
