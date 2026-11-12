import io

import pytest
import torch

from colossalai.nn.lr_scheduler import OneCycleLR


def make_optimizer(optimizer_type):
    parameters = [torch.nn.Parameter(torch.tensor(float(index + 1), dtype=torch.float64)) for index in range(2)]
    optimizer = optimizer_type([{"params": [parameters[0]], "lr": 0.1}, {"params": [parameters[1]], "lr": 0.04}])
    return parameters, optimizer


def check_step(parameters, optimizer, scheduler, expected, reference_optimizer, reference_scheduler, step):
    assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
    for index, (group, reference_group) in enumerate(zip(optimizer.param_groups, reference_optimizer.param_groups)):
        assert group.get("momentum") == reference_group.get("momentum")
        assert group.get("betas") == reference_group.get("betas")
        gradient = torch.tensor((index + 1) * (step + 1) * 0.1, dtype=torch.float64)
        parameters[index].grad = gradient.clone()
        expected[index].grad = gradient.clone()
    optimizer.step()
    reference_optimizer.step()
    scheduler.step()
    reference_scheduler.step()
    for actual, value in zip(parameters, expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("anneal_strategy", ["cos", "linear"])
@pytest.mark.parametrize("three_phase", [False, True])
def test_optional_phases_match_native_learning_rates_momentum_and_updates(optimizer_type, anneal_strategy, three_phase):
    parameters, optimizer = make_optimizer(optimizer_type)
    expected, reference_optimizer = make_optimizer(optimizer_type)
    options = {"total_steps": 10, "anneal_strategy": anneal_strategy, "three_phase": three_phase}
    scheduler = OneCycleLR(optimizer, **options)
    reference_scheduler = torch.optim.lr_scheduler.OneCycleLR(reference_optimizer, max_lr=[0.1, 0.04], **options)

    for step in range(10):
        check_step(parameters, optimizer, scheduler, expected, reference_optimizer, reference_scheduler, step)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("three_phase", [False, True])
def test_epochs_and_steps_per_epoch_define_the_schedule(optimizer_type, three_phase):
    parameters, optimizer = make_optimizer(optimizer_type)
    expected, reference_optimizer = make_optimizer(optimizer_type)
    options = {"total_steps": None, "epochs": 2, "steps_per_epoch": 5, "three_phase": three_phase}
    scheduler = OneCycleLR(optimizer, **options)
    reference_scheduler = torch.optim.lr_scheduler.OneCycleLR(reference_optimizer, max_lr=[0.1, 0.04], **options)
    assert scheduler.total_steps == 10

    for step in range(10):
        check_step(parameters, optimizer, scheduler, expected, reference_optimizer, reference_scheduler, step)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
def test_three_phase_checkpoint_continuation_matches_native_scheduler(optimizer_type):
    parameters, optimizer = make_optimizer(optimizer_type)
    expected, reference_optimizer = make_optimizer(optimizer_type)
    options = {"total_steps": 10, "three_phase": True}
    scheduler = OneCycleLR(optimizer, **options)
    reference_scheduler = torch.optim.lr_scheduler.OneCycleLR(reference_optimizer, max_lr=[0.1, 0.04], **options)
    for step in range(4):
        check_step(parameters, optimizer, scheduler, expected, reference_optimizer, reference_scheduler, step)
    checkpoint = io.BytesIO()
    torch.save(
        {"parameters": parameters, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed, resumed_optimizer = make_optimizer(optimizer_type)
    resumed_scheduler = OneCycleLR(resumed_optimizer, **options)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_scheduler.load_state_dict(saved["scheduler"])
    with torch.no_grad():
        for parameter, value in zip(resumed, saved["parameters"]):
            parameter.copy_(value)

    for step in range(4, 10):
        check_step(
            resumed, resumed_optimizer, resumed_scheduler, expected, reference_optimizer, reference_scheduler, step
        )


def test_unknown_optional_argument_is_not_silently_discarded():
    _, optimizer = make_optimizer(torch.optim.SGD)

    with pytest.raises(TypeError, match="three_phaze"):
        OneCycleLR(optimizer, total_steps=10, three_phaze=True)
