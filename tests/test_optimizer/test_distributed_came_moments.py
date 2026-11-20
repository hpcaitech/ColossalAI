import io

import pytest
import torch

from colossalai.nn.optimizer import CAME, DistributedCAME


def make_optimizers(shape, dtype, betas, weight_decay=0.0):
    values = torch.linspace(0.1, 0.6, 6, dtype=dtype).reshape(shape)
    parameter = torch.nn.Parameter(values.clone())
    expected = torch.nn.Parameter(values.clone())
    options = {"lr": 0.01, "betas": betas, "weight_decay": weight_decay}
    optimizer = DistributedCAME([parameter], **options)
    optimizer.setup_distributed(use_zero=False)
    reference = CAME([expected], **options)
    return parameter, optimizer, expected, reference


def step_and_compare(parameter, optimizer, expected, reference, step):
    gradient = torch.linspace(0.1, 0.9, parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
    gradient = gradient.roll(step, dims=-1) * (step + 1)
    parameter.grad = gradient.clone()
    expected.grad = gradient.clone()
    optimizer.step()
    reference.step()
    torch.testing.assert_close(parameter, expected)
    for key, value in reference.state[expected].items():
        actual = optimizer.state[parameter][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value)
        else:
            assert actual == value
    if parameter.ndim == 2:
        state = optimizer.state[parameter]
        for axis in ("row", "col"):
            assert state[f"exp_avg_res_{axis}"].data_ptr() != state[f"exp_avg_sq_{axis}"].data_ptr()


@pytest.mark.parametrize("shape", [(6,), (2, 3), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("betas", [(0.9, 0.999, 0.9999), (0.4, 0.8, 0.7)])
def test_distributed_moments_and_updates_match_came_across_steps(shape, dtype, betas):
    parameter, optimizer, expected, reference = make_optimizers(shape, dtype, betas, weight_decay=0.1)

    for step in range(6):
        step_and_compare(parameter, optimizer, expected, reference, step)


@pytest.mark.parametrize("shape", [(6,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_continuation_preserves_distinct_moments(shape, dtype):
    parameter, optimizer, expected, reference = make_optimizers(shape, dtype, (0.4, 0.8, 0.7))
    for step in range(2):
        step_and_compare(parameter, optimizer, expected, reference, step)
    checkpoint = io.BytesIO()
    torch.save({"parameter": parameter.detach(), "optimizer": optimizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = torch.nn.Parameter(saved["parameter"])
    resumed_optimizer = DistributedCAME([resumed], lr=0.8)
    resumed_optimizer.setup_distributed(use_zero=False)
    resumed_optimizer.load_state_dict(saved["optimizer"])

    for step in range(2, 6):
        step_and_compare(resumed, resumed_optimizer, expected, reference, step)


def test_vector_momentum_keeps_the_unscaled_closed_form_recurrence():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = DistributedCAME([parameter], lr=0.1, betas=(0.5, 0.5, 0.5))
    optimizer.setup_distributed(use_zero=False)

    for moment, weight in ((0.5, 0.95), (0.75, 0.875), (0.875, 0.7875)):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.full_like(parameter, moment))
        torch.testing.assert_close(parameter, torch.full_like(parameter, weight))


@pytest.mark.parametrize("shape", [(6,), (2, 3)])
def test_first_parameter_update_keeps_existing_behavior(shape):
    parameter, optimizer, expected, reference = make_optimizers(shape, torch.float32, (0.9, 0.999, 0.9999))
    parameter.grad = torch.ones_like(parameter)
    expected.grad = torch.ones_like(expected)

    optimizer.step()
    reference.step()

    torch.testing.assert_close(parameter, expected)
