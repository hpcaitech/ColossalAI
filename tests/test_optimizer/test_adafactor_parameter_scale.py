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
@pytest.mark.parametrize("magnitude", [0.0001, 1.0, 10.0])
@pytest.mark.parametrize("scale_parameter", [False, True])
def test_parameter_scale_tracks_current_weights(optimizer_type, shape, magnitude, scale_parameter):
    parameter = torch.nn.Parameter(torch.full(shape, magnitude))
    expected = torch.nn.Parameter(parameter.detach().clone())
    optimizer = make_optimizer(optimizer_type, [parameter], scale_parameter=scale_parameter, weight_decay=0.02)
    reference = ReferenceAdafactor([expected], scale_parameter=scale_parameter, weight_decay=0.02)
    for step in range(4):
        rms_before_step = parameter.detach().double().square().mean().sqrt().float()
        gradient = (torch.arange(6).reshape(shape) + step + 1).float()
        parameter.grad = gradient.clone()
        expected.grad = gradient.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(parameter, expected)
        if scale_parameter:
            torch.testing.assert_close(optimizer.state[parameter]["RMS"], rms_before_step)


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_parameter_norm_is_accumulated_in_float_for_low_precision_weights(optimizer_type, dtype):
    parameter = torch.nn.Parameter(torch.full((2, 3), 32768.0, dtype=dtype))
    expected = torch.nn.Parameter(parameter.detach().clone())
    optimizer = make_optimizer(optimizer_type, [parameter])
    reference = ReferenceAdafactor([expected])
    parameter.grad = torch.ones_like(parameter)
    expected.grad = torch.ones_like(expected)
    optimizer.step()
    reference.step()
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert optimizer.state[parameter]["RMS"] == 32768.0


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_parameter_scaled_checkpoint_continuation_matches_reference(optimizer_type):
    parameter = torch.nn.Parameter(torch.arange(1.0, 7.0).reshape(2, 3))
    expected = torch.nn.Parameter(parameter.detach().clone())
    options = dict(beta1=0.8, warmup_init=True, weight_decay=0.02)
    optimizer = make_optimizer(optimizer_type, [parameter], **options)
    reference = ReferenceAdafactor([expected], **options)
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        expected.grad = torch.ones_like(expected)
        optimizer.step()
        reference.step()
    checkpoint = io.BytesIO()
    torch.save({"parameter": parameter.detach(), "optimizer": optimizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = torch.nn.Parameter(saved["parameter"])
    resumed_optimizer = make_optimizer(optimizer_type, [resumed], **options)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    for step in range(3):
        for current, current_optimizer in ((parameter, optimizer), (resumed, resumed_optimizer), (expected, reference)):
            current.grad = torch.full_like(current, float(step + 1))
            current_optimizer.step()
        torch.testing.assert_close(resumed, parameter, rtol=0, atol=0)
        torch.testing.assert_close(resumed, expected)


@pytest.mark.parametrize("optimizer_type", [Adafactor, DistributedAdaFactor])
def test_enabling_parameter_scaling_uses_the_current_parameter_norm(optimizer_type):
    parameter = torch.nn.Parameter(torch.full((2, 3), 10.0))
    expected = torch.nn.Parameter(parameter.detach().clone())
    optimizer = make_optimizer(optimizer_type, [parameter], scale_parameter=False)
    reference = ReferenceAdafactor([expected], scale_parameter=False)
    for enabled in (False, True):
        for current, current_optimizer in ((parameter, optimizer), (expected, reference)):
            current_optimizer.param_groups[0]["scale_parameter"] = enabled
            current.grad = torch.ones_like(current)
            current_optimizer.step()
        torch.testing.assert_close(parameter, expected)
