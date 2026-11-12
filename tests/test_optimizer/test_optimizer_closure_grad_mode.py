import pytest
import torch

from colossalai.nn.optimizer import Adafactor, DistributedAdaFactor, DistributedCAME, DistributedLamb

OPTIMIZERS = [Adafactor, DistributedAdaFactor, DistributedCAME, DistributedLamb]


def make_optimizer(optimizer_type, parameters):
    options = {"scale_parameter": False} if optimizer_type in (Adafactor, DistributedAdaFactor) else {"lr": 0.01}
    optimizer = optimizer_type(parameters, **options)
    if isinstance(optimizer, (DistributedAdaFactor, DistributedCAME)):
        optimizer.setup_distributed(use_zero=False)
    elif isinstance(optimizer, DistributedLamb):
        optimizer.setup_distributed(is_zero=False)
    return optimizer


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize("shape", [(6,), (2, 3)])
@pytest.mark.parametrize("outer_grad_enabled", [False, True])
def test_closure_backward_and_updates_match_explicit_gradients(optimizer_type, shape, outer_grad_enabled):
    parameter = torch.nn.Parameter(torch.arange(1.0, 7.0).reshape(shape))
    expected = torch.nn.Parameter(parameter.detach().clone())
    optimizer = make_optimizer(optimizer_type, [parameter])
    reference = make_optimizer(optimizer_type, [expected])
    for step in range(4):
        target = 0.2 * step
        reference.zero_grad()
        reference_loss = (expected - target).square().mean()
        reference_loss.backward()
        reference.step()
        calls = []

        def closure():
            calls.append(torch.is_grad_enabled())
            optimizer.zero_grad()
            loss = (parameter - target).square().mean()
            loss.backward()
            return loss

        with torch.set_grad_enabled(outer_grad_enabled):
            result = optimizer.step(closure)
            assert torch.is_grad_enabled() is outer_grad_enabled
        assert calls == [True]
        assert result.requires_grad
        torch.testing.assert_close(result, reference_loss)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_closure_exception_restores_callers_disabled_grad_mode(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = make_optimizer(optimizer_type, [parameter])
    grad_modes = []

    def closure():
        grad_modes.append(torch.is_grad_enabled())
        raise ValueError("closure failed")

    with torch.no_grad():
        with pytest.raises(ValueError, match="closure failed"):
            optimizer.step(closure)
        assert not torch.is_grad_enabled()
    assert grad_modes == [True]
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    assert not optimizer.state


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_no_closure_keeps_existing_none_return_and_parameter_updates(optimizer_type):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = make_optimizer(optimizer_type, [parameter])
    parameter.grad = torch.ones_like(parameter)

    result = optimizer.step()

    assert result is None
    assert torch.isfinite(parameter).all()
    assert torch.all(parameter < 1.0)
