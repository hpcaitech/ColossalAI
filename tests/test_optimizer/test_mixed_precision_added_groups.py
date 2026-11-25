import pytest
import torch

from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("parameter_input", ["list", "generator", "tensor"])
def test_added_groups_match_native_master_weight_updates(dtype, optimizer_type, parameter_input):
    parameters = [torch.nn.Parameter(torch.linspace(0.5, 1.5, 6).reshape(2, 3).to(dtype)) for _ in range(2)]
    expected = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    masters = [torch.nn.Parameter(parameter.detach().float().clone()) for parameter in parameters]
    options = {"momentum": 0.8} if optimizer_type is torch.optim.SGD else {"betas": (0.7, 0.9)}
    optimizer = MixedPrecisionOptimizer(optimizer_type([parameters[0]], lr=0.1, **options), precision="bf16")
    reference = optimizer_type([{"params": [masters[0]], "lr": 0.1}, {"params": [masters[1]], "lr": 0.03}], **options)

    for step in range(6):
        if step == 2:
            new_params = {
                "list": [parameters[1]],
                "generator": (parameter for parameter in [parameters[1]]),
                "tensor": parameters[1],
            }[parameter_input]
            optimizer.add_param_group({"params": new_params, "lr": 0.03})
        optimizer.zero_grad()
        reference.zero_grad()
        for parameter in expected:
            parameter.grad = None
        active = [0, 1] if step >= 2 and step != 4 else [0]
        target = 0.2 * step
        optimizer.backward(sum((parameters[index].float() - target).square().mean() for index in active))
        sum((expected[index].float() - target).square().mean() for index in active).backward()
        for index in active:
            masters[index].grad = expected[index].grad.float()
        reference.step()
        optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(expected, masters):
                parameter.copy_(master)
        for parameter, value in zip(parameters, expected):
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
        for parameter, master in zip(parameters[: len(optimizer.param_groups)], masters):
            actual_master = optimizer.working_to_master_map[parameter]
            torch.testing.assert_close(actual_master, master, rtol=0, atol=0)
            assert optimizer.master_to_working_map[actual_master] is parameter
        if step >= 2:
            assert optimizer.param_groups[1]["lr"] == 0.03
            assert optimizer.param_groups[1]["params"][0].dtype == torch.float32
            if dtype == torch.bfloat16:
                assert optimizer.param_groups[1]["params"][0] is not parameters[1]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_repeated_working_parameter_rejected_without_adding_a_group(dtype):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    optimizer = MixedPrecisionOptimizer(torch.optim.SGD([parameter], lr=0.1), precision="bf16")
    original_master = optimizer.working_to_master_map[parameter]

    with pytest.raises(ValueError, match="more than one parameter group"):
        optimizer.add_param_group({"params": [parameter]})

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"][0] is original_master
    assert optimizer.working_to_master_map[parameter] is original_master
    optimizer.backward(parameter.sum())
    optimizer.step()
    assert torch.all(parameter < 1)


@pytest.mark.parametrize("invalid_input", ["set", "non_tensor", "non_leaf"])
def test_underlying_optimizer_validation_does_not_change_mappings(invalid_input):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    optimizer = MixedPrecisionOptimizer(torch.optim.SGD([parameter], lr=0.1), precision="bf16")
    new_parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    inputs = {"set": {new_parameter}, "non_tensor": [1], "non_leaf": [new_parameter * 2]}
    error = ValueError if invalid_input == "non_leaf" else TypeError

    with pytest.raises(error):
        optimizer.add_param_group({"params": inputs[invalid_input]})

    assert len(optimizer.param_groups) == 1
    assert len(optimizer.working_to_master_map) == 1
    assert len(optimizer.master_to_working_map) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_added_frozen_parameters_keep_constructor_filtering(dtype):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    frozen = torch.nn.Parameter(torch.full((3,), 2.0, dtype=dtype), requires_grad=False)
    new_parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    optimizer = MixedPrecisionOptimizer(torch.optim.SGD([parameter], lr=0.1), precision="bf16")

    optimizer.add_param_group({"params": [frozen, new_parameter], "lr": 0.2})
    optimizer.backward((parameter + new_parameter).sum())
    optimizer.step()
    optimizer.zero_grad()

    assert frozen not in optimizer.working_to_master_map
    assert len(optimizer.param_groups[1]["params"]) == 1
    torch.testing.assert_close(frozen, torch.full_like(frozen, 2.0), rtol=0, atol=0)
    assert parameter.grad is None
    assert new_parameter.grad is None
    assert all(master.grad is None for master in optimizer.parameters)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_added_groups_participate_in_gradient_clipping(dtype):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    new_parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    optimizer = MixedPrecisionOptimizer(torch.optim.SGD([parameter], lr=0.1), precision="bf16", max_norm=0.25)
    optimizer.add_param_group({"params": [new_parameter], "lr": 0.2})

    optimizer.backward((parameter + 2 * new_parameter).sum())
    optimizer.step()

    norm = 15.0**0.5
    assert optimizer.get_grad_norm() == pytest.approx(norm)
    for actual, rate, gradient in ((parameter, 0.1, 1.0), (new_parameter, 0.2, 2.0)):
        expected = torch.full((3,), 1.0 - rate * gradient * 0.25 / (norm + 1e-6)).to(dtype)
        torch.testing.assert_close(actual, expected)
