from collections import defaultdict
from copy import deepcopy

import pytest
import torch
from packaging.version import Version

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.checkpoint_io.utils import load_param_groups_into_optimizer, save_param_groups

LEGACY_UNSUPPORTED = pytest.mark.skipif(
    Version(torch.__version__) < Version("2.10.0"), reason="Legacy optimizer loading requires PyTorch >= 2.10"
)


def make_optimizer(kind, parameters, groups=1):
    if groups == 2:
        parameters = [{"params": [parameters[0]], "lr": 0.1}, {"params": [parameters[1]], "lr": 0.03}]
    options = {"lr": 0.1, "weight_decay": 0.02}
    if kind == "sgd":
        return torch.optim.SGD(parameters, momentum=0.9, **options)
    return {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind](parameters, betas=(0.7, 0.8), **options)


def make_parameters(dtype=torch.float32):
    return [torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=dtype)) for _ in range(2)]


def take_step(optimizer, parameters, step):
    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.tensor([0.4 + step * 0.2, -0.7 + index * 0.1], dtype=parameter.dtype)
    optimizer.step()


def assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], dict):
            assert_same_state(actual[key], expected[key])
        elif isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sharded_empty_checkpoint_replaces_old_state_and_training_updates(tmp_path, kind, groups, dtype):
    parameters = make_parameters(dtype)
    optimizer = make_optimizer(kind, parameters, groups)
    saved_state = deepcopy(optimizer.state_dict())
    checkpoint = str(tmp_path / "optimizer")
    checkpoint_io = GeneralCheckpointIO()
    checkpoint_io.save_optimizer(optimizer, checkpoint, shard=True)

    for step in range(3):
        take_step(optimizer, parameters, step)
    assert len(optimizer.state) == 2
    reference_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    reference = make_optimizer(kind, reference_parameters, groups)
    reference.load_state_dict(saved_state)
    for group in optimizer.param_groups:
        group["lr"] = 0.9

    checkpoint_io.load_optimizer(optimizer, checkpoint)

    assert not optimizer.state
    assert optimizer.state_dict()["param_groups"] == reference.state_dict()["param_groups"]
    for step in range(3, 6):
        take_step(optimizer, parameters, step)
        take_step(reference, reference_parameters, step)
        for actual, expected in zip(parameters, reference_parameters):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_same_state(optimizer.state_dict()["state"], reference.state_dict()["state"])


@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("legacy", [False, pytest.param(True, marks=LEGACY_UNSUPPORTED)])
def test_loading_groups_prepares_fresh_state_and_parameter_mapping(tmp_path, kind, legacy):
    parameters = make_parameters()
    optimizer = make_optimizer(kind, parameters, groups=2)
    take_step(optimizer, parameters, 0)
    optimizer.state["old_extra_state"] = {"count": 7}
    saved_state = deepcopy(optimizer.state_dict())
    saved_state["param_groups"][0]["params"] = [10]
    saved_state["param_groups"][1]["params"] = [11]
    saved_state["param_groups"][0]["lr"] = 0.02
    group_file = tmp_path / ("groups.bin" if legacy else "groups.safetensors")
    if legacy:
        torch.save(saved_state["param_groups"], group_file)
    else:
        save_param_groups(saved_state, str(group_file))

    mapping = load_param_groups_into_optimizer(optimizer, str(group_file))

    assert mapping.keys() == {10, 11}
    assert mapping[10] is parameters[0]
    assert mapping[11] is parameters[1]
    assert optimizer.param_groups[0]["params"][0] is parameters[0]
    assert optimizer.param_groups[1]["params"][0] is parameters[1]
    assert optimizer.param_groups[0]["lr"] == 0.02
    assert not optimizer.state
    assert isinstance(optimizer.state, defaultdict)
    assert optimizer.state.default_factory is dict


@pytest.mark.parametrize("invalid", ["group_count", "parameter_count", "type"])
@LEGACY_UNSUPPORTED
def test_incompatible_groups_do_not_discard_existing_state(tmp_path, invalid):
    parameters = make_parameters()
    optimizer = make_optimizer("sgd", parameters)
    take_step(optimizer, parameters, 0)
    before = deepcopy(optimizer.state_dict())
    original_groups, original_state = optimizer.param_groups, optimizer.state
    groups = deepcopy(before["param_groups"])
    if invalid == "group_count":
        groups.append(deepcopy(groups[0]))
    elif invalid == "parameter_count":
        groups[0]["params"].append(2)
    else:
        groups = {"unexpected": groups}
    group_file = tmp_path / "groups.bin"
    torch.save(groups, group_file)

    with pytest.raises(ValueError):
        load_param_groups_into_optimizer(optimizer, str(group_file))

    assert optimizer.param_groups is original_groups
    assert optimizer.state is original_state
    assert_same_state(optimizer.state_dict()["state"], before["state"])
    assert optimizer.state_dict()["param_groups"] == before["param_groups"]


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
@LEGACY_UNSUPPORTED
def test_unsharded_empty_checkpoint_already_replaces_old_state(tmp_path, kind):
    parameters = make_parameters()
    optimizer = make_optimizer(kind, parameters)
    checkpoint = str(tmp_path / "optimizer.pt")
    checkpoint_io = GeneralCheckpointIO()
    checkpoint_io.save_optimizer(optimizer, checkpoint)
    take_step(optimizer, parameters, 0)

    checkpoint_io.load_optimizer(optimizer, checkpoint)

    assert not optimizer.state


@pytest.mark.parametrize("kind", ["sgd", "adam", "adamw"])
def test_sharded_empty_checkpoint_loads_into_a_fresh_optimizer(tmp_path, kind):
    optimizer = make_optimizer(kind, make_parameters())
    checkpoint = str(tmp_path / "optimizer")
    checkpoint_io = GeneralCheckpointIO()
    checkpoint_io.save_optimizer(optimizer, checkpoint, shard=True)

    checkpoint_io.load_optimizer(optimizer, checkpoint)

    assert not optimizer.state
