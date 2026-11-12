import pytest
import torch

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.checkpoint_io.utils import load_state_dict_into_model


@pytest.mark.cpu
@pytest.mark.parametrize("use_safetensors", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_sharded_model_rejects_incompatible_parameter_shapes(tmp_path, use_safetensors, strict, nested):
    source = torch.nn.Linear(2, 4)
    target = torch.nn.Linear(2, 3)
    if nested:
        source = torch.nn.Sequential(source)
        target = torch.nn.Sequential(target)
    checkpoint_io = GeneralCheckpointIO()
    checkpoint_io.save_model(source, str(tmp_path), shard=True, use_safetensors=use_safetensors)

    with pytest.raises(RuntimeError, match="size mismatch for .*weight"):
        checkpoint_io.load_model(target, str(tmp_path), strict=strict)


@pytest.mark.cpu
@pytest.mark.parametrize("strict", [False, True])
def test_sharded_model_reports_invalid_tensor_values(strict):
    model = torch.nn.Linear(2, 3)
    state = {"weight": "invalid checkpoint tensor", "bias": torch.ones(3)}

    with pytest.raises(RuntimeError, match="expected torch.Tensor or Tensor-like object"):
        load_state_dict_into_model(model, state, [], strict=strict)


@pytest.mark.cpu
@pytest.mark.parametrize("use_safetensors", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_valid_model_still_loads_across_multiple_shards(tmp_path, use_safetensors, strict):
    source = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    target = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    checkpoint_io = GeneralCheckpointIO()
    checkpoint_io.save_model(source, str(tmp_path), shard=True, size_per_shard=0.00008, use_safetensors=use_safetensors)

    checkpoint_io.load_model(target, str(tmp_path), strict=strict)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    extension = "*.safetensors" if use_safetensors else "*.bin"
    assert len(list(tmp_path.glob(extension))) > 1


@pytest.mark.cpu
@pytest.mark.parametrize("strict", [False, True])
def test_unexpected_keys_keep_their_strictness_behavior(strict):
    model = torch.nn.Linear(2, 3)
    state = dict(model.state_dict(), unrelated=torch.ones(1))

    if strict:
        with pytest.raises(RuntimeError, match="Unexpected key"):
            load_state_dict_into_model(model, state, [], strict=True)
    else:
        load_state_dict_into_model(model, state, [], strict=False)
