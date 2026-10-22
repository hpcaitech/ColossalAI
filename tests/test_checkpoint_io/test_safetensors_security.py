"""Regression tests for issue #6471.

The async optimizer-checkpoint loader used to unpickle whichever entries a
checkpoint file's own metadata named, with an unrestricted unpickler -- a
crafted .safetensors file could execute arbitrary code on load. These tests
check that (a) legitimate optimizer state dicts still round-trip and
(b) a malicious checkpoint is rejected without executing its payload.
"""

import json
import os
import pickle

import pytest
import torch
from safetensors.torch import save_file

from colossalai.utils.safetensors import _flatten_optim_state_dict, _unflatten_optim_state_dict, load_flat


class _Payload:
    def __reduce__(self):
        return (os.system, ("echo pwned > PWNED.txt",))


def _write_malicious_checkpoint(path, key):
    payload = pickle.dumps(_Payload(), protocol=2)
    tensors = {
        "param_groups": torch.tensor([0], dtype=torch.uint8),
        key: torch.tensor(list(payload), dtype=torch.uint8),
    }
    save_file(tensors, str(path), metadata={"non_tensor_keys": json.dumps([key])})


def test_malicious_checkpoint_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "optimizer.safetensors"
    # A key with the exact shape the writer produces, carrying pickle payload bytes.
    _write_malicious_checkpoint(checkpoint, "state.0.candidate")
    with pytest.raises(pickle.UnpicklingError):
        load_flat(str(checkpoint))
    assert not (tmp_path / "PWNED.txt").exists()


def test_off_schema_non_tensor_key_rejected(tmp_path):
    checkpoint = tmp_path / "optimizer.safetensors"
    _write_malicious_checkpoint(checkpoint, "totally.bogus.key.extra")
    with pytest.raises(ValueError):
        load_flat(str(checkpoint))


def test_optimizer_state_dict_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95))
    for _ in range(2):
        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    state_dict = optimizer.state_dict()
    flat, metadata = _flatten_optim_state_dict(state_dict)
    restored = _unflatten_optim_state_dict(dict(flat), {"non_tensor_keys": json.dumps(metadata["non_tensor_keys"])})
    assert restored["param_groups"] == state_dict["param_groups"]
    for idx, fields in state_dict["state"].items():
        for name, value in fields.items():
            restored_value = restored["state"][idx][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored_value, value)
            else:
                assert restored_value == value

    checkpoint = tmp_path / "optimizer.safetensors"
    save_file(flat, str(checkpoint), metadata={"non_tensor_keys": json.dumps(metadata["non_tensor_keys"])})
    assert load_flat(str(checkpoint))["param_groups"] == state_dict["param_groups"]


def test_tensor_nested_in_non_tensor_value_round_trips():
    state_dict = {
        "state": {0: {"step": 3, "bundle": {"vec": torch.tensor([1.0, 2.0]), "name": "x"}}},
        "param_groups": [{"lr": 0.1, "params": [0]}],
    }
    flat, metadata = _flatten_optim_state_dict(state_dict)
    restored = _unflatten_optim_state_dict(dict(flat), {"non_tensor_keys": json.dumps(metadata["non_tensor_keys"])})
    bundle = restored["state"][0]["bundle"]
    assert bundle["name"] == "x"
    assert torch.equal(bundle["vec"], torch.tensor([1.0, 2.0]))
