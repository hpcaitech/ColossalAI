import pytest
import torch
from torch import nn

from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import _route_padded_parameter_aliases
from colossalai.shardformer.layer.parallel_module import PaddingParallelModule


class PaddedLinear(PaddingParallelModule):
    def __init__(self, weight):
        super().__init__(new_num_embeddings=4, old_num_embeddings=3, weight=weight)

    @staticmethod
    def from_native_module(module, process_group=None):
        raise NotImplementedError


class AliasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_alias = nn.Parameter(torch.zeros(3, 2))
        self.parallel = PaddedLinear(self.weight_alias)


def test_alias_route_and_identity():
    model = AliasedModel()
    checkpoint = {"weight_alias": torch.arange(6).float().reshape(3, 2)}
    _route_padded_parameter_aliases(model, checkpoint)
    result = model.load_state_dict(checkpoint, strict=False)
    assert result.missing_keys == ["weight_alias"]
    assert not result.unexpected_keys
    assert model.weight_alias is model.parallel.weight
    torch.testing.assert_close(model.parallel.weight[:3], torch.arange(6).float().reshape(3, 2))
    assert torch.count_nonzero(model.parallel.weight[3:]) == 0


def test_conflicting_alias_values():
    state = {"weight_alias": torch.ones(3, 2), "parallel.weight": torch.zeros(3, 2)}
    with pytest.raises(ValueError, match="Conflicting checkpoint"):
        _route_padded_parameter_aliases(AliasedModel(), state)
    assert set(state) == {"weight_alias", "parallel.weight"}


def test_later_alias_conflict_does_not_mutate_checkpoint():
    model = nn.ModuleDict({"first": AliasedModel(), "second": AliasedModel()})
    state = {
        "first.weight_alias": torch.ones(3, 2),
        "second.weight_alias": torch.ones(3, 2),
        "second.parallel.weight": torch.zeros(3, 2),
    }
    before = dict(state)
    with pytest.raises(ValueError, match="Conflicting checkpoint"):
        _route_padded_parameter_aliases(model, state)
    assert state.keys() == before.keys()
    for name in before:
        assert state[name] is before[name]


def test_duplicate_module_aliases_are_routed():
    model = AliasedModel()
    model.other_parallel = model.parallel
    state = {"weight_alias": torch.ones(3, 2)}
    _route_padded_parameter_aliases(model, state)
    assert set(state) == {"parallel.weight", "other_parallel.weight"}
    model.load_state_dict(state, strict=False)
    assert model.parallel.weight is model.other_parallel.weight


def test_saver_aliases_with_padding_roundtrip():
    model = AliasedModel()
    with torch.no_grad():
        model.weight_alias[:3].fill_(2)
    state = model.state_dict()
    assert state["weight_alias"].shape == (4, 2)
    assert state["parallel.weight"].shape == (3, 2)
    _route_padded_parameter_aliases(model, state)
    target = AliasedModel()
    target.load_state_dict(state, strict=False)
    torch.testing.assert_close(target.parallel.weight, model.parallel.weight)


def test_nonzero_padding_is_not_silently_discarded():
    model = AliasedModel()
    state = {"weight_alias": torch.ones(4, 2)}
    with pytest.raises(ValueError, match="Nonzero padding"):
        _route_padded_parameter_aliases(model, state)
    assert set(state) == {"weight_alias"}


def test_strict_alias_missing_key_is_visible():
    model = AliasedModel()
    state = {"weight_alias": torch.ones(3, 2)}
    _route_padded_parameter_aliases(model, state)
    with pytest.raises(RuntimeError, match="weight_alias"):
        model.load_state_dict(state, strict=True)
