import torch
from torch import nn

from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import _route_padded_parameter_aliases
from colossalai.shardformer.layer.parallel_module import PaddingParallelModule


class _PaddedLinear(PaddingParallelModule):
    def __init__(self, weight):
        super().__init__(new_num_embeddings=4, old_num_embeddings=3, weight=weight)

    @staticmethod
    def from_native_module(module, process_group=None):
        raise NotImplementedError

    def forward(self, value):
        return value @ self.weight.T


class _AliasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = nn.Parameter(torch.zeros(3, 2))
        self.weight_alias = weight
        self.parallel = _PaddedLinear(weight)


def test_load_padded_parameter_through_parallel_alias():
    model = _AliasedModel()
    checkpoint = {"weight_alias": torch.arange(6, dtype=torch.float32).reshape(3, 2)}

    _route_padded_parameter_aliases(model, checkpoint)
    result = model.load_state_dict(checkpoint, strict=False)

    assert result.missing_keys == ["weight_alias"]
    assert not result.unexpected_keys
    expected = torch.tensor([[0, 1], [2, 3], [4, 5], [0, 0]], dtype=torch.float32)
    assert torch.equal(model.parallel.weight, expected)
    assert model.weight_alias is model.parallel.weight
