import torch
import torch.nn as nn

from tests.test_elixir.utils.registry import TEST_MODELS


def mlp_data_fn():
    return dict(x=torch.randn(4, 16))


class MlpModule(nn.Module):

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        return x + (self.proj2(self.act(self.proj1(x))))


class MlpModel(nn.Module):

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = MlpModule(hidden_dim)

    def forward(self, x):
        output = self.mlp(x)
        return output.sum()


TEST_MODELS.register('mlp', MlpModel, mlp_data_fn)
