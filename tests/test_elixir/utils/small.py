import torch
import torch.nn as nn

from tests.test_elixir.utils.mlp import MlpModule
from tests.test_elixir.utils.registry import TEST_MODELS


def small_data_fn():
    return dict(x=torch.randint(low=0, high=20, size=(4, 8)))


class SmallModel(nn.Module):

    def __init__(self, num_embeddings: int = 20, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpModule(hidden_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_embeddings, bias=False)
        self.proj.weight = self.embed.weight

    def forward(self, x):
        x = self.embed(x)
        x = x + self.norm1(self.mlp(x))
        x = self.proj(self.norm2(x))
        x = x.mean(dim=-2)
        return x.sum()


TEST_MODELS.register('small', SmallModel, small_data_fn)
