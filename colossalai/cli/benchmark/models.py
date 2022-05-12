import torch
import colossalai.nn as col_nn


class MLP(torch.nn.Module):

    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            self.layers.append(col_nn.Linear(dim, dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
