import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import model_zoo
from .base import CheckpointModule


class NetWithRepeatedlyComputedLayers(CheckpointModule):
    """
    This model is to test with layers which go through forward pass multiple times.
    In this model, the fc1 and fc2 call forward twice
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 2)
        self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def data_gen():
    return dict(x=torch.rand(16, 5))


def loss_fn(x):
    outputs = x["x"]
    label = torch.randint(low=0, high=2, size=(16,), device=outputs.device)
    return F.cross_entropy(x["x"], label)


def output_transform(x: torch.Tensor):
    return dict(x=x)


model_zoo.register(
    name="custom_repeated_computed_layers",
    model_fn=NetWithRepeatedlyComputedLayers,
    data_gen_fn=data_gen,
    output_transform_fn=output_transform,
    loss_fn=loss_fn,
)
