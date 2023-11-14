import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import model_zoo
from .base import CheckpointModule


class SimpleNet(CheckpointModule):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.embed = nn.Embedding(20, 4)
        self.proj1 = nn.Linear(4, 8)
        self.ln1 = nn.LayerNorm(8)
        self.proj2 = nn.Linear(8, 4)
        self.ln2 = nn.LayerNorm(4)
        self.classifier = nn.Linear(4, 4)

    def forward(self, x):
        x = self.embed(x)
        x = self.proj1(x)
        x = self.ln1(x)
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.classifier(x)
        return x


def data_gen():
    return dict(x=torch.randint(low=0, high=20, size=(16,)))


def loss_fn(x):
    outputs = x["x"]
    label = torch.randint(low=0, high=2, size=(16,), device=outputs.device)
    return F.cross_entropy(x["x"], label)


def output_transform(x: torch.Tensor):
    return dict(x=x)


model_zoo.register(
    name="custom_simple_net",
    model_fn=SimpleNet,
    data_gen_fn=data_gen,
    output_transform_fn=output_transform,
    loss_fn=loss_fn,
)
