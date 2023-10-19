import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import model_zoo
from .base import CheckpointModule


class SubNet(nn.Module):
    def __init__(self, out_features) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, weight):
        return F.linear(x, weight, self.bias)


class NestedNet(CheckpointModule):
    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint)
        self.fc1 = nn.Linear(5, 5)
        self.sub_fc = SubNet(5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sub_fc(x, self.fc1.weight)
        x = self.fc1(x)
        x = self.fc2(x)
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
    name="custom_nested_model",
    model_fn=NestedNet,
    data_gen_fn=data_gen,
    output_transform_fn=output_transform,
    loss_fn=loss_fn,
)
