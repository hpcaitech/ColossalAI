import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import model_zoo
from .base import CheckpointModule


class HangingParamModule(CheckpointModule):
    """
    Hanging Parameter: a parameter dose not belong to a leaf Module.
    It has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.proj1 = nn.Linear(4, 8)
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.proj2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.proj1(x)
        x = F.linear(x, self.weight)
        x = self.proj2(x)
        return x


def data_gen():
    return dict(x=torch.rand(16, 4))


def loss_fn(x):
    outputs = x["x"]
    label = torch.randint(low=0, high=2, size=(16,), device=outputs.device)
    return F.cross_entropy(x["x"], label)


def output_transform(x: torch.Tensor):
    return dict(x=x)


model_zoo.register(
    name="custom_hanging_param_model",
    model_fn=HangingParamModule,
    data_gen_fn=data_gen,
    output_transform_fn=output_transform,
    loss_fn=loss_fn,
)
