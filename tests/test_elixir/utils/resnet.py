import torch
import torch.nn as nn
from torchvision.models import resnet18

from tests.test_elixir.utils.registry import TEST_MODELS


def resnet_data_fn():
    return dict(x=torch.randn(4, 3, 32, 32))


class ResNetModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.r = resnet18()

    def forward(self, x):
        output = self.r(x)
        return output.sum()


TEST_MODELS.register('resnet', ResNetModel, resnet_data_fn)
