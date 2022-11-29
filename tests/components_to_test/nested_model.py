import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.nn import CheckpointModule

from .registry import non_distributed_component_funcs
from .utils import DummyDataGenerator


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


class DummyDataLoader(DummyDataGenerator):

    def generate(self):
        data = torch.rand(16, 5)
        label = torch.randint(low=0, high=2, size=(16,))
        return data, label


@non_distributed_component_funcs.register(name='nested_model')
def get_training_components():

    def model_builder(checkpoint=False):
        return NestedNet(checkpoint)

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = torch.nn.CrossEntropyLoss()
    return model_builder, trainloader, testloader, torch.optim.Adam, criterion
