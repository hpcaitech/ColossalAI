import torch
import torch.nn as nn

from colossalai.nn import CheckpointModule
from colossalai.utils.cuda import get_current_device

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


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


class DummyDataLoader(DummyDataGenerator):

    def generate(self):
        data = torch.randint(low=0, high=20, size=(16,), device=get_current_device())
        label = torch.randint(low=0, high=2, size=(16,), device=get_current_device())
        return data, label


@non_distributed_component_funcs.register(name='simple_net')
def get_training_components():

    def model_builder(checkpoint=False):
        return SimpleNet(checkpoint)

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = torch.nn.CrossEntropyLoss()
    from colossalai.nn.optimizer import HybridAdam
    return model_builder, trainloader, testloader, HybridAdam, criterion
