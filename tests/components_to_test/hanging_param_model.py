import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.nn import CheckpointModule

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


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


class DummyDataLoader(DummyDataGenerator):

    def generate(self):
        data = torch.rand(16, 4)
        label = torch.randint(low=0, high=2, size=(16,))
        return data, label


@non_distributed_component_funcs.register(name='hanging_param_model')
def get_training_components():

    def model_builder(checkpoint=False):
        return HangingParamModule(checkpoint)

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = torch.nn.CrossEntropyLoss()
    from colossalai.nn.optimizer import HybridAdam
    return model_builder, trainloader, testloader, HybridAdam, criterion
