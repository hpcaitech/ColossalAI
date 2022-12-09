import torch
import torch.nn as nn

from colossalai.nn import CheckpointModule
from colossalai.utils.cuda import get_current_device

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


class MlpModel(CheckpointModule):
    """A simple MLP model used for testings.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.proj1 = nn.Linear(32, 128)
        self.activate = nn.GELU()
        self.proj2 = nn.Linear(128, 32)

    def forward(self, x):
        x = self.proj1(x)
        x = self.activate(x)
        x = self.proj2(x)
        return x


class SelfLoss(nn.Module):
    """Loss function used for MlpModel. Set mean_flag to False to get big gradients.
    """

    def forward(self, logits, label, mean_flag: bool = True, *args, **kwargs):
        if mean_flag:
            return logits.mean()
        else:
            return logits.sum()


class DummyDataLoader(DummyDataGenerator):

    def generate(self):
        data = torch.randn(32, 32, device=get_current_device())
        label = 0
        return data, label


@non_distributed_component_funcs.register(name='mlp')
def get_training_components():

    def model_builder(checkpoint=False):
        return MlpModel(checkpoint)

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = SelfLoss()
    from colossalai.nn.optimizer import HybridAdam
    return model_builder, trainloader, testloader, HybridAdam, criterion
