#!/usr/bin/env python

import torch
import torch.nn as nn

from colossalai.nn import CheckpointModule

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


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


class DummyDataLoader(DummyDataGenerator):

    def generate(self):
        data = torch.rand(16, 5)
        label = torch.randint(low=0, high=2, size=(16,))
        return data, label


@non_distributed_component_funcs.register(name='repeated_computed_layers')
def get_training_components():

    def model_builder(checkpoint=False):
        return NetWithRepeatedlyComputedLayers(checkpoint)

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = torch.nn.CrossEntropyLoss()
    return model_builder, trainloader, testloader, torch.optim.Adam, criterion
