import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.nn import CheckpointModule

class SimpleNet(CheckpointModule):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.embed = nn.Embedding(2048, 512)
        self.proj1 = nn.Linear(512, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.proj2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, 512)

    def forward(self, x):
        x = self.embed(x)
        x = self.proj1(x)
        x = self.ln1(x)
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.classifier(x)
        return x


class NetWithRepeatedlyComputedLayers(CheckpointModule):
    """
    This model is to test with layers which go through forward pass multiple times.
    In this model, the fc1 and fc2 call forward twice
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SubNet(nn.Module):

    def __init__(self, out_features) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, weight):
        return F.linear(x, weight, self.bias)


class NestedNet(CheckpointModule):

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint)
        self.fc1 = nn.Linear(1024, 1024)
        self.sub_fc = SubNet(1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sub_fc(x, self.fc1.weight)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NoLeafModule(CheckpointModule):
    """
    In this no-leaf module, it has subordinate nn.modules and a nn.Parameter.
    """

    def __init__(self, checkpoint=False) -> None:
        super().__init__(checkpoint=checkpoint)
        self.proj1 = nn.Linear(1024, 2048)
        self.weight = nn.Parameter(torch.randn(2048, 2048))
        self.proj2 = nn.Linear(2048, 1024)

    def forward(self, x):
        x = self.proj1(x)
        x = F.linear(x, self.weight)
        x = self.proj2(x)
        return x
