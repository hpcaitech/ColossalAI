from torchvision.models import resnet18
from .registry import non_distributed_component_funcs
from pathlib import Path
import os
import torch
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from colossalai.utils import get_dataloader


def get_cifar10_dataloader(train):
    # build dataloaders
    dataset = CIFAR10(root=Path(os.environ['DATA']),
                      download=True,
                      train=train,
                      transform=transforms.Compose(
                          [transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))
    dataloader = get_dataloader(dataset=dataset, shuffle=True, batch_size=16, drop_last=True)
    return dataloader


@non_distributed_component_funcs.register(name='resnet18')
def get_resnet_training_components():

    def model_builder(checkpoint=False):
        return resnet18(num_classes=10)

    trainloader = get_cifar10_dataloader(train=True)
    testloader = get_cifar10_dataloader(train=False)

    criterion = torch.nn.CrossEntropyLoss()
    return model_builder, trainloader, testloader, torch.optim.Adam, criterion
