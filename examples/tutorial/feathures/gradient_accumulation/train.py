import os
from contextlib import nullcontext
from pathlib import Path

import torch
from titans.utils import barrier_context
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader


def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build resnet
    model = resnet18(num_classes=10)

    # build dataloaders
    with barrier_context():
        train_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),
                                download=True,
                                transform=transforms.Compose([
                                    transforms.RandomCrop(size=32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                                ]))

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # verify gradient accumulation
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, train_dataloader, _ = booster.boost(model=model,
                                                                     optimizer=optimizer,
                                                                     criterion=criterion,
                                                                     dataloader=train_dataloader)
    param_by_iter = []
    for idx, (img, label) in enumerate(train_dataloader):
        sync_context = booster.no_sync(model)
        img = img.cuda()
        label = label.cuda()
        model.zero_grad()
        if idx % (gpc.config.gradient_accumulation - 1) != 0:
            with sync_context:
                output = model(img)
                train_loss = criterion(output, label)
                booster.backward(train_loss, optimizer)
        else:
            output = model(img)
            train_loss = criterion(output, label)
            booster.backward(train_loss, optimizer)
            optimizer.step()
            model.zero_grad()

        ele_1st = next(model.parameters()).flatten()[0]
        param_by_iter.append(str(ele_1st.item()))

        if idx != 0 and idx % (gpc.config.gradient_accumulation - 1) == 0:
            break

    for iteration, val in enumerate(param_by_iter):
        print(f'iteration {iteration} - value: {val}')

    if param_by_iter[-1] != param_by_iter[0]:
        print('The parameter is only updated in the last iteration')


if __name__ == '__main__':
    main()
