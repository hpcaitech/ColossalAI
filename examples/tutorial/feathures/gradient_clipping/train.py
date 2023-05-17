import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.utils import get_dataloader


def main():
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    # build resnet
    model = resnet34(num_classes=10)

    # build dataloaders
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
        num_workers=1,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    plugin = TorchDDPPlugin()
    booster = Booster(mixed_precision='fp16', plugin=plugin)

    model, optimizer, criterion, train_dataloader, lr_scheduler = booster.boost(model, optimizer, criterion,
                                                                                train_dataloader, lr_scheduler)

    # engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    #     model,
    #     optimizer,
    #     criterion,
    #     train_dataloader,
    # )

    # verify gradient accumulation
    model.train()
    for idx, (img, label) in enumerate(train_dataloader):
        img = img.cuda()
        label = label.cuda()

        model.zero_grad()
        output = model(img)
        train_loss = criterion(output, label)
        booster.backward(train_loss, optimizer)
        optimizer.clip_grad_by_norm(max_norm=gpc.config.gradient_clipping)
        optimizer.step()
        lr_scheduler.step()

        ele_1st = next(model.parameters()).flatten()[0]
        logger.info(f'iteration {idx}, loss: {train_loss}, 1st element of parameters: {ele_1st.item()}')

        # only run for 4 iterations
        if idx == 3:
            break


if __name__ == '__main__':
    main()
