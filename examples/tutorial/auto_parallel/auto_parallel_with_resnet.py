import argparse
import os
from pathlib import Path

import torch
from titans.utils import barrier_context
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from tqdm import tqdm

import colossalai
from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.utils import get_dataloader

DATA_ROOT = Path(os.environ.get('DATA', '../data')).absolute()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--synthetic', action="store_true", help="use synthetic dataset instead of CIFAR10")
    return parser.parse_args()


def synthesize_data():
    img = torch.rand(gpc.config.BATCH_SIZE, 3, 32, 32)
    label = torch.randint(low=0, high=10, size=(gpc.config.BATCH_SIZE,))
    return img, label


def main():
    args = parse_args()
    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    if not args.synthetic:
        with barrier_context():
            # build dataloaders
            train_dataset = CIFAR10(root=DATA_ROOT,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(size=32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                             std=[0.2023, 0.1994, 0.2010]),
                                    ]))

        test_dataset = CIFAR10(root=DATA_ROOT,
                               train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                               ]))

        train_dataloader = get_dataloader(
            dataset=train_dataset,
            add_sampler=True,
            shuffle=True,
            batch_size=gpc.config.BATCH_SIZE,
            pin_memory=True,
        )

        test_dataloader = get_dataloader(
            dataset=test_dataset,
            add_sampler=True,
            batch_size=gpc.config.BATCH_SIZE,
            pin_memory=True,
        )
    else:
        train_dataloader, test_dataloader = None, None

    # trace the model with meta data
    model = resnet50(num_classes=10).cuda()
    input_sample = {'x': torch.rand([gpc.config.BATCH_SIZE * torch.distributed.get_world_size(), 3, 32, 32]).to('meta')}

    model = autoparallelize(model, input_sample)
    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    for epoch in range(gpc.config.NUM_EPOCHS):
        model.train()

        if args.synthetic:
            # if we use synthetic data
            # we assume it only has 30 steps per epoch
            num_steps = range(30)

        else:
            # we use the actual number of steps for training
            num_steps = range(len(train_dataloader))
            data_iter = iter(train_dataloader)
        progress = tqdm(num_steps)

        for _ in progress:
            if args.synthetic:
                # generate fake data
                img, label = synthesize_data()
            else:
                # get the real data
                img, label = next(data_iter)

            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            train_loss = criterion(output, label)
            train_loss.backward(train_loss)
            optimizer.step()
        lr_scheduler.step()

        # run evaluation
        model.eval()
        correct = 0
        total = 0

        if args.synthetic:
            # if we use synthetic data
            # we assume it only has 10 steps for evaluation
            num_steps = range(30)

        else:
            # we use the actual number of steps for training
            num_steps = range(len(test_dataloader))
            data_iter = iter(test_dataloader)
        progress = tqdm(num_steps)

        for _ in progress:
            if args.synthetic:
                # generate fake data
                img, label = synthesize_data()
            else:
                # get the real data
                img, label = next(data_iter)

            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                output = model(img)
                test_loss = criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

        logger.info(
            f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}",
            ranks=[0])


if __name__ == '__main__':
    main()
