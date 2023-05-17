import os
import time
from cmath import log
from pathlib import Path

import torch
from timm.models import vit_base_patch16_224
from titans.utils import barrier_context
from torchvision import datasets, transforms

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader


def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()


def main():
    # initialize distributed setting
    parser = colossalai.get_default_parser()
    args = parser.parse_args()

    # launch from torch
    colossalai.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # build model
    model = vit_base_patch16_224(drop_rate=0.1)

    # build dataloader
    with barrier_context():
        train_dataset = datasets.CIFAR10(root=Path(os.environ.get('DATA', './data')),
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ]))

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # lr_scheduler
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=1, total_steps=gpc.config.NUM_EPOCHS)

    plugin = TorchDDPPlugin()
    booster = Booster(mixed_precision='fp16', plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
    logger.info("initialized colossalai components", ranks=[0])

    # train
    model.train()
    for epoch in range(gpc.config.NUM_EPOCHS):
        start = get_time_stamp()
        for img, label in train_dataloader:
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            booster.backward(loss, optimizer)
            optimizer.step()
        lr_scheduler.step()
        end = get_time_stamp()
        avg_step_time = (end - start) / len(train_dataloader)
        logger.info('epoch: {}, loss: {}, avg step time: {} / s'.format(epoch, loss.item(), avg_step_time))

    gpc.destroy()


if __name__ == '__main__':
    main()
