import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm

import colossalai
from colossalai.legacy.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import Lamb, Lars


class DummyDataloader:
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def generate(self):
        data = torch.rand(self.batch_size, 3, 224, 224)
        label = torch.randint(low=0, high=10, size=(self.batch_size,))
        return data, label

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def main():
    # initialize distributed setting
    parser = colossalai.legacy.get_default_parser()
    parser.add_argument(
        "--optimizer", choices=["lars", "lamb"], help="Choose your large-batch optimizer", required=True
    )
    args = parser.parse_args()

    # launch from torch
    colossalai.legacy.launch_from_torch(config=args.config)

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # create synthetic dataloaders
    train_dataloader = DummyDataloader(length=10, batch_size=gpc.config.BATCH_SIZE)
    test_dataloader = DummyDataloader(length=5, batch_size=gpc.config.BATCH_SIZE)

    # build model
    model = resnet18(num_classes=gpc.config.NUM_CLASSES)

    # create loss function
    criterion = nn.CrossEntropyLoss()

    # create optimizer
    if args.optimizer == "lars":
        optim_cls = Lars
    elif args.optimizer == "lamb":
        optim_cls = Lamb
    optimizer = optim_cls(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=gpc.config.WARMUP_EPOCHS
    )

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.legacy.initialize(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    logger.info("Engine is built", ranks=[0])

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()
        data_iter = iter(train_dataloader)

        if gpc.get_global_rank() == 0:
            description = "Epoch {} / {}".format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            engine.zero_grad()
            engine.execute_schedule(data_iter, return_output_label=False)
            engine.step()
            lr_scheduler.step()


if __name__ == "__main__":
    main()
