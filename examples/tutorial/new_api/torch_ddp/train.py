import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.cluster import DistCoordinator

# ==============================
# Parse Arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resume', type=int, default=-1, help="resume from the epoch's checkpoint")
parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint', help="checkpoint directory")
parser.add_argument('-i', '--interval', type=int, default=5, help="interval of saving checkpoint")
parser.add_argument('-f', '--fp16', action='store_true', help="use fp16")
args = parser.parse_args()

# ==============================
# Prepare Checkpoint Directory
# ==============================
Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
START_EPOCH = args.resume if args.resume >= 0 else 0

# ==============================
# Launch Distributed Environment
# ==============================
colossalai.launch_from_torch(config={})
coordinator = DistCoordinator()

# update the learning rate with linear scaling
# old_gpu_num / old_lr = new_gpu_num / new_lr
LEARNING_RATE *= coordinator.world_size

# ==============================
# Prepare Booster
# ==============================
plugin = TorchDDPPlugin()
if args.fp16:
    booster = Booster(mixed_precision='fp16', plugin=plugin)
else:
    booster = Booster(plugin=plugin)

# ==============================
# Prepare Train Dataset
# ==============================
transform = transforms.Compose(
    [transforms.Pad(4),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32),
     transforms.ToTensor()])

# CIFAR-10 dataset
with coordinator.priority_execution():
    train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)

# ====================================
# Prepare model, optimizer, criterion
# ====================================
# resent50
model = torchvision.models.resnet18(num_classes=10).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# lr scheduler
lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

# prepare dataloader with torch ddp plugin
train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=100, shuffle=True)

# ==============================
# Resume from checkpoint
# ==============================
if args.resume >= 0:
    booster.load_model(model, f'{args.checkpoint}/model_{args.resume}.pth')
    booster.load_optimizer(optimizer, f'{args.checkpoint}/optimizer_{args.resume}.pth')
    booster.load_lr_scheduler(lr_scheduler, f'{args.checkpoint}/lr_scheduler_{args.resume}.pth')

# ==============================
# Boost with ColossalAI
# ==============================
model, optimizer, criterion, train_dataloader, lr_scheduler = booster.boost(model, optimizer, criterion,
                                                                            train_dataloader, lr_scheduler)

# ==============================
# Train model
# ==============================
total_step = len(train_dataloader)

for epoch in range(START_EPOCH, NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        booster.backward(loss, optimizer)
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, NUM_EPOCHS, i + 1, total_step,
                                                                    loss.item()))

    lr_scheduler.step()

    # save checkpoint every 5 epoch
    if (epoch + 1) % args.interval == 0:
        booster.save_model(model, f'{args.checkpoint}/model_{epoch + 1}.pth')
        booster.save_optimizer(optimizer, f'{args.checkpoint}/optimizer_{epoch + 1}.pth')
        booster.save_lr_scheduler(lr_scheduler, f'{args.checkpoint}/lr_scheduler_{epoch + 1}.pth')
