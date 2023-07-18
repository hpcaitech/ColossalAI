import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ==============================
# Parse Arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=80, help="resume from the epoch's checkpoint")
parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint', help="checkpoint directory")
args = parser.parse_args()

# ==============================
# Prepare Test Dataset
# ==============================
# CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor())

# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# ==============================
# Load Model
# ==============================
model = torchvision.models.resnet18(num_classes=10).cuda()
state_dict = torch.load(f'{args.checkpoint}/model_{args.epoch}.pth')
model.load_state_dict(state_dict)

# ==============================
# Run Evaluation
# ==============================
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
