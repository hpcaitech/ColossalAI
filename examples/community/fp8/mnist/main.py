# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

try:
    from transformer_engine import pytorch as te

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


class Net(nn.Module):
    def __init__(self, use_te=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if use_te:
            self.fc1 = te.Linear(9216, 128)
            self.fc2 = te.Linear(128, 16)
        else:
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        """FWD"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, use_fp8):
    """Training function."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with te.fp8_autocast(enabled=use_fp8):
            output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
            if args.dry_run:
                break


def calibrate(model, device, test_loader):
    """Calibration function."""
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with te.fp8_autocast(enabled=False, calibrating=True):
                model(data)


def test(model, device, test_loader, use_fp8):
    """Testing function."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with te.fp8_autocast(enabled=use_fp8):
                output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--use-fp8", action="store_true", default=False, help="Use FP8 for inference and training without recalibration"
    )
    parser.add_argument("--use-fp8-infer", action="store_true", default=False, help="Use FP8 inference only")
    parser.add_argument("--use-te", action="store_true", default=False, help="Use Transformer Engine")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    if args.use_te or args.use_fp8 or args.use_fp8_infer:
        assert HAVE_TE, "TransformerEngine not installed."

    if args.use_fp8 or args.use_fp8_infer:
        args.use_te = True

    if args.use_te:
        assert use_cuda, "CUDA needed for FP8 execution."

    if args.use_fp8_infer:
        assert not args.use_fp8, "fp8-infer path currently only supports calibration from a bfloat checkpoint"

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(use_te=args.use_te).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.use_fp8)
        test(model, device, test_loader, args.use_fp8)
        scheduler.step()

    if args.use_fp8_infer:
        calibrate(model, device, test_loader)

    if args.save_model or args.use_fp8_infer:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Eval with reloaded checkpoint : fp8=" + str(args.use_fp8_infer))
        weights = torch.load("mnist_cnn.pt")
        model.load_state_dict(weights)
        test(model, device, test_loader, args.use_fp8_infer)


if __name__ == "__main__":
    main()
