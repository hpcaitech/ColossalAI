import os

from torchvision.datasets import CIFAR10


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(dir_path, "data")
    dataset = CIFAR10(root=data_root, download=True)


if __name__ == "__main__":
    main()
