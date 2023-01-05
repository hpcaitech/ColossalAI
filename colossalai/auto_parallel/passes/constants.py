import torch

OUTPUT_SAVED_OPS = [torch.nn.functional.relu, torch.nn.functional.softmax, torch.flatten]

OUTPUT_SAVED_MOD = [
    torch.nn.ReLU,
    torch.nn.Softmax,
]
