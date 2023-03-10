import torch

OUTPUT_SAVED_OPS = [torch.nn.functional.relu, torch.nn.functional.softmax, torch.flatten]

OUTPUT_SAVED_MOD = [
    torch.nn.ReLU,
    torch.nn.Softmax,
]

# SHAPE_ARGUMENT_OPS contains node with (input, *shape) style args.
# This list could be extended if any other method has the same
# argument style as view and reshape.
SHAPE_ARGUMENT_OPS = [torch.Tensor.view, torch.Tensor.reshape, torch.reshape]
