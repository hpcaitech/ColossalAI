import torch
from packaging import version

assert version(torch.__version__) >= version.parse(
    '1.10'), "We require at least 1.10 pytorch version when using graph module for a stable torch.fx interface."
