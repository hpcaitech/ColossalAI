from functools import partial
from operator import imod
from colossalai.utils import checkpoint
import torch.nn as nn
import torch
from colossalai.logging import disable_existing_loggers, get_dist_logger

LOGGER = get_dist_logger()

CONFIG = dict(
    fp16=dict(
        mode=None,
    ),
    zero=dict(
        level=3,
        verbose=False,
        offload_optimizer_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            fast_init=False
        ),
        offload_param_config=dict(
            device='cpu',
            pin_memory=True,
            buffer_count=5,
            buffer_size=1e8,
            max_in_cpu=1e9
        )
    ),
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode=None)
    )
)

def checkpoint_wrapper(module, enable=True):
    if enable:
        module.forward = partial(checkpoint, module.forward)
    return module


class Net(nn.Module):
    def __init__(self, checkpoint=False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        if checkpoint:
            self.fc1 = checkpoint_wrapper(self.fc1)
        self.layers = [
            self.fc1,
            self.fc2,
            self.fc1,
            self.fc2,
            self.fc3
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def check_grads(model, zero_model, loose=False):
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_grad = zero_p.grad.clone().to(p.device)
        assert p.grad.dtype == zero_grad.dtype
        assert allclose(p.grad, zero_grad, loose=loose)
        LOGGER.info(torch.sum(p.grad-zero_grad))

def check_params(model, zero_model, loose=False):
    for p, zero_p in zip(model.parameters(), zero_model.parameters()):
        zero_p = zero_p.clone().to(p.device)
        assert p.dtype == zero_p.dtype
        assert allclose(p, zero_p, loose=loose)

