from functools import partial

import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.booster.accelerator import Accelerator
from colossalai.testing import parameterize, rerun_if_address_is_in_use


@parameterize('device', ['cpu', 'cuda'])
def run_accelerator(device):
    acceleartor = Accelerator(device)
    model = nn.Linear(8, 8)
    model = acceleartor.configure_model(model)
    assert next(model.parameters()).device.type == device
    del model, acceleartor


def run_dist(rank):
    run_accelerator()


@rerun_if_address_is_in_use()
def test_accelerator():
    world_size = 1
    run_func = partial(run_dist)
    mp.spawn(run_func, nprocs=world_size)
