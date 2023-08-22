import os

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.context import MOE_CONTEXT
from colossalai.nn.layer.moe import load_moe_model, save_moe_model
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from tests.test_moe.moe_utils import MoeModel


def exam_moe_checkpoint():
    model = MoeModel(checkpoint=True).to(get_current_device())
    save_moe_model(model, 'temp_path.pth')

    other_model = MoeModel(checkpoint=True).to(get_current_device())
    load_moe_model(other_model, 'temp_path.pth')

    state_0 = model.state_dict()
    state_1 = other_model.state_dict()
    for k, v in state_0.items():
        u = state_1.get(k)
        assert torch.equal(u.data, v.data)

    if dist.get_rank() == 0:
        os.remove('temp_path.pth')


def _run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    exam_moe_checkpoint()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size):
    spawn(_run_dist, world_size)


if __name__ == '__main__':
    test_moe_checkpoint(world_size=4)
