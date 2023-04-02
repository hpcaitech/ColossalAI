import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import colossalai
from colossalai.context import MOE_CONTEXT
from colossalai.nn.layer.moe import load_moe_model, save_moe_model
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.test_moe.test_moe_zero_init import MoeModel
from tests.test_tensor.common_utils import debug_print
from tests.test_zero.common import CONFIG


def exam_moe_checkpoint():
    with ColoInitContext(device=get_current_device()):
        model = MoeModel(checkpoint=True)
    save_moe_model(model, 'temp_path.pth')

    with ColoInitContext(device=get_current_device()):
        other_model = MoeModel(checkpoint=True)
    load_moe_model(other_model, 'temp_path.pth')

    state_0 = model.state_dict()
    state_1 = other_model.state_dict()
    for k, v in state_0.items():
        u = state_1.get(k)
        assert torch.equal(u.data, v.data)

    if dist.get_rank() == 0:
        os.remove('temp_path.pth')


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    exam_moe_checkpoint()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size):
    run_func = partial(_run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_checkpoint(world_size=4)
