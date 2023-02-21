import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        return x


def check_apply(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    input = torch.rand(4, 4, 4, 4).cuda()
    test_input = copy.deepcopy(input)
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
    #     return conv
    model = ConvModel(4, 4).cuda()
    test_model = copy.deepcopy(model)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    meta_args = {'x': torch.rand(4, 4, 4, 4).to('meta')}
    gm = initialize_model(model, meta_args, device_mesh)

    output = gm(input)
    origin_output = test_model(test_input)
    assert output.equal(origin_output)
    origin_loss = origin_output.sum()
    loss = output.sum()

    origin_loss.backward()
    loss.backward()

    grad_0 = test_model.conv.weight.grad.narrow(0, 0, 1)
    grad_1 = test_model.conv.weight.grad.narrow(0, 1, 1)
    grad_2 = test_model.conv.weight.grad.narrow(0, 2, 1)
    grad_3 = test_model.conv.weight.grad.narrow(0, 3, 1)

    if rank == 0:
        assert_close(gm.module.conv.weight.grad.data, grad_0.data)
    elif rank == 1:
        assert_close(gm.module.conv.weight.grad.data, grad_1.data)
    elif rank == 2:
        assert_close(gm.module.conv.weight.grad.data, grad_2.data)
    elif rank == 3:
        assert_close(gm.module.conv.weight.grad.data, grad_3.data)
    else:
        raise ValueError(f'rank {rank} does not exist.')


# skip this test due to pulp not installed in CI environment
@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_apply():
    world_size = 4
    run_func = partial(check_apply, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_apply()
