import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port


class MLP(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, 4 * in_features, bias=False)
        self.linear_2 = torch.nn.Linear(4 * in_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)

        return x


def check_compatibility_with_ddp(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = MLP(4).cuda()
    input = torch.rand(4, 4).cuda()
    output_compare = model(input)
    loss_compare = output_compare.sum()
    loss_compare.backward()
    grad_compare = copy.deepcopy(model.linear_1.weight.grad)

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    meta_args = {'x': torch.rand(4, 4).to('meta')}
    gm, solution = initialize_model(model,
                                    meta_args=meta_args,
                                    device_mesh=device_mesh,
                                    return_solution=True,
                                    solver_preference='tp',
                                    shard_option='shard_last_axis')

    msg = '| TP strategy combination chosen by auto-parallel solver |'
    msg_length = len(msg)
    if rank == 0:
        print('=' * msg_length)
        print(msg)
        print('=' * msg_length)
        for strategy in solution:
            print(strategy)
        print('=' * msg_length)

    dp_process_group = None
    for (ranks, process_group_handle) in device_mesh.process_groups_dict[0]:
        if rank in ranks:
            dp_process_group = process_group_handle
    assert dp_process_group is not None
    gm = DDP(gm, process_group=dp_process_group)
    output = gm(input)

    assert_close(output, output_compare)
    print(f'output on rank{rank} is correct')
    loss = output.sum()

    loss.backward()

    if rank in (0, 2):
        assert_close(gm.module.module.linear_1.weight.grad, grad_compare.narrow(0, 0, 8))

    if rank in (1, 3):
        assert_close(gm.module.module.linear_1.weight.grad, grad_compare.narrow(0, 8, 8))

    print(f'gradient on rank{rank} is correct')


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_compatibility_with_ddp():
    world_size = 4
    run_func = partial(check_compatibility_with_ddp, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_compatibility_with_ddp()
