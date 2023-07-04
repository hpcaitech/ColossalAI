import copy

import pytest
import torch

try:
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
    NO_CODEGEN = False
except:
    NO_CODEGEN = True

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor.process_group import ProcessGroup
from colossalai.testing import assert_close, rerun_if_address_is_in_use, run_on_environment_flag, spawn
from colossalai.utils import get_current_device
from colossalai.zero import post_process_colo_init_ctx, zero_model_wrapper, zero_optim_wrapper


class MLP(torch.nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, 4 * in_features, bias=False)
        self.linear_2 = torch.nn.Linear(4 * in_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)

        return x


def check_auto_parallel_with_gemini(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = MLP(4).half().cuda()
    if rank in [0, 1]:
        input = torch.arange(0, 16).reshape(4, 4).half().cuda()
    elif rank in [2, 3]:
        input = torch.arange(16, 32).reshape(4, 4).half().cuda()
    input_compare = torch.arange(0, 32).reshape(8, 4).half().cuda()
    output_compare = model(input_compare)
    loss_compare = output_compare.sum()
    loss_compare.backward()
    grad_compare = copy.deepcopy(model.linear_1.weight.grad / 2)

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    meta_args = {'x': torch.rand(4, 4).half().to('meta')}
    gm, solution = initialize_model(model,
                                    meta_args=meta_args,
                                    device_mesh=device_mesh,
                                    return_solution=True,
                                    solver_preference='tp',
                                    shard_option='shard_last_axis')

    if rank == 0:
        msg = '| TP strategy combination chosen by auto-parallel solver |'
        msg_length = len(msg)
        print('=' * msg_length)
        print(msg)
        print('=' * msg_length)
        for strategy in solution:
            print(strategy)
        print('=' * msg_length)

    dp_process_group = ProcessGroup(rank=rank, ranks=[0, 1, 2, 3], tp_degree=2, dp_degree=2)
    gemini_config = dict(strict_ddp_mode=False,
                         device=get_current_device(),
                         placement_policy='cpu',
                         pin_memory=True,
                         search_range_m=128)

    post_process_colo_init_ctx(gm, device=get_current_device(), default_pg=dp_process_group)
    gm = zero_model_wrapper(gm, zero_stage=3, gemini_config=gemini_config)
    optimizer = HybridAdam(gm.parameters(), betas=(0, 0))
    optimizer = zero_optim_wrapper(gm, optimizer, initial_scale=1)
    output = gm(input)
    if rank in (0, 1):
        assert_close(output, output_compare.narrow(0, 0, 4))
    else:
        assert_close(output, output_compare.narrow(0, 4, 4))
    print(f'output on rank{rank} is correct')
    loss = output.sum()
    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()

    if rank in (0, 2):
        assert_close(list(optimizer.optim.state.values())[0]['exp_avg'].half(), grad_compare.narrow(0, 0, 8).flatten())

    if rank in (1, 3):
        assert_close(list(optimizer.optim.state.values())[0]['exp_avg'].half(), grad_compare.narrow(0, 8, 8).flatten())

    print(f'gradient on rank{rank} is correct')


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.skipif(NO_CODEGEN, reason='No codegen found')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_auto_parallel_with_gemini():
    spawn(check_auto_parallel_with_gemini, 4)


if __name__ == '__main__':
    test_auto_parallel_with_gemini()
