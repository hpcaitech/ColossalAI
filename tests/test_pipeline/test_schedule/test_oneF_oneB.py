import copy
from functools import partial
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

DIM = 8
NUM_LAYER = 8


class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(NUM_LAYER)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def pp_linear_fwd(
    forward,
    data: torch.Tensor = None,
    input_obj: torch.Tensor = None,
    stage_mgr: PipelineStageManager = None,
):
    if stage_mgr.is_first_stage():
        return {"input_obj": forward(data)}
    elif stage_mgr.is_last_stage():
        return forward(input_obj)
    else:
        return {"input_obj": forward(input_obj)}


def examine_pp(num_microbatch: int, batch_size: int):
    """
    This test is to examine the correctness of 1F1B, compared with torch.
    Be aware it contains some hardcodes.
    """
    world_size = dist.get_world_size()
    dist.get_rank()
    seed_all(1453)

    # create models
    torch_model = MlpModel().cuda()

    pp_model = copy.deepcopy(torch_model).cuda()

    pg_mesh = ProcessGroupMesh(world_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)
    schedule = OneForwardOneBackwardSchedule(stage_manager, num_microbatches=num_microbatch)

    rank = dist.get_rank()
    sharded_model = torch.nn.ModuleList()
    num_local_layer = NUM_LAYER // world_size
    for idx, sub_model in enumerate(pp_model.layers):
        if idx // num_local_layer == rank:
            sharded_model.append(sub_model.cuda())
    assert len(sharded_model) == num_local_layer

    def custom_fwd(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

    sharded_model._forward = MethodType(custom_fwd, sharded_model)
    sharded_model.forward = MethodType(
        partial(
            pp_linear_fwd,
            stage_mgr=stage_manager,
        ),
        sharded_model._forward,
    )

    # create optimizer
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)
    pp_optimizer = OptimizerWrapper(torch.optim.SGD(sharded_model.parameters(), lr=1))

    # create
    seed_all(1453)
    input_list = [torch.rand(batch_size, DIM).cuda()]
    dist.all_reduce(input_list[0])

    criterion = lambda x, *arg, **kwargs: (x * x).mean()

    # forward and backward
    torch_output = torch_model(input_list[0])
    torch_loss = criterion(torch_output)
    torch_loss.backward()
    pp_ret = schedule.forward_backward_step(sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True)

    # check loss
    if stage_manager.is_last_stage():
        assert_close(torch_loss, pp_ret["loss"])

    # check gradients
    for i in range(len(sharded_model)):
        idx = rank * num_local_layer + i
        assert_close(torch_model.layers[idx].weight.grad, sharded_model[i].weight.grad)
        assert_close(torch_model.layers[idx].bias.grad, sharded_model[i].bias.grad)

    # step
    torch_optimizer.step()
    pp_optimizer.step()
    pp_optimizer.zero_grad()

    # check updated param
    for i in range(len(sharded_model)):
        idx = rank * num_local_layer + i
        assert_close(torch_model.layers[idx].weight, sharded_model[i].weight)
        assert_close(torch_model.layers[idx].bias, sharded_model[i].bias)

    # forward only
    with torch.no_grad():
        torch_output = torch_model(input_list[0])
        torch_loss = criterion(torch_output)

        pp_ret = schedule.forward_backward_step(
            sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True
        )
        if stage_manager.is_last_stage():
            assert_close(torch_loss, pp_ret["loss"])

        for layer in sharded_model:
            if layer.weight.grad is None:
                assert layer.weight.grad is None and layer.bias.grad is None
            else:
                assert_close(layer.weight.grad, torch.zeros_like(layer.weight.grad))
                assert_close(layer.bias.grad, torch.zeros_like(layer.bias.grad))


def run_dist(
    rank: int,
    world_size: int,
    port: int,
    num_microbatch: int,
    batch_size: int,
):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    examine_pp(num_microbatch, batch_size)


@pytest.mark.dist
@pytest.mark.parametrize("num_microbatch", [4, 6])
@pytest.mark.parametrize("batch_size", [12])
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_pp(num_microbatch: int, batch_size: int, world_size: int):
    assert NUM_LAYER % world_size == 0
    spawn(
        run_dist,
        world_size,
        num_microbatch=num_microbatch,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    test_pp(num_microbatch=4, batch_size=4, world_size=4)
