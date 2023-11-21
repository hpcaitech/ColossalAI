import copy
from functools import partial
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

WORLD_SIZE = 2
DIM = 8
NUM_MICRO_BATCHS = 4
BATCH_SIZE = 4


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(DIM, DIM)
        self.linear2 = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def pp_linear_fwd(
    forward, data: torch.Tensor = None, input_obj: torch.Tensor = None, stage_mgr: PipelineStageManager = None
):
    if stage_mgr.is_first_stage():
        return {"input_obj": forward(data)}
    elif stage_mgr.is_last_stage():
        return forward(input_obj)
    else:
        return {"input_obj": forward(input_obj)}


def examine_pp():
    """
    This test is to examine the correctness of 1F1B, compared with torch.
    Be aware it contains some hardcodes.
    """
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    seed_all(1453)

    # create models
    torch_model = MlpModel().cuda()

    pp_model = copy.deepcopy(torch_model).cuda()

    DP_DIM, PP_DIM, TP_DIM = 0, 1, 2
    pg_mesh = ProcessGroupMesh(1, world_size, 1)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    schedule = OneForwardOneBackwardSchedule(stage_manager, num_microbatches=NUM_MICRO_BATCHS)

    for idx, (_, sub_model) in enumerate(pp_model.named_children()):
        if idx % (world_size) == local_rank:
            sharded_model = sub_model.cuda()

    sharded_model._forward = sharded_model.forward
    sharded_model.forward = MethodType(partial(pp_linear_fwd, stage_mgr=stage_manager), sharded_model._forward)

    # create optimizer
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)
    pp_optimizer = OptimizerWrapper(torch.optim.SGD(sharded_model.parameters(), lr=1))

    # create
    seed_all(1453)
    input_list = [torch.rand(BATCH_SIZE, DIM).cuda()]
    dist.all_reduce(input_list[0])

    criterion = lambda x, y: (x * x).mean()

    # forward and backward
    torch_output = torch_model(input_list[0])
    torch_loss = criterion(torch_output, _)
    torch_loss.backward()

    pp_ret = schedule.forward_backward_step(
        sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True, return_outputs=True
    )

    # check loss
    if stage_manager.is_last_stage():
        assert torch.allclose(torch_loss, pp_ret["loss"])

    # check gradients
    torch_grad = []
    for torch_p in torch_model.parameters():
        torch_grad.append(torch_p.grad.data)
    for idx, pp_p in enumerate(sharded_model.parameters()):
        assert torch.allclose(torch_grad[idx + local_rank * 2], pp_p.grad.data)

    # step
    torch_optimizer.step()
    pp_optimizer.step()

    # check updated param
    torch_param = []
    for torch_p in torch_model.parameters():
        torch_param.append(torch_p.data)
    for idx, pp_p in enumerate(sharded_model.parameters()):
        assert torch.allclose(torch_param[idx + local_rank * 2], pp_p.data)


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host="localhost")
    examine_pp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp():
    spawn(run_dist, WORLD_SIZE)


if __name__ == "__main__":
    test_pp()
