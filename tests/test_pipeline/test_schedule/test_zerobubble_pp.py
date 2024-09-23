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
from colossalai.pipeline.schedule.v_schedule import PipelineGraph
from colossalai.pipeline.schedule.zero_bubble_pp import ZeroBubbleVPipeScheduler
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

NUM_LAYER = 8
DIM = 4


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
    hidden_states: torch.Tensor = None,
    stage_mgr: PipelineStageManager = None,
    model_chunk_id: int = None,
):
    with stage_mgr.switch_model_chunk_id(model_chunk_id):
        if stage_mgr.is_first_stage():
            return {"hidden_states": forward(data)}
        elif stage_mgr.is_last_stage():
            return forward(hidden_states)
        else:
            return {"hidden_states": forward(hidden_states)}


def run_pp(
    rank: int,
    world_size: int,
    port: int,
    num_microbatch: int,
    batch_size: int,
    num_model_chunk: int,
):
    """
    This test is to examine the correctness of interleaved 1F1B, compared with torch.
    Be aware it contains some hardcodes.
    """
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    # create model
    seed_all(1453)
    torch_model = MlpModel().cuda()
    pp_model = copy.deepcopy(torch_model).cuda()

    pg_mesh = ProcessGroupMesh(world_size)
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, use_zbv=True, num_model_chunks=num_model_chunk
    )

    # schedule list
    mem_f = 34 * 32 + 5 * 4 * 16
    mem_w = -32 * 32
    mem_b = -mem_w - mem_f
    scheduler_nodes = PipelineGraph(
        n_stage=4,
        n_micro=12,
        f_cost=1000,
        b_cost=1000,
        w_cost=1000,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
    ).get_v_schedule()
    schedule = ZeroBubbleVPipeScheduler(
        stage_manager=stage_manager,
        schedule=scheduler_nodes,
        num_model_chunks=num_model_chunk,
        num_microbatch=num_microbatch,
    )

    sharded_model = torch.nn.ModuleList()
    for idx, sub_model in enumerate(pp_model.layers):
        if idx == rank or (NUM_LAYER - idx - 1) == rank:
            sub_model._forward = sub_model.forward
            sub_model.forward = MethodType(
                partial(pp_linear_fwd, stage_mgr=stage_manager, model_chunk_id=len(sharded_model)),
                sub_model._forward,
            )
            sharded_model.append(sub_model.cuda())
    assert (
        len(sharded_model) == num_model_chunk
    ), f"{len(sharded_model)}, {num_model_chunk}, num_model_chunk is not correct"

    # create optimizer
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-5)
    pp_optimizer = OptimizerWrapper(torch.optim.SGD(sharded_model.parameters(), lr=1e-5))

    # create data
    seed_all(115)
    input_list = [torch.rand(batch_size, DIM).cuda()]
    dist.all_reduce(input_list[0])

    def criterion(x, *args, **kwargs):
        return (x * x).mean()

    # forward and backward
    torch_output = torch_model(input_list[0])
    torch_loss = criterion(torch_output)
    torch_loss.backward()

    pp_ret = schedule.forward_backward_step(sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True)

    # check loss
    if stage_manager.is_first_stage(ignore_chunk=True):
        assert_close(torch_loss, pp_ret["loss"])

    # check gradients
    for i in range(num_model_chunk):
        # idx = world_size * i + rank
        if i == 0:
            idx = rank
        else:
            idx = world_size * 2 - rank - 1
        print(f"{i=}, {idx=}, {rank=}, {torch_model.layers[idx].weight.grad=}, {sharded_model[i].weight.grad=}")
        assert_close(torch_model.layers[idx].weight.grad, sharded_model[i].weight.grad)
        assert_close(torch_model.layers[idx].bias.grad, sharded_model[i].bias.grad)

    # step
    torch_optimizer.step()
    pp_optimizer.step()
    pp_optimizer.zero_grad()

    # check updated param
    for i in range(num_model_chunk):
        # idx = world_size * i + rank
        if i == 0:
            idx = rank
        else:
            idx = world_size * 2 - rank - 1
        assert_close(torch_model.layers[idx].weight, sharded_model[i].weight)
        assert_close(torch_model.layers[idx].bias, sharded_model[i].bias)

    # forward one step
    torch_output = torch_model(input_list[0])
    torch_loss = criterion(torch_output)

    pp_ret = schedule.forward_backward_step(
        sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True
    )
    if stage_manager.is_first_stage(ignore_chunk=True):
        print(f"{torch_loss=}, {pp_ret['loss']}")
        assert_close(torch_loss, pp_ret["loss"])


@pytest.mark.dist
@pytest.mark.parametrize("num_microbatch", [12])
@pytest.mark.parametrize("batch_size", [24])
@pytest.mark.parametrize("num_model_chunk", [2])
@rerun_if_address_is_in_use()
def test_pp(num_microbatch: int, batch_size: int, num_model_chunk: int):
    assert NUM_LAYER % num_model_chunk == 0
    spawn(
        run_pp,
        nprocs=NUM_LAYER // num_model_chunk,
        num_microbatch=num_microbatch,
        batch_size=batch_size,
        num_model_chunk=num_model_chunk,
    )


if __name__ == "__main__":
    test_pp(num_microbatch=4, batch_size=12, num_model_chunk=2)
