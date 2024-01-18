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
from colossalai.pipeline.schedule.interleaved_pp import InterleavedSchedule
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
    input_obj: torch.Tensor = None,
    stage_mgr: PipelineStageManager = None,
    model_chunk_id: int = None,
):
    with stage_mgr.switch_model_chunk_id(model_chunk_id):
        if stage_mgr.is_first_stage():
            return {"input_obj": forward(data)}
        elif stage_mgr.is_last_stage():
            return forward(input_obj)
        else:
            return {"input_obj": forward(input_obj)}


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
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host="localhost")

    # create model
    seed_all(1453)
    torch_model = MlpModel().cuda()
    pp_model = copy.deepcopy(torch_model).cuda()

    pg_mesh = ProcessGroupMesh(world_size)
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=num_model_chunk
    )
    schedule = InterleavedSchedule(
        stage_manager=stage_manager,
        num_model_chunks=num_model_chunk,
        num_microbatch=num_microbatch,
    )

    sharded_model = torch.nn.ModuleList()
    for idx, sub_model in enumerate(pp_model.layers):
        if idx % world_size == rank:
            sub_model._forward = sub_model.forward
            sub_model.forward = MethodType(
                partial(pp_linear_fwd, stage_mgr=stage_manager, model_chunk_id=len(sharded_model)),
                sub_model._forward,
            )
            sharded_model.append(sub_model.cuda())
    assert len(sharded_model) == num_model_chunk, "num_model_chunk is not correct"

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

    pp_ret = schedule.forward_backward_step(
        sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True, return_outputs=True
    )

    # check loss
    if stage_manager.is_last_stage(ignore_chunk=True):
        assert torch.allclose(torch_loss, pp_ret["loss"])

    # check gradients
    for i in range(num_model_chunk):
        idx = world_size * i + rank
        assert torch.allclose(torch_model.layers[idx].weight.grad, sharded_model[i].weight.grad)
        assert torch.allclose(torch_model.layers[idx].bias.grad, sharded_model[i].bias.grad)

    # step
    torch_optimizer.step()
    pp_optimizer.step()
    pp_optimizer.zero_grad()

    # check updated param
    for i in range(num_model_chunk):
        idx = world_size * i + rank
        assert torch.allclose(torch_model.layers[idx].weight, sharded_model[i].weight)
        assert torch.allclose(torch_model.layers[idx].bias, sharded_model[i].bias)

    # forward only
    with torch.no_grad():
        torch_output = torch_model(input_list[0])
        torch_loss = criterion(torch_output)

        pp_ret = schedule.forward_backward_step(
            sharded_model, iter(input_list), criterion, pp_optimizer, return_loss=True, return_outputs=True
        )
        if stage_manager.is_last_stage(ignore_chunk=True):
            assert torch.allclose(torch_loss, pp_ret["loss"])

        for layer in sharded_model:
            if layer.weight.grad is None:
                assert layer.weight.grad is None and layer.bias.grad is None
            else:
                assert torch.allclose(layer.weight.grad, torch.zeros_like(layer.weight.grad))
                assert torch.allclose(layer.bias.grad, torch.zeros_like(layer.bias.grad))


@pytest.mark.dist
@pytest.mark.parametrize("num_microbatch", [4, 12])
@pytest.mark.parametrize("batch_size", [12])
@pytest.mark.parametrize("num_model_chunk", [2, 4])
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
    test_pp(num_microbatch=4, batch_size=4, num_model_chunk=4)
