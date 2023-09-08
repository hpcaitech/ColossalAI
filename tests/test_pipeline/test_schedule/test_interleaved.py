import copy
from functools import partial
from types import MethodType

import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.schedule.interleaved_pp import InterleavedSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all


class MlpModel(nn.Module):

    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 8)
        self.linear4 = nn.Linear(8, 8)
        self.linear5 = nn.Linear(8, 8)
        self.linear6 = nn.Linear(8, 8)
        self.linear7 = nn.Linear(8, 8)
        self.linear8 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        return x


def pp_linear_fwd(forward,
                  data: torch.Tensor = None,
                  input_obj: torch.Tensor = None,
                  stage_mgr: PipelineStageManager = None,
                  num_chunks: int = None,
                  model_chunk_id: int = None):

    if stage_mgr.is_first_stage() and model_chunk_id == 0:
        return {'input_obj': forward(data)}
    elif stage_mgr.is_last_stage() and model_chunk_id == num_chunks - 1:
        return forward(input_obj)
    else:
        return {'input_obj': forward(input_obj)}


@parameterize("num_micro_batches", [4, 8, 12])
def examine_pp(num_micro_batches):
    """
    This test is to examine the correctness of interleaved 1F1B, compared with torch.
    Be aware it contains some hardcodes.
    """
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    seed_all(1453)

    NUM_MICRO_BATCHS = num_micro_batches
    BATCH_SIZE = num_micro_batches
    NUM_CHUNKS = 2

    # create model
    torch_model = MlpModel().cuda()

    pp_model = copy.deepcopy(torch_model).cuda()

    DP_DIM, PP_DIM, TP_DIM = 0, 1, 2
    pg_mesh = ProcessGroupMesh(1, world_size, 1)
    stage_manager = PipelineStageManager(pg_mesh, PP_DIM, is_virtual=True)
    schedule = InterleavedSchedule(NUM_MICRO_BATCHS, NUM_CHUNKS, stage_manager)

    sharded_model = torch.nn.ModuleList()
    for idx, (_, sub_model) in enumerate(pp_model.named_children()):
        if idx % (world_size) == local_rank:
            sub_model._forward = sub_model.forward
            sub_model.forward = MethodType(
                partial(pp_linear_fwd,
                        stage_mgr=stage_manager,
                        num_chunks=NUM_CHUNKS,
                        model_chunk_id=len(sharded_model)), sub_model._forward)
            sharded_model.append(sub_model.cuda())

    # create optimizer
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)
    pp_optimizer = OptimizerWrapper(torch.optim.SGD(sharded_model.parameters(), lr=1))

    # create
    seed_all(1453)
    if local_rank == 0:
        input_list = [torch.rand(BATCH_SIZE, 4).cuda()]
    else:
        input_list = [torch.zeros(BATCH_SIZE, 4).cuda()]
    torch.distributed.all_reduce(input_list[0])

    criterion = lambda x, y: torch.mean(x)

    # forward and backward
    torch_output = torch_model(input_list[0])
    torch_loss = criterion(torch_output, _)
    torch_loss.backward()

    pp_ret = schedule.forward_backward_step(sharded_model,
                                            iter(input_list),
                                            criterion,
                                            pp_optimizer,
                                            return_loss=True,
                                            return_outputs=True)

    # check loss
    if stage_manager.is_last_stage():
        assert torch.allclose(torch_loss, pp_ret['loss'])

    # check gradients
    torch_grad = []
    for torch_p in torch_model.parameters():
        torch_grad.append(torch_p.grad.data)

    for idx, pp_p in enumerate(sharded_model.parameters()):
        if idx < 2:
            assert torch.allclose(torch_grad[idx + local_rank * 2], pp_p.grad.data)
        else:
            assert torch.allclose(torch_grad[idx + local_rank * 2 + 6], pp_p.grad.data)

    # step
    torch_optimizer.step()
    pp_optimizer.step()

    # check updated param
    torch_param = []
    for torch_p in torch_model.parameters():
        torch_param.append(torch_p.data)
    for idx, pp_p in enumerate(sharded_model.parameters()):
        if idx < 2:
            assert torch.allclose(torch_param[idx + local_rank * 2], pp_p.data)
        else:
            assert torch.allclose(torch_param[idx + local_rank * 2 + 6], pp_p.data)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    examine_pp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp():
    spawn(run_dist, 4)


if __name__ == '__main__':
    test_pp()
