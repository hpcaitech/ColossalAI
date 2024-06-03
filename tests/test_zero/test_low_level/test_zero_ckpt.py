import copy

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(12, 24)
        self.linear2 = nn.Linear(24, 12)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def loose_close(a, b, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3

    a = a.detach().to(dtype)
    b = b.detach().to(dtype).to(a.device)

    assert_close(a, b, rtol=rtol, atol=atol)


def exam_zero_1_torch_ddp_ckpt():
    """
    We examine the state_dict of zero and DDP.
    Moreover, we examine the zero's loading checkpoint of a torch ckpt.
    """
    local_rank = torch.distributed.get_rank()
    seed_all(1453)

    # create models
    torch_model = MlpModel().cuda()
    zero_model = copy.deepcopy(torch_model)

    torch_model = DDP(torch_model.cuda(), static_graph=True).cuda()

    # create optimizer
    zero_optimizer = torch.optim.Adam(zero_model.parameters(), lr=1)

    # we only test stage 1 here
    # the state dicts of stage 1 and stage 2 are the same
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer, overlap_communication=True, initial_scale=1, reduce_bucket_size=262144
    )

    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1)

    seed_all(1453 + local_rank)
    # create
    input_data = torch.rand(4, 12).cuda()

    # forward
    zero_output = zero_model(input_data)
    torch_output = torch_model(input_data)

    # backward
    zero_optimizer.backward(zero_output.mean().float())
    torch_output.mean().backward()

    # step
    zero_optimizer.step()
    torch_optimizer.step()

    torch_state_dict = torch_optimizer.state_dict()
    zero_state_dict = zero_optimizer.state_dict()

    # examine the original state dict
    for torch_state, zero_state in zip(torch_state_dict["state"].values(), zero_state_dict["state"].values()):
        for t_v, z_v in zip(torch_state.values(), zero_state.values()):
            loose_close(t_v, z_v)

    # empty the optimzer state
    zero_optimizer.optim.state = []

    # zero load a torch checkpoint
    zero_optimizer.load_state_dict(copy.deepcopy(torch_state_dict))
    zero_state_dict = zero_optimizer.state_dict()

    # examine the loaded state dict
    for torch_state, zero_state in zip(torch_state_dict["state"].values(), zero_state_dict["state"].values()):
        for t_v, z_v in zip(torch_state.values(), zero_state.values()):
            loose_close(t_v, z_v)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    exam_zero_1_torch_ddp_ckpt()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_ckpt():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_zero_ckpt()
