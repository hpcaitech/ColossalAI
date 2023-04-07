import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.tensor import ProcessGroup
from colossalai.testing import spawn
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext, LowLevelZeroOptimizer


class MlpModel(nn.Module):

    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def exam_zero_init():
    dp_2_tp_2_pg = ProcessGroup(dp_degree=2, tp_degree=2)
    model1 = MlpModel().cuda()
    with ColoInitContext(device=get_current_device(), default_pg=dp_2_tp_2_pg):
        model2 = MlpModel()
    optimizer1 = LowLevelZeroOptimizer(torch.optim.Adam(model1.parameters(), lr=1))
    optimizer2 = LowLevelZeroOptimizer(torch.optim.Adam(model2.parameters(), lr=1))

    assert optimizer1._local_rank == optimizer2._local_rank
    assert optimizer1._world_size == optimizer2._world_size
    assert optimizer1._dp_global_ranks == optimizer2._dp_global_ranks

    mp_group1 = optimizer1._mp_torch_group
    mp_group2 = optimizer2._mp_torch_group
    assert dist.get_world_size(mp_group1) == dist.get_world_size(mp_group2)
    assert dist.get_rank(mp_group1) == dist.get_rank(mp_group2)


def run_dist(rank, world_size, port):
    config_dict = dict(parallel=dict(data=2, tensor=dict(size=2, mode='1d')))
    colossalai.launch(config=config_dict, rank=rank, world_size=world_size, port=port, host='localhost')
    exam_zero_init()


@pytest.mark.dist
def test_zero_init():
    spawn(run_dist, 4)


if __name__ == '__main__':
    test_zero_init()
