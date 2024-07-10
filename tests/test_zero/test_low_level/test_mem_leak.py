import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.zero import LowLevelZeroOptimizer


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(123, 253)

    def forward(self, x):
        x = self.linear1(x)
        return x


DEL_CALLED = False


class TestLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    def __del__(self):
        super().__del__()
        global DEL_CALLED
        DEL_CALLED = True


def exam_mem_leak(world_size):
    """
    In this test, we test whether del will be called after the optimizer
    is out of scope.
    """
    # create models
    zero_model = MlpModel().cuda()

    # we only test stage 1 here
    # in `check_sharded_param_consistency.py`, we will test whether
    # level 1 and 2 will produce exactly the same results
    zero_optimizer = TestLowLevelZeroOptimizer(torch.optim.SGD(zero_model.parameters(), lr=1))

    del zero_optimizer

    assert DEL_CALLED


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    exam_mem_leak(world_size=world_size)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_1_2():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_zero_1_2()
