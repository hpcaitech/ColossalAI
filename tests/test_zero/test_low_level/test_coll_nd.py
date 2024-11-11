import numpy as np
import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.utils import get_current_device
from colossalai.zero.low_level._utils import all_gather_into_flat_tensor_nd


def check_all_gather_2d():
    seed_all(1024)
    tensor = torch.rand(128, device=get_current_device())
    extra_dp_size, inner_dp_size = 2, 2
    pg_mesh = ProcessGroupMesh(extra_dp_size, inner_dp_size)
    extra_dp_group = pg_mesh.get_group_along_axis(0)
    inner_dp_group = pg_mesh.get_group_along_axis(1)
    ranks = [dist.get_rank(extra_dp_group), dist.get_rank(inner_dp_group)]
    sizes = [dist.get_world_size(extra_dp_group), dist.get_world_size(inner_dp_group)]
    chunk = tensor.chunk(dist.get_world_size())[np.ravel_multi_index(ranks, sizes)].clone()
    out = torch.zeros_like(tensor)
    all_gather_into_flat_tensor_nd(out, chunk, group=(extra_dp_group, inner_dp_group))
    assert torch.equal(out, tensor)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")

    check_all_gather_2d()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_comm_nd():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_comm_nd()
