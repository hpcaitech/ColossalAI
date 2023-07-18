import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.checkpoint.utils import gather_tensor, scatter_tensor
from tests.test_tensor.common_utils import tensor_shard_equal


def run_dist(rank, world_size, port, dp_degree, tp_degree):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(dp_degree=dp_degree, tp_degree=tp_degree)
    x = torch.randn(4, 4)
    param = ColoTensor(torch.nn.Parameter(x), spec=ColoTensorSpec(pg))
    spec = ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D)
    param.set_tensor_spec(*spec)

    gather_tensor(param)
    if dist.get_rank() == 0:
        assert torch.all(x == param)
    else:
        assert tensor_shard_equal(x, param.data, pg.tp_local_rank(), pg.tp_world_size())
    dist.barrier()

    scatter_tensor(param, spec[0])
    assert tensor_shard_equal(x, param.data, pg.tp_local_rank(), pg.tp_world_size())
    assert param.requires_grad is True
    dist.barrier()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [4])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size):
    spawn(run_dist, world_size, dp_degree=2, tp_degree=world_size // 2)


if __name__ == '__main__':
    test_checkpoint(world_size=4)
