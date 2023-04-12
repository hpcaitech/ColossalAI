import pytest
import torch
import torch.nn.functional as F

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


def check_cross_entropy():
    input_t = torch.randn(4, 4, device=get_current_device(), requires_grad=True)
    input_ct = torch.randn(4, 4, device=get_current_device(), requires_grad=True)
    with torch.no_grad():
        input_ct.copy_(input_t)

    target = torch.randint(4, (4,), dtype=torch.int64, device=get_current_device())

    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)
    input_t_colo = ColoTensor.from_torch_tensor(tensor=input_ct, spec=ColoTensorSpec(pg))
    input_shard = input_t_colo.redistribute(ShardSpec([-1], [pg.tp_world_size()]))
    input_shard.set_tensor_spec(dist_spec=None, compute_spec=ComputeSpec(ComputePattern.TP1D))

    output = F.cross_entropy(input_t, target)
    output_colo = F.cross_entropy(input_shard, target)
    assert torch.allclose(output_colo, output)

    output.backward()
    output_colo.backward()

    assert torch.allclose(input_t.grad, input_ct.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_cross_entropy()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_loss_func(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_loss_func(1)
