import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ProcessGroup, ShardSpec
from colossalai.tensor.distspec import DistPlacementPattern
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from tests.test_tensor.common_utils import debug_print, split_param_col_tp1d, split_param_row_tp1d


def exam_view_core(pg):
    # the case of replicated ColoTensors
    x = torch.randn(4, 4).cuda()
    x_colo = ColoTensor(x, ColoTensorSpec(pg))

    y = x.view(2, -1, 2)
    y_colo = x_colo.view(2, -1, 2)

    assert torch.all(y == y_colo)
    assert y_colo.dist_spec.placement == DistPlacementPattern.REPLICATE
    # the perfect case of col-sliced ColoTensors
    split_param_col_tp1d(x_colo, pg)

    z = x.view(torch.Size((2, 1, 2, -1)))
    z_colo = x_colo.view(torch.Size((2, 1, 2, -1)))
    if dist.get_rank() == 0:
        z = z[:, :, :, 0:2]
    else:
        z = z[:, :, :, 2:]
    assert torch.all(z == z_colo)
    assert z_colo.dist_spec == x_colo.dist_spec
    # the perfect case of row-sliced ColoTensors
    split_param_row_tp1d(x_colo, pg)

    z = x.view(torch.Size((-1, 2, 2)))
    z_colo = x_colo.view(torch.Size((-1, 2, 2)))
    if dist.get_rank() == 0:
        z = z[0:2, :, :]
    else:
        z = z[2:, :, :]
    assert torch.all(z == z_colo)
    assert z_colo.dist_spec == x_colo.dist_spec
    # the normal case of row-sliced ColoTensors
    z = x.view(-1, 2, 2, 2)
    z_colo = x_colo.view(-1, 2, 2, 2)
    assert torch.all(z == z_colo)
    assert y_colo.dist_spec.placement == DistPlacementPattern.REPLICATE


def exam_view_autograd(pg):
    x = torch.randn(8, 2, device=get_current_device(), requires_grad=True)
    y = torch.randn(8, 2, device=get_current_device(), requires_grad=True)
    with torch.no_grad():
        y.copy_(x)
    y = ColoTensor(y, ColoTensorSpec(pg))
    y_slice = y.redistribute(ShardSpec([-1], [pg.tp_world_size()]))

    xx = x.view(2, 2, -1)
    yy_slice = y_slice.view(2, 2, -1)
    yy = yy_slice.to_replicate()
    grad = torch.randn(2, 2, 4, device=get_current_device())

    xx.backward(grad)
    yy.backward(grad)
    assert torch.all(x.grad == y.grad)


def exam_view_errors(pg):
    x = torch.randn(8, 2, device=get_current_device())
    x = ColoTensor(x, ColoTensorSpec(pg))
    split_param_row_tp1d(x, pg)

    x.view('a', 'b', 'c')
    x.view(8, -1)
    x.view([-2, -2, -2])
    x.view((-1, -1, -1))


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())
    exam_view_core(pg)
    exam_view_autograd(pg)
    # exam_view_errors(pg)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@rerun_if_address_is_in_use()
def test_view(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_view(2)
