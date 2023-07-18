import pytest
import torch
from torch.nn import functional as F

import colossalai
from colossalai.tensor import ColoTensor, ColoTensorSpec, ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.test_tensor.common_utils import split_param_col_tp1d, split_param_row_tp1d, tensor_equal, tensor_shard_equal


def run_with_spec(spec_init_func, pg: ProcessGroup):
    model = torch.nn.Embedding(12, 32).cuda()
    weight = ColoTensor(torch.nn.Parameter(model.weight.detach()), ColoTensorSpec(pg))

    spec_init_func(weight, pg)

    x = torch.tensor((0, 3, 6, 9)).cuda()
    out = model(x)
    colo_out = F.embedding(x, weight)
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    # compare grad inside a TP group
    assert tensor_shard_equal(model.weight.grad, weight.grad, pg.tp_local_rank(), pg.tp_world_size())


def run_dist(rank, world_size, port):
    # config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=world_size)
    run_with_spec(split_param_row_tp1d, pg)
    run_with_spec(split_param_col_tp1d, pg)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_embedding_1d(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_embedding_1d(4)
