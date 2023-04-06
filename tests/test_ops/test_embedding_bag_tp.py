import pytest
import torch
from torch.nn import functional as F

import colossalai
from colossalai.tensor import ColoParameter, ColoTensorSpec, ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.test_tensor.common_utils import split_param_col_tp1d, tensor_equal, tensor_shard_equal


def run_with_spec(spec_init_func):
    pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())
    model = torch.nn.EmbeddingBag(10, 4).cuda()
    weight = ColoParameter(model.weight.clone(), True, ColoTensorSpec(pg))

    spec_init_func(weight, pg)

    inputs = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]).cuda()
    offsets = torch.tensor([0, 4]).cuda()
    out = model(inputs, offsets=offsets)
    colo_out = F.embedding_bag(inputs, weight, offsets=offsets)
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, weight.grad, pg.tp_local_rank(), pg.tp_world_size())


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(split_param_col_tp1d)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_embedding_bag_1d(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_embedding_bag_1d(4)
