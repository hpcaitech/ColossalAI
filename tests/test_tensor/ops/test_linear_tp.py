from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import ColoTensorSpec, ProcessGroup, ColoTensor
from tests.test_tensor.common_utils import tensor_equal, tensor_shard_equal, split_param_col_tp1d, split_param_row_tp1d


def run_with_spec(spec_init_func, split_bias):
    pg = ProcessGroup(tp_degree=torch.distributed.get_world_size())
    model = torch.nn.Linear(4, 8).cuda()
    weight = ColoTensor(torch.nn.Parameter(model.weight.detach()), ColoTensorSpec(pg))
    bias = ColoTensor(torch.nn.Parameter(model.bias.detach()), ColoTensorSpec(pg))

    spec_init_func(weight, pg)
    if split_bias:
        spec_init_func(bias, pg)

    x = torch.rand(2, 4).cuda()
    out = model(x)
    colo_out = F.linear(x, weight, bias)
    colo_out = colo_out.to_replicate()
    assert tensor_equal(out, colo_out)
    grad = torch.rand_like(out)
    out.backward(grad)
    colo_out.backward(grad)
    assert tensor_shard_equal(model.weight.grad, weight.grad, pg.tp_local_rank(), pg.tp_world_size())
    assert tensor_shard_equal(model.bias.grad, bias.grad, pg.tp_local_rank(), pg.tp_world_size())


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_with_spec(spec_init_func=split_param_col_tp1d, split_bias=False)
    run_with_spec(spec_init_func=split_param_row_tp1d, split_bias=True)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_linear_1d(4)
