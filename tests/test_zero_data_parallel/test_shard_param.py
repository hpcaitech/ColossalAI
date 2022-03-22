from copy import deepcopy
from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize
from colossalai.utils import free_port
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)
from colossalai.zero.sharded_param import ShardedTensor
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from tests.test_zero_data_parallel.common import CONFIG, allclose


@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_shard_tensor_with_strategy(shard_strategy_class, world_size):
    t = ShardedTensor(tensor=torch.randn(world_size * 2, 3))
    assert list(t.origin_shape) == [world_size * 2, 3]
    assert list(t.shape) == [world_size * 2, 3]

    shard_strategy = shard_strategy_class()

    # test shard strategy
    shard_strategy.shard([t])
    assert list(t.shape) == [6], f"{list(t.shape)} vs 6"
    shard_strategy.gather([t])
    assert list(t.shape) == [world_size * 2, 3], f"{list(t.shape)} vs {[world_size * 2, 3]}"


def _run_shard_tensor(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_shard_tensor_with_strategy(world_size=world_size)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_shard_tensor(world_size):
    run_func = partial(_run_shard_tensor, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def _run_shard_param_v2(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    param = torch.nn.Parameter(torch.randn(2, 3))
    param_ref = deepcopy(param)
    sparam = ShardedParamV2(param=param, process_group=None)

    allclose(sparam.sharded_data_tensor.payload, param_ref.data)

    sparam.remove_torch_payload()
    assert (param.data.numel() == 1)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
def test_shard_param_v2(world_size):
    run_func = partial(_run_shard_param_v2, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_shard_tensor(2)
    test_shard_param_v2(2)
