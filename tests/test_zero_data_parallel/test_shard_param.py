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
from colossalai.testing import rerun_on_exception
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
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_shard_tensor(world_size):
    run_func = partial(_run_shard_tensor, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


def _run_shard_param_v2(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    param = torch.nn.Parameter(torch.randn(2, 3))
    param_ref = deepcopy(param)
    sparam = ShardedParamV2(param=param, process_group=None)

    allclose(sparam.sharded_data_tensor.payload, param_ref.data)

    # Test get memory usage
    sparam.fp32_grad = torch.randn(2, 3)
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    assert cpu_mem_use == 2 * 3 * 4 * 2, f"cpu_mem_use: {cpu_mem_use}"

    sparam.remove_torch_payload()
    assert (param.data.numel() == 1)
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    # 4 is size of dummy tensor of param.data
    assert cpu_mem_use == 2 * 3 * 4 * 2 + 4

    sparam.fp16_grad = torch.randn(2, 3).cuda().half()
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    assert cpu_mem_use == 2 * 3 * 4 * 2 + 4
    assert cuda_mem_use == 2 * 3 * 2

    sparam.fp16_grad = None
    sparam.fp32_grad = torch.randn(2, 3)
    sparam.remove_torch_payload()
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    assert cpu_mem_use == 2 * 3 * 4 * 2 + 4
    assert cuda_mem_use == 0

    # append a grad to torch param
    param.data = sparam.sharded_data_tensor.payload
    param.grad = torch.randn(2, 3)
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    assert cpu_mem_use == 2 * 3 * 4 * 2 + 2 * 3 * 4, f"cpu_mem_use {cpu_mem_use}"
    assert cuda_mem_use == 0

    # reuse torch grad for sparam
    sparam.fp32_grad = param.grad
    cuda_mem_use, cpu_mem_use = sparam.get_memory_usage()
    assert cpu_mem_use == 2 * 3 * 4 * 2
    assert cuda_mem_use == 0


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_shard_param_v2(world_size):
    run_func = partial(_run_shard_param_v2, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    # test_shard_tensor(2)
    test_shard_param_v2(2)
