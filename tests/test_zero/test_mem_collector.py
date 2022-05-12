import torch
import colossalai
import pytest
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity, colo_set_process_memory_fraction
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.shard_utils import BucketTensorShardStrategy
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from functools import partial


class MyTestModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(512, 512)
        self.weight = nn.Parameter(torch.randn(1024, 512))
        self.proj2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.proj1(x)
        x = F.linear(x, self.weight)
        x = self.proj2(x)

        return x


def run_mem_collector_testing():
    cuda_capacity = colo_device_memory_capacity(get_current_device())
    fraction = (50 * 1024**2) / cuda_capacity
    # limit max memory to 50MB
    colo_set_process_memory_fraction(fraction)
    shard_strategy = BucketTensorShardStrategy()
    with ZeroInitContext(target_device=get_current_device(), shard_strategy=shard_strategy, shard_param=True):
        model = MyTestModel()

    model = ShardedModelV2(module=model,
                           shard_strategy=shard_strategy,
                           reduce_scatter_bucket_size_mb=1,
                           tensor_placement_policy='auto')

    data = torch.randn(2, 512, device=get_current_device())

    output = model(data)
    loss = torch.mean(output)
    model.backward(loss)

    cuda_model_data_list = model._memstats_collector.model_data_list('cuda')
    assert cuda_model_data_list == [1311744, 1836032, 1836032, 1311744, 1836032, 1836032]

    cuda_non_model_data_list = model._memstats_collector.non_model_data_list('cuda')
    assert cuda_non_model_data_list[0] > cuda_non_model_data_list[1]
    assert cuda_non_model_data_list[-2] > cuda_non_model_data_list[-1]


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_mem_collector_testing()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_mem_collector(world_size=2):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_mem_collector()
