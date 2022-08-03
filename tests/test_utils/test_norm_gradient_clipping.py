
from cmath import inf
import copy
from colossalai.tensor import distspec, process_group, ColoTensorSpec
from colossalai.tensor.colo_parameter import ColoParameter

import colossalai
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.logging import disable_existing_loggers
from colossalai.utils import clip_grad_norm_fp32, free_port
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from colossalai.zero.shard_utils.tensor_shard_strategy import TensorShardStrategy
from functools import partial
from colossalai.testing import parameterize, rerun_if_address_is_in_use


def allclose(tensor_a: torch.Tensor, tensor_b: torch.Tensor, loose=False) -> bool:
    if loose:
        return torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    return torch.allclose(tensor_a, tensor_b)


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    a = torch.tensor([2.,3.], dtype=torch.float,requires_grad=True, device="cuda")
    b = torch.tensor([6.,4.], dtype=torch.float,requires_grad=True, device="cuda")
    shard_spec = distspec.shard(dims=[0], num_partitions=[world_size])
    pg = process_group.ProcessGroup(rank=rank, ranks=[i for i in range(world_size)], tp_degree=world_size)
    tensor_spec = ColoTensorSpec(pg=pg,dist_attr=shard_spec)
    colo_a = ColoParameter(data=a, spec=tensor_spec)
    colo_b = ColoParameter(data=b, spec=tensor_spec)

    colo_loss = 3*colo_a**3 - colo_b**2
    colo_loss.sum().backward()
    loss = 3*a**3 - b**2
    loss.sum().backward()
    clip_grad_norm_([a,b],1.0)
    #print(colo_a.grad, colo_b.grad)
    total_norm = clip_grad_norm_fp32([colo_a,colo_b],1.0,3.0)
    #assert allclose(a.grad, colo_a.grad)
    print(colo_a.grad, colo_b.grad)
    print(total_norm)
    print(colo_a.get_tp_world_size(), colo_a.get_process_group().get_ranks_in_dp())

@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_zero_clip_grad():
    world_size = 4
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_clip_grad()
