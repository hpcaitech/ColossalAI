#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torch.distributed as dist
from torchvision.models import resnet50

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import TensorShardStrategy


def run_dist(rank, world_size, port):
    # this test only runs on resnet18
    # as this model has sync batch normalization
    # need to configure cudnn deterministic so that
    # randomness of convolution layers will be disabled
    zero_config = dict(model_config=dict(shard_strategy=TensorShardStrategy()))
    colossalai.launch(config=dict(zero=zero_config, cudnn_determinstic=True, cudnn_benchmark=False),
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')

    with ZeroInitContext(target_device=torch.cuda.current_device(),
                         shard_strategy=gpc.config.zero.model_config.shard_strategy,
                         shard_param=True):
        model = resnet50()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    engine, *args = colossalai.initialize(model, optimizer, criterion)

    # train for dummy iterations
    engine.train()
    for _ in range(2):
        data = torch.rand(4, 3, 128, 128).cuda().half()
        label = torch.randint(0, 10, size=(4,)).cuda()
        engine.zero_grad()
        out = engine(data)
        loss = engine.criterion(out, label)
        engine.backward(loss)
        engine.step()

    # test
    # need to make sure the batch norm stats are synchronized
    # so that given the same input, the model will produce the same
    # output on different ranks
    engine.eval()
    data = torch.rand(4, 3, 128, 128).cuda().half()
    dist.broadcast(data, src=0, group=gpc.get_group(ParallelMode.DATA))

    # predict
    out = engine(data)

    # test if results are equal
    tensor_list = [torch.empty_like(out) for _ in range(world_size - 1)]
    tensor_list.insert(rank, out)
    dist.all_gather(tensor_list=tensor_list, tensor=out, group=gpc.get_group(ParallelMode.DATA))

    assert torch.all(tensor_list[0] == tensor_list[1]), \
        'expected the output from different ranks to be the same, but got different values'


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_sharded_optim_with_sync_bn():
    """
    This test is to make sure that buffers are synchronized between ranks
    when using ZeRO. An example of module buffer is the running stats of
    BatchNormalization layer, i.e. mean and var.

    If the buffers are not synchronized, the model will produce different
    output even though the input and parameters are the same. This is not
    wanted if we are doing predictions.

    """
    spawn(run_dist, 2)


if __name__ == '__main__':
    test_sharded_optim_with_sync_bn()
