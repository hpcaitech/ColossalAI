import os
from typing import Callable, List, Optional, Type, Union
import time

import pytest
import torch
import torch.nn as nn
from titans.dataloader.cifar10 import build_cifar
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from tqdm import tqdm

from rpc_test_utils import rpc_run, parse_args
import colossalai
import colossalai.nn as col_nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.context import ParallelMode
from colossalai.pipeline.pipelinable import PipelinableContext, PipelinableModel
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine


def flatten(x):
    return torch.flatten(x, 1)


class Flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def run_master(args):
    batch_size = args.batch_size
    chunk = args.chunk
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches

    assert chunk == 1

    pipelinable = PipelinableContext()

    # build model partitions
    with pipelinable:
        # input : [B, 3, 32, 32]
        model = resnet50()

    exec_seq = [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', (flatten, "behind"), 'fc'
    ]
    pipelinable.to_layer_list(exec_seq)
    module_partitions: List[PipelinableModel] = [
        pipelinable.partition(chunk, stage_num, pp_rank) for pp_rank in range(world_size)
    ]

    # build dataloader
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(batch_size, root, padding=4, crop=32, resize=32)
    criterion = nn.CrossEntropyLoss()

    partition_1 = module_partitions[0]
    partition_2 = []
    for model in module_partitions[1]._module_list:
        partition_2.append(model)
    partition_2.insert(len(partition_2) - 1, Flatten())
    partition_2 = nn.Sequential(*partition_2)
    module_partitions = [partition_1, partition_2]

    pp_engine = OneFOneBPipelineEngine(module_partitions=module_partitions,
                                       stage_num=stage_num,
                                       num_microbatches=num_microbatches,
                                       device=device,
                                       chunk=chunk,
                                       criterion=criterion,
                                       checkpoint=False)

    pp_engine.initialize_optimizer(torch.optim.Adam, lr=1e-3)
    s = time.time()

    for bx, by in tqdm(train_dataloader):
        pp_engine.forward_backward(bx, labels=by, forward_only=False)

    cost_time = time.time() - s

    print("total cost time :", cost_time)
    print("cost time per batch:", cost_time / len(train_dataloader))


@pytest.mark.skip("Test for performance, no need for CI")
def main():
    args = parse_args()
    # this is due to limitation of partition function
    args.world_size = 2
    args.chunk = 1
    rpc_run(args, run_master)


if __name__ == '__main__':
    main()
