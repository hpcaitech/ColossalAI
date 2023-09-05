import os
import time

import pytest
import torch
import torch.nn as nn
from rpc_test_utils import parse_args, rpc_run
from titans.dataloader.cifar10 import build_cifar
from torchvision.models import resnet50
from tqdm import tqdm

from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.pipeline.rpc import OneFOneBPipelineEngine


def flatten(x):
    return torch.flatten(x, 1)


def partition(pp_rank: int, chunk: int, stage_num: int):
    pipelinable = PipelinableContext()

    # build model partitions
    with pipelinable:
        # input : [B, 3, 32, 32]
        _ = resnet50()

    pipelinable.policy = "customized"

    exec_seq = [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', (flatten, "behind"), 'fc'
    ]
    pipelinable.to_layer_list(exec_seq)
    partition = pipelinable.partition(chunk, stage_num, pp_rank)
    return partition


def run_master(args):
    batch_size = args.batch_size
    chunk = args.chunk
    device = args.device
    world_size = args.world_size
    stage_num = world_size
    num_microbatches = args.num_microbatches

    # build dataloader
    root = os.environ.get('DATA', './data')
    train_dataloader, test_dataloader = build_cifar(batch_size, root, padding=4, crop=32, resize=32)
    criterion = nn.CrossEntropyLoss()

    pp_engine = OneFOneBPipelineEngine(partition_fn=partition,
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
