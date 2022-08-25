import os
import argparse

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch import autograd
from torch.optim import SGD, Adam, RMSprop, Optimizer
from colorama import Back, Style

from colossalai.pipeline.rpc.PipelineBase import FillDrainPipelineEngine, OneFOneBPipelineEngine
from colossalai.testing import assert_close
from rpc_test_utils import rpc_run, parse_args, RpcTestModel


def run_master(args):
    torch.manual_seed(100)

    device = args.device
    stage_num = args.world_size
    chunk = args.chunk
    actual_stage_num = stage_num * chunk
    use_interleave = args.use_interleave
    use_checkpoint = args.use_checkpoint
    num_microbatches = args.num_microbatches
    optimizer_class = globals()[args.optimizer]

    lr = 1e-3

    sample_num = 1024
    feat_num = 100
    h = 100
    batch_size = 1024

    assert sample_num % batch_size == 0
    batch_num = sample_num // batch_size

    input_sample = torch.randn((sample_num, feat_num), device=device)

    module_partitions = [RpcTestModel(pp_rank, actual_stage_num, feat_num, h) for pp_rank in range(actual_stage_num)]

    engine = OneFOneBPipelineEngine(module_partitions=module_partitions,
                                    stage_num=stage_num,
                                    num_microbatches=num_microbatches,
                                    device=device,
                                    chunk=chunk,
                                    use_interleave=use_interleave,
                                    checkpoint=use_checkpoint)

    engine.initialize_optimizer(optimizer_class, lr=lr)

    _ = engine.forward_backward(input_sample)
    engine.step()

    cuda_rpc_result = []
    single_result = []
    actual_stage_num = engine._get_actual_stage_num()

    # compute parameters after updating in cuda rpc
    parameters = engine.remote_parameters()
    for stage_id in range(actual_stage_num):
        for p in parameters[stage_id]:
            cuda_rpc_result.append(p)

    # compute forward result and backward grad of parameters just in rank_0
    test_model = nn.Sequential(*module_partitions).to(device)
    optimizer: Optimizer = optimizer_class(test_model.parameters(), lr=lr)
    input_sample = input_sample.requires_grad_()
    out_val = test_model(input_sample).sum()
    autograd.backward(out_val)
    optimizer.step()
    optimizer.zero_grad()

    for p in test_model.parameters():
        single_result.append(p)

    assert len(cuda_rpc_result) == len(single_result)
    for r_c, r_s in zip(cuda_rpc_result, single_result):
        assert_close(r_c, r_s, 0.001, 0.001)


if __name__ == "__main__":
    args = parse_args()
    rpc_run(args, run_master)
