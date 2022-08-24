import os
import argparse

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch import autograd
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

    forward_result = engine.forward_backward(input_sample)

    cuda_rpc_result = []
    single_result = []
    actual_stage_num = engine._get_actual_stage_num()

    # compute forward result and backward grad of parameters in cuda rpc
    cuda_rpc_result.append(sum(forward_result[0]))
    grad = engine.remote_grad()
    for stage_id in range(actual_stage_num):
        for p in grad[stage_id]:
            cuda_rpc_result.append(p)

    # compute forward result and backward grad of parameters just in rank_0
    test_model = nn.Sequential(*module_partitions).to(device)
    input_sample = input_sample.requires_grad_()
    out_val = test_model(input_sample).sum()
    autograd.backward(out_val)
    single_result.append(out_val)
    for p in test_model.parameters():
        single_result.append(p.grad)

    assert len(cuda_rpc_result) == len(single_result)
    for r_c, r_s in zip(cuda_rpc_result, single_result):
        assert_close(r_c, r_s, 0.001, 0.001)


if __name__ == "__main__":
    args = parse_args()
    rpc_run(args, run_master)
