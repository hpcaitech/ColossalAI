import torch
from torch import nn

from colossalai.pipeline.rpc._pipeline_schedule import FillDrainPipelineEngine, OneFOneBPipelineEngine
from rpc_test_utils import rpc_run, parse_args, RpcTestModel


def run_master(args):
    torch.manual_seed(100)

    epoch = args.epoch
    device = args.device
    stage_num = args.world_size
    chunk = args.chunk
    num_microbatches = args.num_microbatches
    actual_stage_num = stage_num * chunk
    use_checkpoint = args.use_checkpoint

    sample_num = 1024
    feat_num = 10
    h = 10
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
                                    checkpoint=use_checkpoint)

    for _ in range(epoch):
        _ = engine.forward_backward(input_sample, forward_only=False)


if __name__ == "__main__":
    args = parse_args()
    rpc_run(args, run_master)
