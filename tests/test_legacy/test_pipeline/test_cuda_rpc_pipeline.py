import torch
from rpc_test_utils import RpcTestModel, parse_args, rpc_run

from colossalai.legacy.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine

# global variable for model created
feat_num = 100
h = 100


def partition(pp_rank: int, chunk: int, stage_num: int):
    torch.manual_seed(1024)
    partition = RpcTestModel(pp_rank, stage_num, feat_num, h)
    return partition


def run_master(args):
    torch.manual_seed(100)

    epoch = args.epoch
    device = args.device
    stage_num = args.world_size
    chunk = args.chunk
    num_microbatches = args.num_microbatches
    use_checkpoint = args.use_checkpoint

    sample_num = 1024
    batch_size = 1024

    assert sample_num % batch_size == 0

    input_sample = torch.randn((sample_num, feat_num), device=device)

    engine = OneFOneBPipelineEngine(
        partition_fn=partition,
        stage_num=stage_num,
        num_microbatches=num_microbatches,
        device=device,
        chunk=chunk,
        checkpoint=use_checkpoint,
    )

    for _ in range(epoch):
        _ = engine.forward_backward(input_sample, forward_only=False)


if __name__ == "__main__":
    args = parse_args()
    rpc_run(args, run_master)
