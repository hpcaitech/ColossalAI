import torch
from rpc_test_utils import RpcTestModel, parse_args, rpc_run
from torch import autograd, nn

from colossalai.legacy.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.testing import assert_close

feat_num = 100
h = 100


def partition(pp_rank: int, chunk: int, stage_num: int):
    torch.manual_seed(1024)
    partition = RpcTestModel(pp_rank, stage_num, feat_num, h)
    return partition


def run_master(args):
    torch.manual_seed(100)

    device = args.device
    stage_num = args.world_size
    chunk = args.chunk
    actual_stage_num = stage_num * chunk
    use_checkpoint = args.use_checkpoint
    num_microbatches = args.num_microbatches

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
    test_model = nn.Sequential(
        *[partition(pp_rank, chunk, actual_stage_num) for pp_rank in range(actual_stage_num)]
    ).to(device)
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
