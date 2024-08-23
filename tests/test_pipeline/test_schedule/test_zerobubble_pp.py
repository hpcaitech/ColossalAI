from copy import deepcopy
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.schedule.zero_bubble_pp import ZeroBubbleVPipeScheduler
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn


class MlpModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=None) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def test_zerobubble_pipeline_base(
    rank: int,
    world_size: int,
    port: int,
):
    # init dist
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    pg_mesh = ProcessGroupMesh(world_size)

    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=world_size)

    scheduler = ZeroBubbleVPipeScheduler(
        schedule=[],
        stage_manager=stage_manager,
        num_model_chunks=world_size,
        num_microbatch=1,
        overlap_p2p=False,
    )

    rank = dist.get_rank()

    # init model and input
    num_layers = 8
    in_dim = out_dim = 8
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    input0 = torch.rand(in_dim, out_dim, requires_grad=True).to(rank)

    input_base = input0.clone()
    model_base = deepcopy(model)

    if rank == 0:
        # layer 0 & 7 to chunk 0 on rank0
        chunk_0 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 0 or idx == 7:
                chunk_0.append(sub_model)
    elif rank == 1:
        # layer 1 & 6 to chunk 1 on rank1
        chunk_1 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 1 or idx == 6:
                chunk_1.append(sub_model)
    elif rank == 2:
        # layer 2 & 5 to chunk 2 on rank2
        chunk_2 = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 2 or idx == 5:
                chunk_2.append(sub_model)
    else:
        # layer 3 & 4 to chunk 3 on rank3
        chunk_3 = torch.nn.Sequential().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 3 or idx == 4:
                chunk_3.append(sub_model)
    print(
        f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
    )

    def criterion(x, *args, **kwargs):
        return (x * x).mean()

    ##########################
    # Step1: fwd
    ##########################
    ######
    # fwd 1->4
    ######
    # chunk 0 id 0 (layer 0) fwd
    if rank == 0:
        chunk_id = 0
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            input_obj=input0,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 0 id 0 (layer 0)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 1 id 0 (layer 1)  fwd
    if rank == 1:
        chunk_id = 0
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 1 id 0 (layer 1)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 2 id 0 (layer 2)  fwd
    if rank == 2:
        chunk_id = 0
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 2 id 0 (layer 2)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    # chunk 3 id 0 (layer 3)  fwd
    if rank == 3:
        chunk_id = 0
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 3 id 0 (layer 3)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

        ######
        # fwd 4->1
        ######

    if rank == 3:
        chunk_id = 1
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 3 id 1 (layer 4)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 2:
        chunk_id = 1
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 2 id 1 (layer 5)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 1:
        chunk_id = 1
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        print(
            f"chunk 1 id 1 (layer 6)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    if rank == 0:
        chunk_id = 1
        scheduler.schedule_f(
            scheduled_node=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            input_obj=None,
            criterion=criterion,
            accum_loss=None,
            outputs=None,
        )
        # print(f"fwd output {output7}")
        print(
            f"chunk 0 id 1 (layer 7)fwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
        )

    ##########################
    # Step2: bwd
    ##########################
    ######
    # bwd rank 4->1
    ######
    # chunk 0 id 1 (layer 7) bwd
    if rank == 0:
        chunk_id = 1
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )
        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # # chunk 1 id 1 (layer 6) bwd
    if rank == 1:
        chunk_id = 1
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )
        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # chunk 2 id 1 (layer 5) bwd
    if rank == 2:
        chunk_id = 1
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # chunk 3 id 1 (layer 4) bwd
    if rank == 3:
        chunk_id = 1
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    #     ######
    #     # bwd rank 1->4
    #     ######

    # chunk 3 id 0 (layer 3) bwd
    if rank == 3:
        chunk_id = 0
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )
        # print(f"input_grad3 {input_grad3}")
        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_3,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # chunk 2 id 0 (layer 2) bwd
    if rank == 2:
        chunk_id = 0
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )
        # print(f"input_grad2 {input_grad2}")
        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_2,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # chunk 1 id 0 (layer 1) bwd
    if rank == 1:
        chunk_id = 0
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_1,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    # chunk 0 id 0 (layer 0) bwd
    if rank == 0:
        chunk_id = 0
        scheduler.schedule_b(
            scheduled_node=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )
        # print(f"input_grad0 {input_grad0}")

        scheduler.schedule_w(
            scheduled_node=None,
            non_w_pending=None,
            model_chunk=chunk_0,
            model_chunk_id=chunk_id,
            # optimizer: OptimizerWrapper,
        )

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base)
    # loss_base = output_base.mean()
    loss_base = criterion(output_base)
    loss_base.backward()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    # assert weight
    if rank == 0:
        # layer 0
        assert_close(chunk_0[0].weight, model_base.layers[0].weight)
        assert_close(chunk_0[0].weight.grad, model_base.layers[0].weight.grad)
        # layer 7
        assert_close(chunk_0[1].weight, model_base.layers[7].weight)
        assert_close(chunk_0[1].weight.grad, model_base.layers[7].weight.grad)
    if rank == 1:
        # layer 1
        assert_close(chunk_1[0].weight, model_base.layers[1].weight)
        assert_close(chunk_1[0].weight.grad, model_base.layers[1].weight.grad)
        # layer 6
        assert_close(chunk_1[1].weight, model_base.layers[6].weight)
        assert_close(chunk_1[1].weight.grad, model_base.layers[6].weight.grad)

    if rank == 2:
        # layer 2
        assert_close(chunk_2[0].weight, model_base.layers[2].weight)
        assert_close(chunk_2[0].weight.grad, model_base.layers[2].weight.grad)
        # layer 5
        assert_close(chunk_2[1].weight, model_base.layers[5].weight)
        assert_close(chunk_2[1].weight.grad, model_base.layers[5].weight.grad)

    if rank == 3:
        # layer 3
        assert_close(chunk_3[0].weight, model_base.layers[3].weight)
        assert_close(chunk_3[0].weight.grad, model_base.layers[3].weight.grad)
        # layer 4
        assert_close(chunk_3[1].weight, model_base.layers[4].weight)
        assert_close(chunk_3[1].weight.grad, model_base.layers[4].weight.grad)


# @pytest.mark.dist
# @pytest.mark.parametrize("num_microbatch", [4])
# @pytest.mark.parametrize("batch_size", [4])
# @pytest.mark.parametrize("num_model_chunk", [2])
@rerun_if_address_is_in_use()
def test_pp():
    spawn(
        test_zerobubble_pipeline_base,
        nprocs=4,
    )


if __name__ == "__main__":

    test_pp()
