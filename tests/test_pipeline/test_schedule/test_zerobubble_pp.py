from copy import deepcopy
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import OptimizerWrapper
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.schedule.v_schedule import PipelineGraph, ScheduledNode
from colossalai.pipeline.schedule.zero_bubble_pp import ZeroBubbleVPipeScheduler
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


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


# 1) Test manual v_schedule with multiple microbatch
@parameterize(
    "test_config",
    [
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
    ],
)
def run_fwd_bwd_iter_input(test_config):
    # init dist
    rank = dist.get_rank()
    pp_size = test_config["pp_size"]
    pg_mesh = ProcessGroupMesh(pp_size)
    num_microbatch = test_config["num_microbatches"]
    num_model_chunk = test_config["num_model_chunk"]
    # stage_manager
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=num_model_chunk
    )

    # schedule list
    zbv_schedule = [
        # stage 0
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=3),
        ],
        # stage 1
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=3),
        ],
        # stage 2
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=3),
        ],
        # stage 3
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=3),
        ],
    ]

    scheduler = ZeroBubbleVPipeScheduler(
        schedule=zbv_schedule,  # hint: send whole schedule or local schedule only ?
        stage_manager=stage_manager,
        num_model_chunks=pp_size,
        num_microbatch=num_microbatch,
        overlap_p2p=False,
    )

    # loss func
    def criterion(x, *args, **kwargs):
        return (x * x).mean()

    # init model and input
    batch_size = 4
    num_layers = 8
    in_dim = out_dim = 8
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    data_iter = [torch.rand(batch_size, in_dim, out_dim, requires_grad=True).to(rank)]

    input_base = [t.clone() for t in data_iter]
    model_base = deepcopy(model)

    if rank == 0:
        # layer 0 & 7 to chunk 0 on rank0
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 0 or idx == 7:
                local_chunk.append(sub_model)
    elif rank == 1:
        # layer 1 & 6 to chunk 1 on rank1
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 1 or idx == 6:
                local_chunk.append(sub_model)
    elif rank == 2:
        # layer 2 & 5 to chunk 2 on rank2
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 2 or idx == 5:
                local_chunk.append(sub_model)
    else:
        # layer 3 & 4 to chunk 3 on rank3
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 3 or idx == 4:
                local_chunk.append(sub_model)
    # init optimizer
    optimizer_base = torch.optim.SGD(model_base.parameters(), lr=1e-5)
    optimizer_pp = OptimizerWrapper(torch.optim.SGD(local_chunk.parameters(), lr=1e-5))

    print(
        f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
    )

    torch.cuda.synchronize()
    result = scheduler.forward_backward_step(
        model_chunk=local_chunk,
        data_iter=iter(data_iter),
        criterion=criterion,
        optimizer=optimizer_pp,
        return_loss=True,
        return_outputs=True,
    )

    optimizer_pp.step()

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base[0])
    loss_base = criterion(output_base)
    loss_base.backward()
    optimizer_base.step()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # assert weight
    ##########################
    if rank == 0:
        # layer 0
        assert_close(local_chunk[0].weight, model_base.layers[0].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[0].weight.grad)
        # layer 7
        assert_close(local_chunk[1].weight, model_base.layers[7].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[7].weight.grad)
    if rank == 1:
        # layer 1
        assert_close(local_chunk[0].weight, model_base.layers[1].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[1].weight.grad)
        # layer 6
        assert_close(local_chunk[1].weight, model_base.layers[6].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[6].weight.grad)
    if rank == 2:
        # layer 2
        assert_close(local_chunk[0].weight, model_base.layers[2].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[2].weight.grad)
        # layer 5
        assert_close(local_chunk[1].weight, model_base.layers[5].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[5].weight.grad)
    if rank == 3:
        # layer 3
        assert_close(local_chunk[0].weight, model_base.layers[3].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[3].weight.grad)
        # layer 4
        assert_close(local_chunk[1].weight, model_base.layers[4].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[4].weight.grad)


# 2) add optimizer base 1)
@parameterize(
    "test_config",
    [
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 8,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
    ],
)
def run_fwd_bwd_vschedule_with_optim(test_config):
    # init dist
    rank = dist.get_rank()
    pp_size = test_config["pp_size"]
    pg_mesh = ProcessGroupMesh(pp_size)
    num_microbatch = test_config["num_microbatches"]
    num_model_chunk = test_config["num_model_chunk"]
    # stage_manager
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=num_model_chunk
    )

    h, a, s = 4096, 32, 1024
    mem_f = 34 * h + 5 * a * s
    mem_w = -32 * h
    mem_b = -mem_w - mem_f
    graph = PipelineGraph(
        n_stage=pp_size,
        n_micro=num_microbatch,
        f_cost=1,
        b_cost=1,
        w_cost=1,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
        # max_mem=mem_f * (p * 2 + m_offset),
    )

    zbv_schedule = graph.get_v_schedule()

    scheduler = ZeroBubbleVPipeScheduler(
        schedule=zbv_schedule,  # hint: send whole schedule or local schedule only ?
        stage_manager=stage_manager,
        num_model_chunks=num_model_chunk,
        num_microbatch=num_microbatch,
        overlap_p2p=False,
    )

    # init loss func
    def criterion(x, *args, **kwargs):
        return (x * x).mean()

    # init model and input
    batch_size = test_config["batch_size"]
    num_layers = 8
    assert num_layers % num_model_chunk == 0, f"Model with {num_layers} layer can not dist on {num_model_chunk} chunk"
    in_dim = out_dim = 4096
    before_init_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Before init Model: {before_init_memory :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    data_iter = [torch.rand(batch_size, in_dim, out_dim, requires_grad=True).to(rank)]

    input_base = [t.clone() for t in data_iter]
    model_base = deepcopy(model)

    if rank == 0:
        # layer 0 & 7 to chunk 0 on rank0
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 0 or idx == 7:
                local_chunk.append(sub_model)
    elif rank == 1:
        # layer 1 & 6 to chunk 1 on rank1
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 1 or idx == 6:
                local_chunk.append(sub_model)
    elif rank == 2:
        # layer 2 & 5 to chunk 2 on rank2
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 2 or idx == 5:
                local_chunk.append(sub_model)
    else:
        # layer 3 & 4 to chunk 3 on rank3
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 3 or idx == 4:
                local_chunk.append(sub_model)

    # init optimizer
    optimizer_base = torch.optim.SGD(model_base.parameters(), momentum=0.1, lr=1e-5)
    optimizer_pp = OptimizerWrapper(torch.optim.SGD(local_chunk.parameters(), momentum=0.1, lr=1e-5))

    after_init_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"After init Model & input: {after_init_memory :.5f} GB on device {stage_manager.get_rank()};")

    torch.cuda.synchronize()
    result = scheduler.forward_backward_step(
        model_chunk=local_chunk,
        data_iter=iter(data_iter),
        criterion=criterion,
        optimizer=optimizer_pp,
        return_loss=True,
        return_outputs=True,
    )

    optimizer_pp.step()

    after_pp_step_memory = torch.cuda.memory_allocated() / 1024**3

    # assert memory
    if rank != 0:
        # w.grad hid_dim * hid_dim * 4(fp32) * 2 (2 layer in each stage) / 1024**3
        # output hid_dim * hid_dim * 4(fp32) / 1024**3
        # optim state hid_dim * hid_dim * 4(fp32) * 2 (2 layer in each stage) / 1024**3
        print(f"rank {rank}: {(after_pp_step_memory - after_init_memory)} <= {(in_dim * in_dim * 4 * 5 / 1024**3)}")
        assert (after_pp_step_memory - after_init_memory) <= (in_dim * in_dim * 4 * 5 / 1024**3)
    else:
        # rank0 will also hold output;
        print(
            f"rank {rank}: {round((after_pp_step_memory - after_init_memory), 5)} <= {round((in_dim * in_dim * 4 * 5 / 1024**3 + batch_size * in_dim * in_dim * 4 / 1024**3), 5)}"
        )
        assert round((after_pp_step_memory - after_init_memory), 5) <= round(
            (in_dim * in_dim * 4 * 5 / 1024**3 + batch_size * in_dim * in_dim * 4 / 1024**3), 5
        )

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base[0])
    loss_base = criterion(output_base)
    loss_base.backward()
    optimizer_base.step()

    ##########################
    # assert loss & output
    ##########################
    # only chunk 1 stage 0 hold loss and output
    if rank == 0:
        assert_close(result["loss"], loss_base)
        assert_close(result["outputs"], output_base)

    # print(f"pp result {result}; base result loss:{loss_base} output_base:{output_base} ")
    ##########################
    # assert weight
    ##########################
    if rank == 0:
        # layer 0
        assert_close(local_chunk[0].weight, model_base.layers[0].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[0].weight.grad)
        # layer 7
        assert_close(local_chunk[1].weight, model_base.layers[7].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[7].weight.grad)
    if rank == 1:
        # layer 1
        assert_close(local_chunk[0].weight, model_base.layers[1].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[1].weight.grad)
        # layer 6
        assert_close(local_chunk[1].weight, model_base.layers[6].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[6].weight.grad)
    if rank == 2:
        # layer 2
        assert_close(local_chunk[0].weight, model_base.layers[2].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[2].weight.grad)
        # layer 5
        assert_close(local_chunk[1].weight, model_base.layers[5].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[5].weight.grad)
    if rank == 3:
        # layer 3
        assert_close(local_chunk[0].weight, model_base.layers[3].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[3].weight.grad)
        # layer 4
        assert_close(local_chunk[1].weight, model_base.layers[4].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[4].weight.grad)

    ##########################
    # assert optim state
    ##########################
    optim_base_state = optimizer_base.state_dict()["state"]
    optim_pp_state = optimizer_pp.state_dict()["state"]
    optim_base_param_groups = optimizer_base.state_dict()["param_groups"][0]
    optim_pp_param_groups = optimizer_pp.state_dict()["param_groups"][0]
    # if rank == 0:
    #     print(f"optim_base_state {optim_base_state}")

    # assert param group
    for (key_base, val_base), (key_pp, val_pp) in zip(optim_base_param_groups.items(), optim_pp_param_groups.items()):
        if key_base == key_pp:
            if key_base != "params":
                assert val_base == val_pp
            else:
                # BUG:
                # param_base: [0, 1, 2, 3, 4, 5, 6, 7];
                # params pp: [0, 1];
                assert val_base[:2] == val_pp

    # assert state
    assert_close(optim_pp_state[0]["momentum_buffer"], optim_base_state[2 * rank]["momentum_buffer"])
    assert_close(optim_pp_state[1]["momentum_buffer"], optim_base_state[2 * rank + 1]["momentum_buffer"])


# TODO:4) support Hybrid base 3)
def run_with_hybridplugin(test_config):
    pass


# TODO:5) support MoEHybrid base 3)
@parameterize(
    "test_config",
    [
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
    ],
)
def run_with_moehybridplugin(test_config):
    model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["initial_scale"] = 2**16
    model_list = [
        "transformers_bert",
    ]


# TODO:6) support booster & Hybrid base 4)

# TODO:7) support booster & MoEHybrid base 4)


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # run_fwd_bwd_iter_input()
    run_fwd_bwd_vschedule_with_optim()
    # run_with_moehybridplugin()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp():
    spawn(
        run_dist,
        nprocs=4,
    )


if __name__ == "__main__":
    test_pp()
