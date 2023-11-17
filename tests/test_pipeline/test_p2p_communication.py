import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device

WORLD_SIZE = 2


def check_p2p_communication():
    pg_mesh = ProcessGroupMesh(WORLD_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, 0)
    p2p = PipelineP2PCommunication(stage_manager)

    rank = dist.get_rank()

    tensor = torch.ones(1, device=get_current_device())
    data = [
        "tensor",
        tensor,
        [tensor],
        {"tensor": tensor},
    ]

    if rank == 0:
        for obj in data:
            p2p.send_forward(obj)
        for obj in data:
            recv_obj = p2p.send_forward_recv_backward(obj)
            assert recv_obj == obj
    elif rank == 1:
        for obj in data:
            recv_obj = p2p.recv_forward()
            assert recv_obj == obj
        for obj in data:
            recv_obj = p2p.send_backward_recv_forward(obj)
            assert recv_obj == obj

    if rank == 1:
        for obj in data:
            p2p.send_backward(obj)
    elif rank == 0:
        for obj in data:
            recv_obj = p2p.recv_backward()
            assert recv_obj == obj


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_p2p_communication()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pipeline_p2p():
    spawn(run_dist, WORLD_SIZE)


if __name__ == "__main__":
    test_pipeline_p2p()
