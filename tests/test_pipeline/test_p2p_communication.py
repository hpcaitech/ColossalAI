import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.p2p import PipelineP2PCommunication, create_send_metadata
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn

WORLD_SIZE = 2


def check_p2p_communication():
    pg_mesh = ProcessGroupMesh(WORLD_SIZE)
    stage_manager = PipelineStageManager(pg_mesh, 0)
    p2p = PipelineP2PCommunication(stage_manager, overlap_p2p=False)
    rank = dist.get_rank()

    tensor = torch.ones(1, device=get_accelerator().get_current_device())
    data = [
        "tensor",
        tensor,
        [tensor],
        {"tensor": tensor},
    ]

    if rank == 0:
        for obj in data:
            p2p.send_forward(obj)
        for i in range(len(data)):
            recv_obj, _ = p2p.send_forward_recv_backward(data[i], send_first=False)
            assert recv_obj == data[-(i + 1)]
    elif rank == 1:
        for obj in data:
            recv_obj, _ = p2p.recv_forward()
            assert recv_obj == obj
        for i in range(len(data)):
            p2p.send_backward(data[-(i + 1)])
            recv_obj, _ = p2p.recv_forward()
            assert recv_obj == data[i]

    if rank == 1:
        for obj in data:
            p2p.send_backward(obj)
        for i in range(len(data)):
            recv_obj, _ = p2p.send_backward_recv_forward(data[i], send_first=True)
            assert recv_obj == data[-(i + 1)]
    elif rank == 0:
        for obj in data:
            recv_obj, _ = p2p.recv_backward()
            assert recv_obj == obj
        for i in range(len(data)):
            recv_obj, _ = p2p.send_forward_recv_backward(data[-(i + 1)], send_first=False)
            assert recv_obj == data[i]

    if rank == 0:
        recv_obj, _ = p2p.send_forward_recv_backward(
            tensor,
            send_metadata=False,
            metadata_recv=create_send_metadata(tensor),
        )
        assert recv_obj == tensor
    elif rank == 1:
        recv_obj, _ = p2p.recv_forward(metadata_recv=create_send_metadata(tensor))
        assert recv_obj == tensor
        p2p.send_backward(tensor, send_metadata=False)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_p2p_communication()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pipeline_p2p():
    spawn(run_dist, WORLD_SIZE)


if __name__ == "__main__":
    test_pipeline_p2p()
