import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


def check_p2p_communication():
    pg_mesh = ProcessGroupMesh(2)
    stage_manager = PipelineStageManager(pg_mesh, 0)
    p2p = PipelineP2PCommunication(stage_manager)

    rank = dist.get_rank()

    tensor = torch.ones(1, device=get_current_device())

    if rank == 0:
        p2p.send_forward(tensor)
        p2p.send_forward([tensor])
        p2p.send_forward({"tensor": tensor})
    else:
        obj = p2p.recv_forward()
        assert torch.equal(obj, tensor)
        obj = p2p.recv_forward()
        assert type(obj) == list and len(obj) == 1 and torch.equal(obj[0], tensor)
        obj = p2p.recv_forward()
        assert type(obj) == dict and "tensor" in obj and torch.equal(obj["tensor"], tensor)

    if rank == 1:
        p2p.send_backward(tensor)
        p2p.send_backward([tensor])
        p2p.send_backward({"tensor": tensor})
    else:
        obj = p2p.recv_backward()
        assert torch.equal(obj, tensor)
        obj = p2p.recv_backward()
        assert type(obj) == list and len(obj) == 1 and torch.equal(obj[0], tensor)
        obj = p2p.recv_backward()
        assert type(obj) == dict and "tensor" in obj and torch.equal(obj["tensor"], tensor)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_p2p_communication()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pipeline_p2p():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_pipeline_p2p()
