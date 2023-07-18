import pytest
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from colossalai.core import global_context as gpc
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_layer(rank, world_size, port):
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    physical_mesh_id = torch.arange(0, 4)
    assert rank == gpc.get_global_rank()

    tensor_to_check = torch.tensor([2, 2, 2, 2]).cuda()
    mesh_shape = (2, 2)
    # [[0, 1,
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    for axis in range(len(mesh_shape)):
        tensor = torch.ones(4).cuda()
        pg = device_mesh.get_process_group(axis=axis)
        dist.all_reduce(tensor, op=ReduceOp.SUM, group=pg)
        assert tensor.equal(tensor_to_check)

    gpc.destroy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_logical_pg():
    spawn(check_layer, 4)


if __name__ == '__main__':
    test_logical_pg()
