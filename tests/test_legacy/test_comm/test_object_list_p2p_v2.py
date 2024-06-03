import pytest
import torch

from colossalai.legacy.communication.p2p_v2 import recv_backward, recv_forward, send_backward, send_forward
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn

disable_existing_loggers()

# config
world_size = 4
CONFIG = dict(parallel=dict(pipeline=4))
torch.manual_seed(123)
use_scatter_gather_tensors = False

# data
torch.manual_seed(123)
LIST_LENGTH = 3
TENSOR_SIZE = torch.Size((3, 3))
TENSOR_SIZE_LIST = [TENSOR_SIZE for i in range(LIST_LENGTH)]
data = torch.rand(3, 3)
data_list = [torch.rand(3, 3) for i in range(LIST_LENGTH)]
grad = torch.rand(3, 3)
grad_list = [torch.rand(3, 3) for i in range(LIST_LENGTH)]


def check_send_recv_forward():
    disable_existing_loggers()
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    if local_rank == 0:
        device = torch.device("cuda:0")
        data_to_send = data.to(device)
        data_list_to_send = []
        for data_in_list in data_list:
            data_list_to_send.append(data_in_list.to(device))

        send_forward(data_to_send, scatter_gather_tensors=use_scatter_gather_tensors)
        send_forward(data_list_to_send, scatter_gather_tensors=use_scatter_gather_tensors)

    elif local_rank == 1:
        device = torch.device("cuda:1")

        data_recv = recv_forward(TENSOR_SIZE, scatter_gather_tensors=use_scatter_gather_tensors)
        data_list_recv = recv_forward(TENSOR_SIZE_LIST, scatter_gather_tensors=use_scatter_gather_tensors)

        data_to_check = data.to(device)

        assert data_recv.equal(data_to_check)

        for data_recv, data_send in zip(data_list_recv, data_list):
            data_to_check = data_send.to(device)
            data_recv = data_recv.to(device)
            assert data_recv.equal(data_to_check)


def check_send_recv_backward():
    disable_existing_loggers()
    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        device = torch.device("cuda:0")
        grad_recv = recv_backward(TENSOR_SIZE)
        grad_list_recv = recv_backward(TENSOR_SIZE_LIST)

        grad_to_check = grad.to(device)
        grad_recv = grad_recv[0].to(device)

        assert grad_recv.equal(grad_to_check)
        for grad_recv, grad_send in zip(grad_list_recv, grad_list):
            grad_recv = grad_recv.to(device)
            grad_to_check = grad_send.to(device)
            assert grad_recv.equal(grad_to_check)
    else:
        device = torch.device("cuda:1")
        grad_to_send = grad.to(device)
        grad_list_to_send = []
        for grad_in_list in grad_list:
            grad_list_to_send.append(grad_in_list.to(device))
        send_backward(grad_to_send)
        send_backward(grad_list_to_send)


def check_small_pipeline():
    disable_existing_loggers()
    # make sure the rank is 4
    assert gpc.world_size == 4, "make sure to set world size to 4 to start the training process"
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    if local_rank == 0:
        obj = [1, torch.randn(2, 2).cuda(), None]
        send_forward(obj)
    elif local_rank == 1:
        obj = recv_forward()
        send_forward(obj)
    elif local_rank == 2:
        obj = recv_forward()
        send_forward(obj)
    elif local_rank == 3:
        obj = recv_forward()
    else:
        pass


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    disable_existing_loggers()
    # check_send_recv_forward()
    check_small_pipeline()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_object_list_p2p():
    spawn(check_layer, world_size)


if __name__ == "__main__":
    disable_existing_loggers()
    test_object_list_p2p()
