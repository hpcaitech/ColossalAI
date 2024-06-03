import pytest
import torch

from colossalai.legacy.communication.p2p import (
    recv_backward,
    recv_forward,
    send_backward,
    send_backward_recv_forward,
    send_forward,
    send_forward_recv_backward,
)
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(pipeline=2))
torch.manual_seed(123)
LIST_LENGTH = 3
TENSOR_SIZE = torch.Size((3, 3))
TENSOR_SIZE_LIST = [TENSOR_SIZE for i in range(LIST_LENGTH)]
data = torch.rand(3, 3)
data_list = [torch.rand(3, 3) for i in range(LIST_LENGTH)]
grad = torch.rand(3, 3)
grad_list = [torch.rand(3, 3) for i in range(LIST_LENGTH)]


def check_send_recv_forward():
    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        device = torch.device("cuda:0")
        data_to_send = data.to(device)
        data_list_to_send = []
        for data_in_list in data_list:
            data_list_to_send.append(data_in_list.to(device))
        send_forward(data_to_send)
        send_forward(data_list_to_send)
    else:
        device = torch.device("cuda:1")
        data_recv = recv_forward(TENSOR_SIZE)
        data_list_recv = recv_forward(TENSOR_SIZE_LIST)
        data_to_check = data.to(device)
        assert data_recv.equal(data_to_check)
        for data_recv, data_send in zip(data_list_recv, data_list):
            data_to_check = data_send.to(device)
            assert data_recv.equal(data_to_check)


def check_send_recv_backward():
    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        device = torch.device("cuda:0")
        grad_recv = recv_backward(TENSOR_SIZE)
        grad_list_recv = recv_backward(TENSOR_SIZE_LIST)
        grad_to_check = grad.to(device)
        assert grad_recv.equal(grad_to_check)
        for grad_recv, grad_send in zip(grad_list_recv, grad_list):
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


def check_send_recv_forward_backward():
    if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
        device = torch.device("cuda:0")
        data_list_to_send = []
        for data_in_list in data_list:
            data_list_to_send.append(data_in_list.to(device))
        grad_list_recv = send_forward_recv_backward(data_list_to_send, TENSOR_SIZE_LIST)

        for grad_recv, grad_send in zip(grad_list_recv, grad_list):
            grad_to_check = grad_send.to(device)
            assert grad_recv.equal(grad_to_check)
    else:
        device = torch.device("cuda:1")
        grad_list_to_send = []
        for grad_in_list in grad_list:
            grad_list_to_send.append(grad_in_list.to(device))
        data_list_recv = send_backward_recv_forward(grad_list_to_send, TENSOR_SIZE_LIST)
        for data_recv, data_send in zip(data_list_recv, data_list):
            data_to_check = data_send.to(device)
            assert data_recv.equal(data_to_check)


def check_layer(rank, world_size, port):
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    check_send_recv_forward()
    check_send_recv_backward()
    check_send_recv_forward_backward()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_object_list_p2p():
    spawn(check_layer, 2)


if __name__ == "__main__":
    test_object_list_p2p()
