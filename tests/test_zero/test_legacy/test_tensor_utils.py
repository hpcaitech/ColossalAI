import pytest
import torch

import colossalai
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero.legacy.gemini.stateful_tensor import StatefulTensor
from colossalai.zero.legacy.gemini.tensor_utils import (
    colo_model_data_move_to_cpu,
    colo_model_data_tensor_move,
    colo_model_data_tensor_move_inline,
    colo_model_tensor_clone,
    colo_tensor_mem_usage,
)


def _run_colo_tensor_mem_usage():
    for i in range(1):
        if i == 1:
            t1 = StatefulTensor(torch.randn(2, 2))
            t2 = StatefulTensor(torch.randn(4, 4))
            c1, g1 = colo_tensor_mem_usage(t1)
            c2, g2 = colo_tensor_mem_usage(t2)
            assert c1 * 4 == c2
            assert g1 * 4 == g2
        else:
            t1 = torch.randn(2, 2)
            t2 = torch.randn(4, 4)
            c1, g1 = colo_tensor_mem_usage(t1)
            c2, g2 = colo_tensor_mem_usage(t2)
            assert c1 * 4 == c2
            assert g1 * 4 == g2


def _run_colo_model_data_tensor_move_inline():
    for t in [StatefulTensor(torch.randn(2, 3)), torch.randn(2, 3)]:
        colo_model_data_tensor_move_inline(t, get_current_device())
        assert t.device == get_current_device()


def _run_colo_model_data_tensor_move():
    for t in [(StatefulTensor(torch.ones(2, 3)), StatefulTensor(torch.zeros(2, 3).to(get_current_device()))),
              (torch.ones(2, 3), torch.zeros(2, 3).to(get_current_device()))]:
        cpu_t, cuda_t = t
        colo_model_data_tensor_move(cpu_t, cuda_t)
        assert cuda_t.device == get_current_device()


def _run_colo_model_data_move_to_cpu():
    for t in [StatefulTensor(torch.randn(2, 2)), torch.randn(4, 4)]:
        colo_model_data_move_to_cpu(t)
        assert t.device == torch.device("cpu")


def _run_colo_model_tensor_clone():
    for t in [
            StatefulTensor(torch.randn(2, 2).cuda(torch.cuda.current_device())),
            torch.randn(4, 4).cuda(torch.cuda.current_device())
    ]:
        if issubclass(type(t), StatefulTensor):
            assert t.payload.device == get_current_device()
        else:
            assert t.device == get_current_device()
        p = colo_model_tensor_clone(t, get_current_device())
        assert p.device == get_current_device()
        for i in range(2):
            for j in range(2):
                if issubclass(type(t), StatefulTensor):
                    assert t.payload.device == p.device
                    assert t.payload[i][j] == p[i][j]
                else:
                    assert t.device == p.device
                    assert t[i][j] == p[i][j]


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    _run_colo_tensor_mem_usage()
    _run_colo_model_data_tensor_move_inline()
    _run_colo_model_data_tensor_move()
    _run_colo_model_data_move_to_cpu()
    _run_colo_model_tensor_clone()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_zero_tensor_utils(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_tensor_utils(world_size=2)
