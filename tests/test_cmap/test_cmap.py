from colossalai.c_vmap import cmap

# from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn

from functools import partial
import torch
import pytest

cases = [{"raw_pt": True, "dst": -1},
         {"raw_pt": True, "dst": 0},
         {"raw_pt": False, "dst": -1},
         {"raw_pt": False, "dst": 0}]
disable_existing_loggers()
world_size = 8
CONFIG = dict(parallel=dict(pipeline=world_size))
torch.manual_seed(123)


def test_single_input():
    x = torch.randn(8*10, 10)
    for case in cases:
        f = lambda x: x*x
        c_f = cmap(f, **case)
        assert c_f(x) == x*x


def test_multiple_input():
    x = torch.randn(8*10, 10)
    y = torch.randn(8*10, 10)
    for case in cases:
        f = lambda x, y: x*y
        c_f = cmap(f, **case)
        assert c_f(x,y) == x*y


def test_nested():
    x = torch.randn(8*10, 8*10, 10)
    for case in cases:
        f = lambda x: x*x
        c_f = cmap(cmap(f, **case), **case)
        assert c_f(x) == x*x


def test_non_zero_in_dims():
    x = torch.randn(10, 8*10, 10)
    for case in cases:
        f = lambda x: x*x
        c_f = cmap(f, in_dims=1, **case)
        c_f(x)


def test_non_zero_out_dims():
    x = torch.randn(10, 8*10, 10)
    for case in cases:
        f = lambda x: x*x
        c_f = cmap(f, out_dims=1,**case)
        c_f(x)


def test_non_zero_in_out_dims():
    x = torch.randn(10, 8*10, 10)
    for case in cases:
        f = lambda x: x*x
        c_f = cmap(f, in_dims=1, out_dims=2, **case)
        c_f(x)


def test_multiple_in_multiple_out():
    x = torch.randn(10, 8*10, 10)
    y = torch.randn(10, 10, 8*10)
    for case in cases:
        def f(x,y):
            return x*y, x+y 
        c_f = cmap(f, in_dims=(1, 2), out_dims=(1, 2) **case)
        c_f(x, y)


def test_nn_module():
    x = torch.randn(8, 10)
    model = torch.nn.Linear(10, 10, bias=False)
    model.cuda()
    for case in cases:
        c_f = cmap(model, **case)
        c_f(x)


def test_functools_partial():
    x = torch.randn(10)
    y = torch.randn(8, 10)
    for case in cases:
        f = partial(torch.mul, x)
        c_f = cmap(f, **case)
        c_f(y)


def test_kwargs():
    def f(x, scale=1.0):
        return x*scale
    x = torch.randn(8*10, 10)
    for case in cases:
        c_f = cmap(f, **case)
        c_f(x, scale=8.0)


def check_vmap(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl', verbose=False)

    test_single_input()
    test_multiple_input()
    test_nested()
    test_non_zero_in_dims()
    test_non_zero_out_dims()
    test_nn_module()
    test_functools_partial()
    test_kwargs()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_cmap():
    spawn(check_vmap, world_size)


if __name__ == '__main__':
    test_cmap()
