from copy import deepcopy
from functools import partial
from tempfile import TemporaryDirectory
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer

import colossalai
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.checkpoint_io.io import load, save
from colossalai.utils.checkpoint_io.meta import ParamDistMeta, ParamRedistMeta, RankRedistMeta, RedistMeta


def check_model_state_dict(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> None:
    assert set(a.keys()) == set(b.keys())
    for k, v in a.items():
        assert torch.equal(v, b[k])


def check_optim_state_dict(a: dict, b: dict, ignore_param_gruops: bool = False) -> None:
    assert set(a['state'].keys()) == set(b['state'].keys())
    for k, state in a['state'].items():
        b_state = b['state'][k]
        for v1, v2 in zip(state.values(), b_state.values()):
            if isinstance(v1, Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2
    if not ignore_param_gruops:
        assert a['param_groups'] == b['param_groups']


class DummyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(20, 1)


def prepare_model_optim(shard: bool = False, zero: bool = False):
    model = DummyModel()
    if shard:
        model.fc.weight.data = model.fc.weight.chunk(2, 1)[dist.get_rank() % 2]
    if zero:
        dp_rank = dist.get_rank() // 2
        model.fc.weight.data = model.fc.weight.reshape(-1).split([3, model.fc.weight.size(1) - 3], 0)[dp_rank]
        if dp_rank != 0:
            model.fc.bias.data = torch.empty(0, dtype=model.fc.bias.dtype)
    for p in model.parameters():
        p.grad = torch.rand_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    return model, optimizer


def reset_model_optim(model: Module, optimizer: Optimizer, scalar: float = 0.0):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(scalar)
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, Tensor):
                    v.fill_(scalar)


def get_dist_metas(nprocs: int, zero: bool = False):
    dp_world_size = nprocs // 2
    dist_metas = []
    for rank in range(nprocs):
        if zero:
            dist_metas.append({
                'fc.weight':
                    ParamDistMeta(rank // 2,
                                  dp_world_size,
                                  rank % 2,
                                  2,
                                  tp_shard_dims=[1],
                                  tp_num_parts=[2],
                                  zero_numel=10,
                                  zero_orig_shape=[1, 10]),
                'fc.bias':
                    ParamDistMeta(rank // 2, dp_world_size, 0, 1, zero_numel=1, zero_orig_shape=[1])
            })
        else:
            dist_metas.append({
                'fc.weight': ParamDistMeta(rank // 2, dp_world_size, rank % 2, 2, tp_shard_dims=[1], tp_num_parts=[2]),
                'fc.bias': ParamDistMeta(rank // 2, dp_world_size, 0, 1)
            })
    return dist_metas


def get_redist_meta(nprocs: int):
    dp_world_size = nprocs // 2
    rank_meta = {
        'fc.weight': {rank: RankRedistMeta(rank // 2, rank % 2, 0) for rank in range(nprocs)},
        'fc.bias': {rank: RankRedistMeta(rank // 2, 0, 0) for rank in range(nprocs)}
    }
    param_meta = {
        'fc.weight': ParamRedistMeta(dp_world_size, 2, tp_shard_dims=[1], tp_num_parts=[2]),
        'fc.bias': ParamRedistMeta(dp_world_size, 1)
    }
    return RedistMeta(rank_meta, [], param_meta)


@pytest.mark.parametrize('max_shard_size_gb', [80 / 1024**3, 0])
def test_save_global_load_global(max_shard_size_gb: float):
    model, optimizer = prepare_model_optim()
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer, max_shard_size_gb=max_shard_size_gb)
        new_model, new_optimizer = prepare_model_optim()
        load(dir_name, new_model, new_optimizer, max_shard_size_gb=max_shard_size_gb)
        check_model_state_dict(model.state_dict(), new_model.state_dict())
        check_optim_state_dict(optimizer.state_dict(), new_optimizer.state_dict())


def run_dist(rank, world_size, port, test_fn):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_fn()


def launch_dist(fn, world_size: int):
    spawn(run_dist, world_size, test_fn=fn)


def save_dist(dir_name: str, zero: bool):
    model, optmizer = prepare_model_optim(shard=True, zero=zero)
    reset_model_optim(model, optmizer)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    save(dir_name, model, optmizer, dist_meta=get_dist_metas(world_size, zero)[rank])


def load_and_check_dist(dir_name: str):
    world_size = dist.get_world_size()
    model, optmizer = prepare_model_optim(shard=True)
    reset_model_optim(model, optmizer)
    model_state_dict = deepcopy(model.state_dict())
    optimizer_state_dict = deepcopy(optmizer.state_dict())
    reset_model_optim(model, optmizer, 1)
    load(dir_name, model, optmizer, get_redist_meta(world_size), get_dist_metas(world_size))
    check_model_state_dict(model_state_dict, model.state_dict())
    check_optim_state_dict(optimizer_state_dict, optmizer.state_dict())


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_save_global_load_dist():
    model, optimizer = prepare_model_optim()
    reset_model_optim(model, optimizer)
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer)
        fn = partial(load_and_check_dist, dir_name)
        launch_dist(fn, 4)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_save_dist_load_dist():
    with TemporaryDirectory() as dir_name:
        # save tp + dp
        fn = partial(save_dist, dir_name, False)
        launch_dist(fn, 2)
        # load tp + dp
        fn = partial(load_and_check_dist, dir_name)
        launch_dist(fn, 2)
    with TemporaryDirectory() as dir_name:
        # save tp + zero
        fn = partial(save_dist, dir_name, True)
        launch_dist(fn, 4)
        # load tp + dp
        fn = partial(load_and_check_dist, dir_name)
        launch_dist(fn, 2)
        launch_dist(fn, 4)


if __name__ == '__main__':
    test_save_global_load_global(80 / 1024**3)
    test_save_global_load_global(0)
    test_save_global_load_dist()
    test_save_dist_load_dist()
