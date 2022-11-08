import os
from functools import partial
from tempfile import TemporaryDirectory

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.checkpoint_io.constant import GLOBAL_META_FILE_NAME
from colossalai.utils.checkpoint_io.io import redist, save
from colossalai.utils.checkpoint_io.meta import (ParamDistMeta, ParamRedistMeta, PipelineRedistMeta, RankRedistMeta,
                                                 RedistMeta)
from torch.optim import Adam


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
        p.grad = torch.ones_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    return model, optimizer


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


def check_checkpoint_shape(dir_name: str):
    global_meta = torch.load(os.path.join(dir_name, GLOBAL_META_FILE_NAME))
    for meta_name in global_meta['meta']:
        meta = torch.load(os.path.join(dir_name, meta_name))
        assert meta['dist_meta'] is not None
        assert len(meta['params']) == 2
        assert len(meta['model']) == 1 and len(meta['optimizer']) == 1
        model_state_dict = torch.load(os.path.join(dir_name, meta['model'][0]))
        assert len(model_state_dict) == 2
        assert model_state_dict['fc.weight'].size(1) == 10
        optimizer_state_dict = torch.load(os.path.join(dir_name, meta['optimizer'][0]))
        assert len(optimizer_state_dict['state']) == 2
        assert 'param_groups' in optimizer_state_dict and 'state' in optimizer_state_dict
        assert optimizer_state_dict['state'][0]['exp_avg'].size(1) == 10
        assert optimizer_state_dict['state'][0]['exp_avg_sq'].size(1) == 10


def test_global_to_dist():
    model, optimizer = prepare_model_optim()
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer)
        with TemporaryDirectory() as output_dir:
            redist(dir_name, output_dir, get_redist_meta(4), get_dist_metas(4))
            check_checkpoint_shape(output_dir)


def run_dist(rank, world_size, port, func):
    colossalai.launch(config={'parallel': {
        'tensor': {
            'mode': '1d',
            'size': 2
        }
    }},
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')
    func()


def run_save_dist(dir_name: str, zero: bool):
    model, optmizer = prepare_model_optim(shard=True, zero=zero)
    rank = dist.get_rank()
    save(dir_name, model, optmizer, dist_meta=get_dist_metas(4, zero)[rank])


@pytest.mark.dist
@pytest.mark.parametrize("zero", [False, True])
@rerun_if_address_is_in_use()
def test_dist_to_dist(zero: bool):
    with TemporaryDirectory() as dir_name:
        fn = partial(run_save_dist, dir_name, zero)
        world_size = 4
        proc_fn = partial(run_dist, world_size=world_size, port=free_port(), func=fn)
        mp.spawn(proc_fn, nprocs=world_size)
        with TemporaryDirectory() as output_dir:
            redist(dir_name, output_dir, get_redist_meta(4), get_dist_metas(4))
            if not zero:
                assert len(os.listdir(output_dir)) == 0
            else:
                check_checkpoint_shape(output_dir)


if __name__ == '__main__':
    test_global_to_dist()
    test_dist_to_dist(False)
    test_dist_to_dist(True)
