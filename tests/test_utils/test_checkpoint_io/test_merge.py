from colossalai.utils.checkpoint_io.meta import ParamDistMeta
from colossalai.utils.checkpoint_io.constant import GLOBAL_META_FILE_NAME
from colossalai.utils.checkpoint_io.io import save, merge
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from tempfile import TemporaryDirectory
from torch.optim import Adam
from functools import partial
import torch
import os
import pytest
import colossalai
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


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


def test_merge_global():
    model, optimizer = prepare_model_optim()
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer)
        with TemporaryDirectory() as output_dir:
            merge(dir_name, output_dir)
            assert len(os.listdir(output_dir)) == 0
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer, max_shard_size_gb=80 / 1024**3)
        with TemporaryDirectory() as output_dir:
            merge(dir_name, output_dir)
            assert len(os.listdir(output_dir)) == 0


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
    dp_world_size = dist.get_world_size() // 2
    if not zero:
        dist_metas = {
            'fc.weight': ParamDistMeta(rank // 2, dp_world_size, rank % 2, 2, tp_shard_dims=[1], tp_num_parts=[2]),
            'fc.bias': ParamDistMeta(rank // 2, dp_world_size, 0, 1)
        }
    else:
        dist_metas = {
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
        }
    save(dir_name, model, optmizer, dist_meta=dist_metas)


@pytest.mark.dist
@pytest.mark.parametrize("zero", [False, True])
@rerun_if_address_is_in_use()
def test_merge_tp_dp(zero: bool):
    with TemporaryDirectory() as dir_name:
        fn = partial(run_save_dist, dir_name, zero)
        world_size = 4
        proc_fn = partial(run_dist, world_size=world_size, port=free_port(), func=fn)
        mp.spawn(proc_fn, nprocs=world_size)
        with TemporaryDirectory() as output_dir:
            merge(dir_name, output_dir)
            assert len(os.listdir(output_dir)) == 5
            global_meta = torch.load(os.path.join(output_dir, GLOBAL_META_FILE_NAME))
            assert len(global_meta['meta']) == 1
            meta = torch.load(os.path.join(output_dir, global_meta['meta'][0]))
            assert meta['dist_meta'] is None
            assert len(meta['params']) == 2
            assert len(meta['model']) == 1 and len(meta['optimizer']) == 1
            model_state_dict = torch.load(os.path.join(output_dir, meta['model'][0]))
            assert len(model_state_dict) == 2
            assert model_state_dict['fc.weight'].size(1) == 20
            optimizer_state_dict = torch.load(os.path.join(output_dir, meta['optimizer'][0]))
            assert len(optimizer_state_dict['state']) == 2
            assert 'param_groups' in optimizer_state_dict and 'state' in optimizer_state_dict
            assert optimizer_state_dict['state'][0]['exp_avg'].size(1) == 20
            assert optimizer_state_dict['state'][0]['exp_avg_sq'].size(1) == 20


if __name__ == '__main__':
    test_merge_global()
    test_merge_tp_dp(False)
    test_merge_tp_dp(True)
