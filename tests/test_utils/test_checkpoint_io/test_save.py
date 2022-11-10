import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Dict

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.checkpoint_io.constant import (GLOBAL_META_FILE_NAME, META_CKPT_FILE_NAME, MODEL_CKPT_FILE_NAME,
                                                     OTHER_CKPT_FILE_NAME)
from colossalai.utils.checkpoint_io.io import save
from colossalai.utils.checkpoint_io.meta import ParamDistMeta
from torch import Tensor
from torch.optim import Adam


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


def prepare_model_optim():
    model = DummyModel()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    return model, optimizer


def test_overwrite():
    model = DummyModel()
    with TemporaryDirectory() as dir_name:
        with open(os.path.join(dir_name, MODEL_CKPT_FILE_NAME.replace('.bin', '-shard0.bin')), 'a') as f:
            pass
        with pytest.raises(RuntimeError, match=r'Save error: Checkpoint ".+" exists\. \(overwrite = False\)'):
            save(dir_name, model)


def test_save_global():
    model, optimizer = prepare_model_optim()
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer)
        assert len(os.listdir(dir_name)) == 5
        global_meta = torch.load(os.path.join(dir_name, GLOBAL_META_FILE_NAME))
        assert len(global_meta['meta']) == 1 and global_meta['meta'][0] == META_CKPT_FILE_NAME
        meta = torch.load(os.path.join(dir_name, META_CKPT_FILE_NAME))
        assert len(meta['model']) == 1
        assert len(meta['optimizer']) == 1
        model_state_dict = torch.load(os.path.join(dir_name, meta['model'][0]))
        check_model_state_dict(model.state_dict(), model_state_dict)
        optimizer_state_dict = torch.load(os.path.join(dir_name, meta['optimizer'][0]))
        check_optim_state_dict(optimizer.state_dict(), optimizer_state_dict)
        other_state_dict = torch.load(os.path.join(dir_name, OTHER_CKPT_FILE_NAME))
        assert len(other_state_dict) == 0


def test_save_global_shard():
    model, optimizer = prepare_model_optim()
    with TemporaryDirectory() as dir_name:
        save(dir_name, model, optimizer, max_shard_size_gb=80 / 1024**3)
        assert len(os.listdir(dir_name)) == 7
        meta = torch.load(os.path.join(dir_name, META_CKPT_FILE_NAME))
        assert len(meta['model']) == 2 and len(meta['optimizer']) == 2
        model_state_dicts = [torch.load(os.path.join(dir_name, name)) for name in meta['model']]
        assert len(set(model_state_dicts[0].keys()) & set(model_state_dicts[1].keys())) == 0
        check_model_state_dict(model.state_dict(), {**model_state_dicts[0], **model_state_dicts[1]})
        optimizer_state_dicts = [torch.load(os.path.join(dir_name, name)) for name in meta['optimizer']]
        assert len(set(optimizer_state_dicts[0]['state'].keys()) & set(optimizer_state_dicts[1]['state'].keys())) == 0
        assert 'param_groups' in optimizer_state_dicts[0] and 'param_groups' not in optimizer_state_dicts[1]
        check_optim_state_dict(
            optimizer.state_dict(), {
                'state': {
                    **optimizer_state_dicts[0]['state'],
                    **optimizer_state_dicts[1]['state']
                },
                'param_groups': optimizer_state_dicts[0]['param_groups']
            })


def run_dist(rank, world_size, port, func):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    func()


def run_save_dist(dir_name):
    model, optmizer = prepare_model_optim()
    dist_metas = {
        'fc.weight': ParamDistMeta(dist.get_rank(), dist.get_world_size(), 0, 1),
        'fc.bias': ParamDistMeta(dist.get_rank(), dist.get_world_size(), 0, 1)
    }
    save(dir_name, model, optmizer, dist_meta=dist_metas)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_save_dist():
    with TemporaryDirectory() as dir_name:
        fn = partial(run_save_dist, dir_name)
        world_size = 2
        proc_fn = partial(run_dist, world_size=world_size, port=free_port(), func=fn)
        mp.spawn(proc_fn, nprocs=world_size)
        assert len(os.listdir(dir_name)) == 8
        global_meta = torch.load(os.path.join(dir_name, GLOBAL_META_FILE_NAME))
        assert len(global_meta['meta']) == 2
        for rank, meta_name in enumerate(global_meta['meta']):
            meta = torch.load(os.path.join(dir_name, meta_name))
            assert meta.get('dist_meta', None) is not None
            assert len(meta['model']) == 1 and len(meta['optimizer']) == 1
            model_state_dict = torch.load(os.path.join(dir_name, meta['model'][0]))
            assert len(model_state_dict) == 2
            optimizer_state_dict = torch.load(os.path.join(dir_name, meta['optimizer'][0]))
            assert len(optimizer_state_dict['state']) == 2
            assert 'param_groups' in optimizer_state_dict


if __name__ == '__main__':
    test_overwrite()
    test_save_global()
    test_save_global_shard()
    test_save_dist()
