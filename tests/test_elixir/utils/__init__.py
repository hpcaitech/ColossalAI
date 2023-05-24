import torch
from torch.testing import assert_close
from torch.utils._pytree import tree_map

from . import gpt, mlp, opt, resnet, small
from .registry import TEST_MODELS


def to_cuda(input_dict):

    def local_fn(t):
        if isinstance(t, torch.Tensor):
            t = t.cuda()
        return t

    ret = tree_map(local_fn, input_dict)
    return ret


def allclose(ta, tb, **kwargs):
    assert_close(ta, tb, **kwargs)
    return True


def assert_dict_keys(test_dict, keys):
    assert len(test_dict) == len(keys)
    for k in keys:
        assert k in test_dict


def assert_dict_values(da, db, fn):
    assert len(da) == len(db)
    for k, v in da.items():
        assert k in db
        if not torch.is_tensor(v):
            continue
        u = db.get(k)
        if u.device != v.device:
            v = v.to(u.device)
        # print(f"checking key {k}: {u.shape} vs {v.shape}")
        assert fn(u.data, v.data), f'max diff {torch.max(torch.abs(u.data - v.data))}'
